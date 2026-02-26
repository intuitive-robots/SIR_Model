import numpy as np
from torch import nn
import torch
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.norm import LayerNorm
import torch.nn.functional as F

from networks.graph_encoder.utils import debatch_graphs_masks, soft_histogram_loss, soft_histrogram_except_middle_loss
from networks.transformer.transformer_decoders import TransformerFiLMDecoder
from utils.generate_graph_dataset_robocasa import get_num_relevant_nodes_per_task

class Multi_GNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 edge_dim,
                 num_layer,
                 layer_name,
                 pool_name,
                 heads,
                 dropout,
                 modalities,
                ) -> None:
        super(Multi_GNN, self).__init__()
        
        self.models = nn.ModuleDict()
        for mod in modalities:
            self.models[mod] = GNN(input_dim[mod], hidden_dim, output_dim, edge_dim, num_layer, layer_name, pool_name, heads, dropout)

    def forward(self, input):
        for key in input:
            input[key] = self.models[key](input[key])
        return input
    
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, num_layer, layer_name, pool_name, heads, dropout) -> None:
        super(GNN, self).__init__()
        
        if layer_name == "GATv2":
            self.layer_class = GATv2Conv
        else:
            raise ValueError(f"Unsupported GNN layer: {layer_name}")
        
        if pool_name == "mean":
            self.pooling = global_mean_pool
        elif pool_name == "add":
            self.pooling = global_add_pool
        elif pool_name == "max":
            self.pooling = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pool_name}")
        
        if layer_name == "GATv2":
            self.input_layer = self.layer_class(input_dim, hidden_dim, edge_dim=edge_dim, heads=heads, dropout=dropout, concat=False) # TODO check concat again
            
        self.input_norm = LayerNorm(hidden_dim)
        self.input_projection_layer = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            if layer_name == "GATv2":
                self.layers.append(self.layer_class(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=heads, dropout=dropout, concat=False))
                
        self.norms = nn.ModuleList()
        for _ in range(num_layer):
            self.norms.append(LayerNorm(hidden_dim))
            
        self.output = nn.Linear(hidden_dim, output_dim)
        
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Initialize weights of Linear and Embedding layers
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, GATv2Conv):
            # Initialize GATv2Conv layers (attention and projections)
            if hasattr(module, 'att'):
                nn.init.xavier_uniform_(module.att, gain=1.0)  # Attention coefficients
            
            # GATv2Conv typically has its projection weights internally as Linear layers.
            # Use `apply` recursively to initialize them.
            if hasattr(module, 'lin_l'):
                nn.init.xavier_uniform_(module.lin_l.weight, gain=1.0)
                if module.lin_l.bias is not None:
                    nn.init.zeros_(module.lin_l.bias)
            if hasattr(module, 'lin_r'):
                nn.init.xavier_uniform_(module.lin_r.weight, gain=1.0)
                if module.lin_r.bias is not None:
                    nn.init.zeros_(module.lin_r.bias)
        
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm layers
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, batch):
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_attr
        edge_index = edge_index.to(torch.int64)
        
        # Input projection
        input = self.input_layer(x, edge_index, edge_weight)
        input = self.leaky_relu(self.input_norm(self.input_projection_layer(x) + input))
        if self.training:
            input = self.dropout(input)
        
        x = input
        # Iterating through the layers
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, edge_index, edge_weight)
            x = self.leaky_relu(norm(x + residual))
            if self.training:
                x = self.dropout(x)
                
        output = x
        # Pooling and projection to create the final embedding
        pooling = self.pooling(output, batch.batch)
        
        embedding = self.output(pooling)
        
        return embedding
    
class Sparsification_Module(nn.Module):
    def __init__(self, 
                 coars_type,
                 in_dim,
                 hidden_dim,
                 num_heads,
                 num_layers,
                 dropout_prob=0.2,
                 sampling_strategy="topk_PerTask", 
                 differentiable_dropping_nodes=False,
                 scoring_strategy="sigmoid", 
                 sample_abs: int=None,
                 gamma=-0.1,
                 zeta=1.1,
                 beta=0.66, 
                 l0_reg_weight=1.0,
                 variance_weight=0.0,
                 sum_mask_weight=0.0,
                 target_sum_node_probs=0.0,
                 uniform_dist_weight=0.0,
                 uniform_except_middle_weight=0.0,
                 uniform_except_middle_offset=0.0,
                 threshold_sum_weight=0.0):
        super().__init__()

        self.scoring_strategy = scoring_strategy
        self.sampling_strategy = sampling_strategy
        self.differentiable_dropping = differentiable_dropping_nodes
        self.needs_seperate_optimizer = False
        self.coarsening_loss = dict()

        if coars_type in ["TransformerCoarsening", "TransformerFiLMDecoder"]:
            self.coarsening_layer = TransformerCoarsening(in_dim, hidden_dim, num_heads, num_layers, dropout_prob, coars_type)
            self.requires_debatched_graphs = True
        elif coars_type in ["Random", "RandomButConsistentDuringRollout"] or sampling_strategy == "Random":
            self.sampling_strategy = "Random"
            self.coarsening_layer = None
            self.requires_debatched_graphs = True
        
        self.sample_abs = sample_abs
        self.coars_type = coars_type

        self.sig = nn.Sigmoid()
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        self.eps = 1e-20
        self.const1 = self.beta*np.log(-self.gamma/self.zeta + self.eps)
        self.l0_reg_weight = l0_reg_weight

        self.variance_weight = variance_weight
        self.sum_mask_weight = sum_mask_weight
        self.target_sum_node_probs = target_sum_node_probs
        self.uniform_dist_weight = uniform_dist_weight
        self.uniform_except_middle_weight = uniform_except_middle_weight
        self.uniform_except_middle_offset = uniform_except_middle_offset
        self.threshold_sum_weight = threshold_sum_weight

    def l0_train(self, logAlpha, min, max):
        U = torch.rand(logAlpha.size()).type_as(logAlpha) + self.eps
        s = self.sig((torch.log(U / (1 - U)) + logAlpha) / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = F.hardtanh(s_bar, min, max)
        return mask

    def l0_test(self, logAlpha, min, max):
        s = self.sig(logAlpha/self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = F.hardtanh(s_bar, min, max)
        return mask

    def get_loss2(self, logAlpha):
        return self.sig(logAlpha - self.const1)

    def forward(self, x, edge_index=None, edge_attr=None, edge_weights=None, lang_emb=None, batch=None, task_names=None):

        self.coarsening_loss = dict()

        node_mask, edge_mask, num_graphs = debatch_graphs_masks(x, None, batch)
        
        scores = self.coarsening_layer(x, edge_index, edge_attr, node_mask, lang_emb, num_graphs)

        if self.scoring_strategy == "sigmoid":
            probs = torch.sigmoid(scores)
        elif self.scoring_strategy == "minmax":
            probs = torch.zeros_like(scores, dtype=torch.float32, device=x.device)
            for i in range(num_graphs):
                probs[node_mask == i] = (scores[node_mask == i] - scores[node_mask == i].min()) / (scores[node_mask == i].max() - scores[node_mask == i].min() + 1e-8)
        elif self.scoring_strategy == "l0_reg":
            if self.training:
                probs = self.l0_train(scores, 0, 1)
                if "coars_l0_reg" in self.coarsening_loss:
                    self.coarsening_loss['coars_l0_reg']['value'] += self.get_loss2(scores).mean()
                else:
                    self.coarsening_loss['coars_l0_reg'] = {'value': self.get_loss2(scores).mean(), 'weight': self.l0_reg_weight}
            else:
                probs = self.l0_test(scores, 0, 1)
        else:
            raise NotImplementedError(f"Scoring strategy '{self.scoring_strategy}' is not implemented.")

        assert (probs != probs).sum() == 0, "NaN values in probabilities after coarsening layer."

        if self.training or (hasattr(self, "not_rollout") and self.not_rollout):
            for i in range(num_graphs):
                
                l = -torch.var(probs[node_mask == i], unbiased=False) / num_graphs
                if 'coars_variance_loss' in self.coarsening_loss:
                    self.coarsening_loss['coars_variance_loss']['value'] += l
                else:
                    self.coarsening_loss['coars_variance_loss'] = {'value': l, 'weight': self.variance_weight}
                l = (probs[node_mask==i].sum() - self.target_sum_node_probs).abs() / num_graphs
                if 'coars_sum_mask_loss' in self.coarsening_loss:
                    self.coarsening_loss['coars_sum_mask_loss']['value'] += l
                else:
                    self.coarsening_loss['coars_sum_mask_loss'] = {'value': l, 'weight': self.sum_mask_weight}
    
                l = (torch.sum(probs[node_mask==i] * torch.log(probs[node_mask==i] + 1e-20))) / num_graphs
                if 'coars_entropy_loss' in self.coarsening_loss:
                    self.coarsening_loss['coars_entropy_loss']['value'] += l
                else:
                    self.coarsening_loss['coars_entropy_loss'] = {'value': l, 'weight': self.entropy_reg_weight}
            
                l = soft_histogram_loss(probs[node_mask == i]) / num_graphs
                if 'coars_uniform_loss' in self.coarsening_loss:
                    self.coarsening_loss['coars_uniform_loss']['value'] += l
                else:
                    self.coarsening_loss['coars_uniform_loss'] = {'value': l, 'weight': self.uniform_dist_weight}

                l = soft_histrogram_except_middle_loss(probs[node_mask == i], offset=self.uniform_except_middle_offset, scores_in_upper=self._calc_k_for_topk((node_mask == i).sum().item(), task_name=task_names[i] if task_names else None)) / num_graphs
                if 'coars_uniform_except_middle_loss' in self.coarsening_loss:
                    self.coarsening_loss['coars_uniform_except_middle_loss']['value'] += l
                else:
                    self.coarsening_loss['coars_uniform_except_middle_loss'] = {'value': l, 'weight': self.uniform_except_middle_weight}

        if self.sampling_strategy in ["topk", "topk_fix", "topk_PerTask"]:
            probs_new = torch.zeros_like(scores, dtype=torch.float32, device=x.device)
            for i in range(num_graphs):
                current_node_mask = node_mask == i
                top_k_values, top_k_indices = torch.topk(probs[current_node_mask], self._calc_k_for_topk(current_node_mask.sum().item(), task_name=task_names[i] if task_names else None), dim=-1, largest=True, sorted=False)

                full_indices = current_node_mask.nonzero(as_tuple=False).squeeze(1)[top_k_indices]
                probs_new[full_indices] = top_k_values

            indexes = ((probs_new != 0).to(probs_new.dtype) - probs).detach() + probs
            probs_new = probs * indexes

            probs = probs_new
            if edge_weights is not None:
                edge_weights = edge_weights * probs[edge_index[0]] * probs[edge_index[1]]
            sampled_nodes = (probs != 0).nonzero(as_tuple=False).squeeze(-1)
        else:
            raise NotImplementedError(f"Sampling strategy '{self.sampling_strategy}' is not implemented.")
        
        return edge_weights, sampled_nodes, probs

    def _calc_k_for_topk(self, num_nodes_in_graph, task_name=None):
        if self.sampling_strategy == "topk":
            return max(min(self.sample_abs, num_nodes_in_graph - 2), 2)
        elif self.sampling_strategy == "topk_fix":
            return min(self.sample_abs, num_nodes_in_graph)
        elif self.sampling_strategy == "topk_PerTask":
            k = get_num_relevant_nodes_per_task(task_name)
            return min(k, num_nodes_in_graph)
        elif self.sampling_strategy == "topk_NN":
            raise NotImplementedError()
            if task not in self.TopkKNN_values:
                k = self.TopkKNN(language_goal)
                self.TopkKNN_values[task] = k
                self.coarsening_loss['coars_topk_K_NN_loss'] = {'value': k, 'weight': self.topk_K_NN_loss_weight}
            else:
                k = self.TopkKNN_values[task]

            return min(k, num_nodes_in_graph)
        else:
            return min(self.sample_abs, num_nodes_in_graph)
        
class TransformerCoarsening(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout_prob=0.2, coars_type="TransformerCoarsening"):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        if coars_type == "TransformerCoarsening":
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_prob, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif coars_type == "TransformerFiLMDecoder":
            self.encoder = TransformerFiLMDecoder(
                embed_dim=embed_dim,
                n_heads=num_heads,
                attn_pdrop=dropout_prob,
                resid_pdrop=dropout_prob,
                n_layers=num_layers,
                block_size = 128,
                film_cond_dim = embed_dim)        
        else:
            raise ValueError(f"Unknown coarsening type: {coars_type}")

        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 1)
        )

        self.coars_type = coars_type

    def forward(self, x, edge_index=None, edge_attr=None, node_mask=None, lang_emb=None, num_graphs=None):
       
        x = self.embedding(x)  # [N, D]

        for i in range(num_graphs):

            assert (x != x).sum() == 0, "NaN-Values in x."
            
            if self.coars_type == "TransformerFiLMDecoder":                
                x[node_mask==i] = self.encoder(x[node_mask==i].unsqueeze(0), lang_emb[i]).squeeze(0)
            else:
                x[node_mask==i] = self.encoder(x[node_mask==i].unsqueeze(0)).squeeze(0)  # [N, D]
        
        scores = self.scorer(x).squeeze(-1)  # [N]

        assert (scores != scores).sum() == 0, "NaN-Values in scores."

        return scores