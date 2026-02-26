import torch
import torch.nn.functional as F

def soft_histogram_loss(scores, num_bins=10, sigma=0.05):
    # Bin centers
    device = scores.device
    bin_centers = torch.linspace(0, 1, steps=num_bins, device=device)

    # Compute soft assignment of each score to bins
    diffs = scores.unsqueeze(1) - bin_centers.unsqueeze(0)  # [N, num_bins]
    weights = torch.exp(-0.5 * (diffs / sigma) ** 2)        # Gaussian kernel
    hist = weights.sum(dim=0)
    hist = hist / hist.sum()

    target = torch.full_like(hist, 1.0 / num_bins)
    return F.mse_loss(hist, target)

def soft_histrogram_except_middle_loss(scores, num_bins=10, sigma=0.05, offset=0.1, scores_in_upper=5):
    
    device = scores.device
    bin_centers = torch.cat((torch.linspace(0, offset, steps=num_bins // 2, device=device), torch.linspace(1-offset, 1, steps=num_bins // 2, device=device)))

    # Compute soft assignment of each score to bins
    diffs = scores.unsqueeze(1) - bin_centers.unsqueeze(0)  # [N, num_bins]
    weights = torch.exp(-0.5 * (diffs / sigma) ** 2)        # Gaussian kernel
    hist = weights.sum(dim=0)
    hist = hist / hist.sum()

    target = torch.cat((
         torch.full((num_bins // 2, ), max(len(scores) - scores_in_upper, 0), device=device), 
         torch.full((num_bins // 2, ), min(scores_in_upper, len(scores)), device=device)
         ))
    target = target / target.sum()

    return F.mse_loss(hist, target)

def debatch_graphs_masks(x, edge_index, batch):

    if "batch" not in list(batch.keys()):
        batch.batch = torch.zeros(x.shape[0], device=x.device, dtype=int)
        batch.num_graphs = 1

    if "num_graphs" not in batch.keys():
        batch.num_graphs = batch.batch.max().item() + 1

    batch_index = batch.batch
    num_graphs = batch.num_graphs

    if edge_index is not None:
        edge_mask = torch.zeros(edge_index.shape[1], device=x.device, dtype=torch.int32)

        # Iterate through all graphs in the batch
        for i in range(num_graphs):
            # Get the mask for nodes belonging to the i-th graph
            active_nodes = (batch_index == i).nonzero().squeeze()
            
            edge_mask_i = torch.isin(edge_index[0], active_nodes) & torch.isin(edge_index[1], active_nodes)
            edge_mask[edge_mask_i] = i
    else:
        edge_mask = torch.zeros([0,2], device=x.device, dtype=torch.int32)

    return batch_index, edge_mask, num_graphs