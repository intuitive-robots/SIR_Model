import torch

GRAPH_MODALITY_LIST = ["one_hot_labels", "bb_coordinates", "cropped_image_feature"]

def combine_graph_modalities(graph_data, graph_mod, idx=None, j=None):
    if idx is None and j is None:
        base_graph = graph_data[graph_mod[0]].clone()
        feats = [graph_data[mod].x for mod in graph_mod]
    else:
        base_graph = graph_data[graph_mod[0]][idx][j].clone()
        feats = [graph_data[mod][idx][j].x for mod in graph_mod]
    base_graph.x = torch.cat(feats, dim=-1)
    
    return base_graph

def fuse_graphs(graph_data_left, graph_data_right, mod, step_idx=None, is_cropped_fusion=False):
    if step_idx is None:
        left_graph = graph_data_left[mod]
        right_graph = graph_data_right[mod]
    else:
        left_graph = graph_data_left[mod][step_idx]
        right_graph = graph_data_right[mod][step_idx]
    
    left_names = left_graph.node_names
    right_names = right_graph.node_names
    all_names = sorted(list(set(left_names) | set(right_names)))
    
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    num_nodes = len(all_names)
    
    feat_dim_l = left_graph.x.shape[1]
    feat_dim_r = right_graph.x.shape[1]
    
    # Create zero-filled tensors for the fused graph
    # Shape: [Total_Unique_Nodes, Left_Dim]
    x_l_mapped = torch.zeros((num_nodes, feat_dim_l), device=left_graph.x.device, dtype=left_graph.x.dtype)
    # Shape: [Total_Unique_Nodes, Right_Dim]
    x_r_mapped = torch.zeros((num_nodes, feat_dim_r), device=right_graph.x.device, dtype=right_graph.x.dtype)
    
    # Map Left Features
    for src_i, name in enumerate(left_names):
        target_i = name_to_idx[name]
        x_l_mapped[target_i] = left_graph.x[src_i]
        
    # Map Right Features
    for src_i, name in enumerate(right_names):
        target_i = name_to_idx[name]
        x_r_mapped[target_i] = right_graph.x[src_i]

    fused_x, bb_index = handle_fusing(x_l_mapped, x_r_mapped, mod, is_cropped_fusion)

    src, dst = torch.meshgrid(
        torch.arange(num_nodes, device=left_graph.x.device),
        torch.arange(num_nodes, device=left_graph.x.device),
        indexing="ij"
    )
    
    src = src.flatten()
    dst = dst.flatten()
    
    # Filter out self-loops (i != j)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    
    fused_edge_index = torch.stack((src, dst), dim=0)
    
    # if "bb_coordinates" in mod:
    #     fused_edge_attr = calculate_weight_dim_distance(fused_x, bb_index, src, dst)
    # else:
    fused_edge_attr = torch.ones(fused_edge_index.shape[1], device=left_graph.x.device, dtype=torch.float32)

    fused_graph = type(left_graph)(
        x=fused_x,
        edge_index=fused_edge_index,
        edge_attr=fused_edge_attr,
        node_names=all_names  # Store the new unified list of names
    )
    
    return fused_graph

def handle_fusing(left, right, mod, cropped_fusion):
    active_mods = []
    
    if GRAPH_MODALITY_LIST[0] in mod:
        ohl_index = mod.split(GRAPH_MODALITY_LIST[0]).index('')
        ohl_length = 37
        active_mods.append((ohl_index, 'ohl', ohl_length))
    if GRAPH_MODALITY_LIST[1] in mod:
        bb_index = mod.split(GRAPH_MODALITY_LIST[1]).index('')
        bb_length = 10
        active_mods.append((bb_index, 'bb', bb_length))
    if GRAPH_MODALITY_LIST[2] in mod:
        cropped_index = mod.split(GRAPH_MODALITY_LIST[2]).index('')
        # We use -1 as placeholder for dynamic length
        active_mods.append((cropped_index, 'crop', -1))
    
    # 2. Sort by rank (position in the tensor)
    active_mods.sort(key=lambda x: x[0])
    
    # 3. Calculate the dynamic length of 'cropped_image_feature'
    total_input_len = left.shape[-1]
    known_len = sum(m[2] for m in active_mods if m[2] != -1)
    crop_len = total_input_len - known_len
    
    # 4. Iterate, slice, and process
    fused_parts = []
    current_ptr = 0
    
    bb_index = -1
    
    for rank, name, length in active_mods:
        # Resolve actual length if dynamic
        eff_len = crop_len if length == -1 else length
        
        # Slice the current modality from both views
        l_slice = left[..., current_ptr : current_ptr + eff_len]
        r_slice = right[..., current_ptr : current_ptr + eff_len]
        
        if name == 'bb':
            # BB: User wants BOTH values included -> Concatenate (Length becomes 2x)
            fused_parts.append(torch.cat((l_slice, r_slice), dim=-1))
            bb_index = current_ptr
        elif name == 'crop':
            if cropped_fusion:
                fused_parts.append(l_slice)
            else:
                fused_parts.append(torch.cat((l_slice, r_slice), dim=-1))
        else:
            # OHL: ONE value (identical) -> Take Left
            fused_parts.append(l_slice)
            
        current_ptr += eff_len
    
    # 5. Reassemble the fused features
    return torch.cat(fused_parts, dim=-1), bb_index


def pad_sequence(sequence, target_length):
    """Helper to pad a sequence by repeating the last frame if it's too short."""
    current_len = sequence.shape[0]
    if current_len < target_length:
        pad_qty = target_length - current_len
        last_frame = sequence[-1].unsqueeze(0)
        padding = last_frame.repeat(pad_qty, *([1]*(len(sequence.shape)-1)))
        return torch.cat((sequence, padding), dim=0)
    return sequence

def convert_weight_to_dim_distance(graph_data, graph_mod):
    for i in range(len(graph_data['bb_coordinates'])):
        for j in range(len(graph_data['bb_coordinates'][i])):
            middle_points = graph_data['bb_coordinates'][i][j].x[:,8:]
            
            row, col = graph_data['bb_coordinates'][i][j].edge_index
            
            source_points = middle_points[row] 
            target_points = middle_points[col]
            
            distance = torch.abs(source_points - target_points)
            
            for mod in graph_mod:
                graph_data[mod][i][j].edge_attr = distance
    return graph_data

def calculate_weight_dim_distance(graph_data, start_index, row, col):
    # TODO fix for all possible combinations
    middle_points_left = graph_data[:,start_index+8:start_index+10]
    middle_points_right = graph_data[:,start_index+18:start_index+20]
    
    source_points_left = middle_points_left[row]
    target_points_left = middle_points_left[col]
    
    source_points_right = middle_points_right[row]
    target_points_right = middle_points_right[col]
    
    distance_left = torch.abs(source_points_left - target_points_left)
    distance_right = torch.abs(source_points_right - target_points_right)
    
    distance = torch.concat((distance_left, distance_right), dim=-1)
    
    return distance