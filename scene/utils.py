import open3d as o3d
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def visualize_octree_space(pts, node_info_list, depth_colors, offscreen=False):
    if pts is not None:
        if type(pts)==torch.Tensor:
            points = pts.cpu().numpy()
        elif type(pts)==torch.nn.Parameter:
            points = pts.detach().cpu().numpy()
        else:
            points = pts
        N = points.shape[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
        pcd_list = [pcd]
    else:
        pcd_list = []
    lvls_list = []
    for node_info in node_info_list:
        lvl_voxel_list = []
        if len(node_info['node_list']) > 0:
            for node in node_info['node_list']:
                color = depth_colors[node['depth']]
                if type(node['bounds'])==torch.Tensor:
                    assert node['bounds'].shape[0]==2, 'bounds shouuld be 2x3'
                    bounds = node['bounds'].cpu().numpy()
                elif type(node['bounds'])==np.ndarray:
                    assert node['bounds'].shape[0]==2, 'bounds shouuld be 2x3'
                    bounds = node['bounds']
                else:
                    raise NotImplementedError
                assert (bounds[0]<bounds[1]).all(), 'bounds[0]<bounds[1] failed'
                bbox = o3d.geometry.AxisAlignedBoundingBox(np.array(bounds[0]), np.array(bounds[1]))
                bbox.color = color
                lvl_voxel_list.append(bbox)
            lvls_list.append(lvl_voxel_list)
    if not offscreen:
        o3d.visualization.draw_geometries([item for sublist in lvls_list for item in sublist]+pcd_list)
    num_boxes = sum([len(lvl_voxel_list) for lvl_voxel_list in lvls_list])
    print('total boxes num: {}'.format(num_boxes))
    return [item for sublist in lvls_list for item in sublist]+pcd_list



def check_num_points(N, node_info_list):
    print('check points number == {}'.format(N))
    val_num_pts_dict = {}
    for lvl, depth_i_node_list in enumerate(node_info_list):
        cur_lvl_pts_list = []
        if depth_i_node_list['count'] > 0:
            for node in depth_i_node_list['node_list']:
                cur_lvl_pts_list.append(len(node['points']))
        val_num_pts_dict[lvl] = cur_lvl_pts_list
        print('level {} leaf-node num: {}'.format(lvl, depth_i_node_list['count']))
        print('level {} points num: {}'.format(lvl, sum(cur_lvl_pts_list)))
    num_pts_perlvl = [sum(val_num_pts_dict[lvl]) for lvl in val_num_pts_dict.keys()]
    print('total points num: {}'.format(sum(num_pts_perlvl)))
    if sum(num_pts_perlvl) == N:
        print('check points number passed')
    else:
        print('check points number failed !!!!!!!!')

def collect_nodeinfo_perdepth(node, node_info_list, current_depth):
    if node is not None:
        node_info = {
            'depth': current_depth,
            'child_index': node_info_list[current_depth]['count'],
            'bounds': node.bounds
        }
        node_info_list[current_depth]['count'] += 1
        node_info_list[current_depth]['node_list'].append(node_info)

        for child in node.children:
            collect_nodeinfo_perdepth(child, node_info_list, current_depth + 1)
def collect_leafnodeinfo_perdepth(node, node_info_list, current_depth):
    if node is not None:
        if node.is_leaf():
            node_info = {
                'depth': current_depth,
                'bounds': node.bounds,
                'points': node.points
            }
            node_info_list[current_depth]['leaf_count'] += 1
            node_info_list[current_depth]['node_list'].append(node_info)
        else:
            for child in node.children:
                collect_leafnodeinfo_perdepth(child, node_info_list, current_depth + 1)


def o3d_knn_semantic_filter(pts, semantics, num_knn, similarity_threshold):
    """
    Finds nearby points based on spatial proximity and filters them by semantic feature similarity.

    Parameters:
    - pts: numpy array of point coordinates, shape (N, 3).
    - semantics: numpy array of semantic features, shape (N, 128).
    - num_knn: number of nearest neighbors to search for each point.
    - similarity_threshold: threshold for cosine similarity to retain neighbors.

    Returns:
    - filtered_indices: array of indices of neighbors filtered by semantic similarity.
    - filtered_sq_dists: array of squared distances of the filtered neighbors.
    """
    indices, sq_dists = o3d_knn(pts, num_knn)
    filtered_indices = []
    filtered_sq_dists = []

    for i, neighbors in enumerate(indices):
        # Calculate cosine similarity between the semantic feature of the point and its neighbors
        semantic_vec = semantics[i]
        neighbor_semantics = semantics[neighbors]
        
        # Compute cosine similarity
        dot_product = np.dot(neighbor_semantics, semantic_vec)
        semantic_norm = np.linalg.norm(semantic_vec)
        neighbor_norms = np.linalg.norm(neighbor_semantics, axis=1)
        cosine_similarity = dot_product / (neighbor_norms * semantic_norm)
        
        # Filter neighbors based on similarity threshold
        valid_indices = neighbors[cosine_similarity >= similarity_threshold]
        valid_dists = np.array(sq_dists[i])[cosine_similarity >= similarity_threshold]

        filtered_indices.append(valid_indices)
        filtered_sq_dists.append(valid_dists)

    return filtered_sq_dists, filtered_indices

def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def o3d_semantic_knn(pts, semantic_features, num_knn=20):
    indices = []
    sq_dists = []
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    all_indices = []
    all_dists = []
    num_points = len(pcd.points)

    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        all_indices.append(i[1:])  
        all_dists.append(d[1:])

    all_indices = np.array(all_indices)
    all_dists = np.array(all_dists)

    for idx in range(num_points):
        spatial_indices = all_indices[idx]
        spatial_dists = all_dists[idx]
        feature = semantic_features[idx].reshape(1, -1)  

        candidate_features = semantic_features[spatial_indices].squeeze(1)
        cos_similarities = cosine_similarity(feature, candidate_features).flatten()

        top_k_idx = np.argsort(-cos_similarities)[:num_knn]
        indices.append(spatial_indices[top_k_idx])
        sq_dists.append(spatial_dists[top_k_idx])

    return np.array(sq_dists), np.array(indices)


def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def weighted_l2_loss_v3(curr_offset, prev_offset, weights):


    R_i_t = torch.eye(3).unsqueeze(0).expand(curr_offset.shape[0], 3, 3).to(curr_offset.device)
    R_i_t1 = R_i_t.clone()
    R_i_t_inv = torch.inverse(R_i_t)

    rotated_curr_offset = torch.matmul(R_i_t1, torch.matmul(R_i_t_inv, curr_offset.unsqueeze(-1))).squeeze(-1)
    diff = prev_offset - rotated_curr_offset
    loss = (weights * (diff ** 2).sum(dim=1)).mean()

    return loss



def pytorch_knn(pts, num_knn, batch_size=1024):
    pts_tensor = torch.tensor(pts, dtype=torch.float32).cuda()  
    num_points = pts_tensor.size(0)
    indices = []
    sq_dists = []

    for i in range(0, num_points, batch_size):
        end = min(i + batch_size, num_points)
        batch_pts = pts_tensor[i:end]
        batch_sq_dists = torch.cdist(batch_pts, pts_tensor)
        top_k_values, top_k_indices = torch.topk(batch_sq_dists, num_knn + 1, largest=False, sorted=False)
        indices.append(top_k_indices)
        sq_dists.append(top_k_values)

    if sq_dists and indices:  
        sq_dists_tensor = torch.cat(sq_dists)
        indices_tensor = torch.cat(indices)

    results = (sq_dists_tensor[:, 1:].cpu().numpy(), indices_tensor[:, 1:].cpu().numpy())
    del pts_tensor, sq_dists 
    return results

