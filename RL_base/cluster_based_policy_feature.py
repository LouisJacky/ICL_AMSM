import h5py
import sklearn.cluster as cluster
import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F
import math

# def cluster_to_fix(can_features, device, args):
#     cluster_kmeans = KMeans(n_clusters=args.candidate_num, random_state=0, max_iter=20)
#     kmeans_ret = cluster_kmeans.fit(can_features)
#     sample_id = []
#     for cluster_ids in range(args.candidate_num):
#         sub_cluster_ids = np.where(kmeans_ret.labels_ == cluster_ids)[0]
#         sub_cluster_features = torch.from_numpy(can_features[kmeans_ret.labels_ == cluster_ids]).to(device)
#         centers = torch.from_numpy(cluster_kmeans.cluster_centers_[cluster_ids]).to(device)
#         scores = torch.matmul(sub_cluster_features, centers.unsqueeze(1))
#         cand_prob = scores.squeeze(-1).clone().detach()
#         cand_prob = cand_prob.cpu().numpy()
#         cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # replace np.nan with 0
#         cand_prob /= cand_prob.sum()  # make probabilities sum to 1
#         # sample shot_pids from the cand_prob distribution
#         cids = np.random.choice(range(len(sub_cluster_features)), 1, p=cand_prob, replace=False)
#         sample_id.append(sub_cluster_ids[cids[0]])
#
#     return sample_id
def cluster_to_fix(can_features, device, args):
    cluster_kmeans = KMeans(n_clusters=args.cluster_num, random_state=0, max_iter=20, n_init='auto')
    kmeans_ret = cluster_kmeans.fit(can_features)
    cluster_centers = torch.from_numpy(cluster_kmeans.cluster_centers_).to(device)
    cluster_id = kmeans_ret.labels_


    return cluster_id, cluster_centers


def cluster_to_offline_nearest(can_features, labels, cluster_centers, device, query_id, args):
    cluster_query_id = labels[query_id]
    sub_cluster_ids = np.where(labels == cluster_query_id)[0]


    if len(sub_cluster_ids) < (args.limit_number):  #  deal_with_small_clusters
        centers = torch.from_numpy(cluster_centers).to(device)
        target_idx = torch.mm(centers, torch.from_numpy(can_features[query_id]).to(device).unsqueeze(1))
        target_idx = target_idx.cpu().detach().numpy().tolist()
        cand_ids = sorted(range(len(target_idx)), key=lambda i: target_idx[i], reverse=True)
        for cand_id in cand_ids:
            if cand_id == cluster_query_id:
                continue
            new_cluster_id = np.where(labels == cand_id)[0]
            sub_cluster_ids = np.concatenate((sub_cluster_ids, new_cluster_id))
            if len(sub_cluster_ids) > (args.limit_number):   # re-assign samples in small clusters to make them bigger than fix_num
                cluster_query_id = cand_id
                break

    sub_cluster_features = []
    for feature_label_id in range(len(labels)):
        if feature_label_id == query_id:  # delete itself ground truth feature
            continue
        if labels[feature_label_id] == cluster_query_id:
            sub_cluster_features.append(can_features[feature_label_id])


    sub_cluster_features = torch.from_numpy(np.stack(sub_cluster_features, axis=0)).to(device)
    ground_truth_idx = np.where(sub_cluster_ids == query_id)[0]
    sub_cluster_ids = np.delete(sub_cluster_ids, ground_truth_idx)  # delete itself ground truth label


    # sample shot_pids from the cand_prob distribution
    if sub_cluster_features.size()[0] <= args.candidate_num: ## choose candiate sample
        sample_id = sub_cluster_ids.tolist()
    else:
        centers = torch.from_numpy(can_features[query_id]).to(device)
        scores = torch.matmul(sub_cluster_features, centers.unsqueeze(1))
        # scores = F.cosine_similarity(sub_cluster_features, centers.unsqueeze(0), dim=1).unsqueeze(-1)
        # cos_sim = F.cosine_similarity(sub_cluster_features[0], centers, dim=0)

        cand_prob = scores.squeeze(-1).clone().detach()
        cand_prob = cand_prob.cpu().numpy()
        cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # replace np.nan with 0
        cand_prob /= cand_prob.sum()  # make probabilities sum to 1
        # cids = np.random.choice(range(len(cand_prob)), args.candidate_num, p=cand_prob, replace=False)
        all_cids = sorted(range(len(cand_prob)), key=lambda i: cand_prob[i], reverse=True)
        cids = all_cids[:math.ceil(args.candidate_num / 2)] + all_cids[-math.ceil(args.candidate_num / 2):]
        # cids = np.random.choice(all_cids, args.candidate_num, replace=False)
        sample_id = [sub_cluster_ids[cid] for cid in cids]


    return sample_id


def cluster_nearest_farthest(cluster_centrid, cluster_features):
    cluster_features = torch.from_numpy(np.stack(cluster_features, axis=0))
    scores = torch.matmul(cluster_features, cluster_centrid.unsqueeze(1))
    # cos_sim = F.cosine_similarity(sub_cluster_features[0], centers, dim=0)

    cand_prob = scores.squeeze(-1).clone().detach()
    cand_prob = cand_prob.cpu().numpy()
    cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # replace np.nan with 0
    cand_prob /= cand_prob.sum()  # make probabilities sum to 1
    # cids = np.random.choice(range(len(cand_prob)), args.candidate_num, p=cand_prob, replace=False)
    all_cids = sorted(range(len(cand_prob)), key=lambda i: cand_prob[i], reverse=True)
    sample_id = [all_cids[0], all_cids[-1]]
    return sample_id

def cluster_nearest_farthest2(cluster_centrid, cluster_features):
    # cluster_centrid 参数不再使用
    return list(range(len(cluster_features)))
    # cluster_features = torch.from_numpy(np.stack(cluster_features, axis=0))
    # scores = torch.matmul(cluster_features, cluster_centrid.unsqueeze(1))
    # # cos_sim = F.cosine_similarity(sub_cluster_features[0], centers, dim=0)
    #
    # cand_prob = scores.squeeze(-1).clone().detach()
    # cand_prob = cand_prob.cpu().numpy()
    # cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # replace np.nan with 0
    # cand_prob /= cand_prob.sum()  # make probabilities sum to 1
    # # cids = np.random.choice(range(len(cand_prob)), args.candidate_num, p=cand_prob, replace=False)
    # all_cids = sorted(range(len(cand_prob)), key=lambda i: cand_prob[i], reverse=True)
    # sample_id = all_cids[:]
    # return sample_id

def cluster_nearest_farthest_new(can_features, labels, cluster_centers, device, query_id, args):
    # 获取查询样本所属的聚类
    cluster_query_id = labels[query_id]

    # 获取同一聚类的所有样本索引
    cluster_sample_ids = np.where(labels == cluster_query_id)[0]

    # 排除查询样本本身
    cluster_sample_ids = cluster_sample_ids[cluster_sample_ids != query_id]

    return cluster_sample_ids

def cluster_to_nearest(can_features, device, query_id, args):
    cluster_kmeans = KMeans(n_clusters=args.cluster_num, random_state=0, max_iter=20, n_init='auto')
    kmeans_ret = cluster_kmeans.fit(can_features)
    cluster_query_id = kmeans_ret.labels_[query_id]
    sub_cluster_ids = np.where(kmeans_ret.labels_ == cluster_query_id)[0]


    if len(sub_cluster_ids) < (args.shot_number * 2):  #  deal_with_small_clusters
        centers = torch.from_numpy(cluster_kmeans.cluster_centers_).to(device)
        target_idx = torch.mm(centers, torch.from_numpy(can_features[query_id]).to(device).unsqueeze(1))
        target_idx = target_idx.cpu().detach().numpy().tolist()
        cand_ids = sorted(range(len(target_idx)), key=lambda i: target_idx[i], reverse=True)
        for cand_id in cand_ids:
            if cand_id == cluster_query_id:
                continue
            sub_cluster_ids = np.where(kmeans_ret.labels_ == cand_id)[0]
            if len(sub_cluster_ids) > (args.shot_number * 2):   # re-assign samples in small clusters to make them bigger than fix_num
                cluster_query_id = cand_id
                break

    sub_cluster_features = []
    for feature_label_id in range(len(kmeans_ret.labels_)):
        if feature_label_id == query_id:  # delete itself ground truth feature
            continue
        if kmeans_ret.labels_[feature_label_id] == cluster_query_id:
            sub_cluster_features.append(can_features[feature_label_id])


    sub_cluster_features = torch.from_numpy(np.stack(sub_cluster_features, axis=0)).to(device)
    ground_truth_idx = np.where(sub_cluster_ids == query_id)[0]
    sub_cluster_ids = np.delete(sub_cluster_ids, ground_truth_idx)  # delete itself ground truth label


    # sample shot_pids from the cand_prob distribution
    if sub_cluster_features.size()[0] <= args.candidate_num: ## choose candiate sample
        sample_id = sub_cluster_ids.tolist()
    else:
        centers = torch.from_numpy(can_features[query_id]).to(device)
        # scores = torch.matmul(sub_cluster_features, centers.unsqueeze(1))
        scores = F.cosine_similarity(sub_cluster_features, centers.unsqueeze(0), dim=1).unsqueeze(-1)
        # cos_sim = F.cosine_similarity(sub_cluster_features[0], centers, dim=0)

        cand_prob = scores.squeeze(-1).clone().detach()
        cand_prob = cand_prob.cpu().numpy()
        cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # replace np.nan with 0
        cand_prob /= cand_prob.sum()  # make probabilities sum to 1
        # cids = np.random.choice(range(len(cand_prob)), args.candidate_num, p=cand_prob, replace=False)
        cids = sorted(range(len(cand_prob)), key=lambda i: cand_prob[i], reverse=True)[:args.candidate_num]
        sample_id = [sub_cluster_ids[cid] for cid in cids]


    return sample_id


