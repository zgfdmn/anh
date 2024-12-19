import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_sample_pair(pairs,sps):
    anchor_same_idx = []
    positive_diverse_idx = []
    negative_same_idx = []
    for i in range(len(pairs)):
        pair = pairs[i]
        sp = sps[i]
        for j in pair:
            a = [j[0]] * len(sp)
            n = [j[1]] * len(sp)
            anchor_same_idx.extend(a)
            positive_diverse_idx.extend(sp)
            negative_same_idx.extend(n)
    return anchor_same_idx, positive_diverse_idx, negative_same_idx

def intra_class_sample_pairs(embedding, label, id):
    # p = []
    # n = []

    pairs = []
    sps = []# 与p同类
    # sn = []  # 与n同类
    for i in range(len(embedding)):
        pair = []
        sp = []
        x_1 = embedding[i]
        y_1 = label[i]
        id_1 = id[i]
        for j in range(len(embedding)):
            x_2 = embedding[j]
            y_2 = label[j]
            id_2 = id[j]
            ne_result = x_2.ne(x_1)
            if ne_result.any():#x_2不等于x_1
                if id_2 == id_1 and y_1 != y_2:  # 同一患者的两种不同状态
                    # p.append(i)
                    # n.append(j)
                    pair.append([i, j])
                # if id_2 != id_1 and y_1 != y_2:  # 不同患者且与y_1状态不一致
                #     sn.append(j)
                if id_2 != id_1 and y_1 == y_2:  # 不同患者且与y_1状态一致
                    sp.append(j)
        pairs.append(pair)
        sps.append(sp)
    anchor_same_idx, positive_diverse_idx, negative_same_idx = get_sample_pair(pairs,sps)
    # pairs = list(set(pairs))
    # # 三元组对应的索引
    # # pairs0 = [[item for item in sublist] + [num] for sublist in pairs for num in sp]#与第一个索引对应样本同类
    # # pairs1 = [[item for item in sublist] + [num] for sublist in pairs for num in sn]#与第二个索引对应样本同类
    # # 将三元组分开
    # p = [sublist[0] for sublist in pairs]
    # n = [sublist[1] for sublist in pairs]
    # # anchor_idx, positive_idx, negative_idx
    anchor_same_idx = torch.tensor(anchor_same_idx)#同一患者
    positive_diverse_idx = torch.tensor(positive_diverse_idx)#不同患者
    negative_same_idx = torch.tensor(negative_same_idx)#同一患者

    return anchor_same_idx, positive_diverse_idx

# class intra_class_sample_pairs_loss:
#     def __init__(self,distance):
#         self.distance = distance
#
#     def compute_loss(self,embedding,label,id):
#         anchor_same_idx, positive_diverse_idx, negative_same_idx = intra_class_sample_pairs(embedding,label,id)
#         if len(anchor_same_idx) == 0:
#             return 0
#         mat = self.distance(embedding, embedding)
#         pn_diverse_dists = mat[positive_diverse_idx, negative_same_idx]
#         an_same_dists = mat[anchor_same_idx, negative_same_idx]
#         current_margins = self.distance.margin(pn_diverse_dists,an_same_dists)
#         loss = torch.nn.functional.relu(current_margins)
#         return loss
#
#     def get_default_reducer(self):
#         return AvgNonZeroReducer()
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import mahalanobis

def ma_distance(p, distr):

    # p: a point
    # distr : a distribution

    # covariance matrix
    cov = torch.cov(distr)

    # average of the points in distr
    avg_distri = torch.average(distr, axis=0)

    dis = mahalanobis(p, avg_distri, cov)

    return dis

def mahalanobis_distance(X):
    mean = torch.mean(X, dim=0)

    # 计算协方差矩阵
    cov = torch.cov(X.T)  # 注意：这里X需要转置，因为torch.cov计算的是列向量的协方差
    det_cov = torch.det(cov)

    is_invertible_by_det = det_cov != 0
    # print(f"Matrix is invertible by determinant: {is_invertible_by_det}")
    if is_invertible_by_det:
        cov_inv = torch.linalg.inv(cov)
    else:
        cov_inv = torch.linalg.pinv(cov)
    # 计算每个数据点与均值的差异
    diff = X - mean

    # 计算马氏距离矩阵
    mahalanobis_distance = torch.mm(diff, torch.mm(cov_inv, diff.t()))
    return mahalanobis_distance
def distance_loss(x,y,m):
    return x-y


# def intra_loss(embedding,label,id,distance):
def intra_structure(embedding, label, id, mean_d0=7.4345, mean_d1=8.2030, k0=0.5, k1=0.5):
    # d_0 = 0
    num_0 = 0
    loss_d0 = 0
    # d_1 = 0
    num_1 = 0
    loss_d1 = 0
    LOSS = 0
    for i in range(len(embedding)):
        x_1 = embedding[i]
        y_1 = label[i]
        id_1 = id[i]
        if y_1 == 0 :
            for j in range(1,len(embedding)):
                x_2 = embedding[j]
                y_2 = label[j]
                id_2 = id[j]
                if id_2 != id_1 and y_1 == y_2:
                    d_0 = np.linalg.norm(x_1.detach() - x_2.detach()) - mean_d0 - k0
                    num_0 += 1
                    loss_d0 += max(d_0, 0)
        else:
            for h in range(1,len(embedding)):
                x_3 = embedding[h]
                y_3 = label[h]
                id_3 = id[h]
                if id_3 != id_1 and y_1 == y_3:
                    # d = np.linalg.norm(x_1.detach() - x_3.detach())
                    d_1 = np.linalg.norm(x_1.detach() - x_3.detach()) - mean_d1 - k1
                    num_1 += 1
                    loss_d1 += max(d_1,0)
        LOSS += loss_d0 / (num_0 + 0.00001) + loss_d1 / (num_1 +0.00001)

    return LOSS/len(embedding)

