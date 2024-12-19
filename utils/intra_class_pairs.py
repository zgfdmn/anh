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

    pairs = []
    sps = []

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
            if ne_result.any():
                if id_2 == id_1 and y_1 != y_2:

                    pair.append([i, j])

                if id_2 != id_1 and y_1 == y_2:
                    sp.append(j)
        pairs.append(pair)
        sps.append(sp)
    anchor_same_idx, positive_diverse_idx, negative_same_idx = get_sample_pair(pairs,sps)

    anchor_same_idx = torch.tensor(anchor_same_idx)
    positive_diverse_idx = torch.tensor(positive_diverse_idx)
    negative_same_idx = torch.tensor(negative_same_idx)

    return anchor_same_idx, positive_diverse_idx, negative_same_idx


def distance_loss(x,y,m):
    return x-y-m

def intra_loss(embedding, label, id, m):
    anchor_same_idx, positive_diverse_idx, negative_same_idx = intra_class_sample_pairs(embedding,label,id)
    if len(anchor_same_idx) == 0:
        return 0
    mat = torch.cdist(embedding, embedding, 2)

    pn_diverse_dists = mat[positive_diverse_idx, negative_same_idx]
    an_same_dists = mat[anchor_same_idx, negative_same_idx]
    current_margins = distance_loss(pn_diverse_dists,an_same_dists,m)

    loss = torch.nn.functional.relu(current_margins)

    return torch.mean(loss)

