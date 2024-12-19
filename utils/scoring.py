import torch

def find_argKmin(dist_mat, k, dim=1, same_set=False):
    ''' Vectorized retrieval of a Q x R divergence matrix, where
    Q is size of query set, R is size of reference set.

    If same_set is True, assumes reference and query embeddings are
    from different sets. Otherwise if they are same set, then
    compute k+1 neighbors and toss nearest neighbor.
    '''
    max_k = min(dist_mat.shape[1], k + 1)
    if isinstance(dist_mat, torch.Tensor):
        topk_idx = torch.topk(
            dist_mat, max_k, dim=dim, largest=False, sorted=True)[1]
    elif isinstance(dist_mat, LazyTensor):
        topk_idx = dist_mat.argKmin(max_k, dim=dim)
    else:
        raise TypeError('dist_mat must be of type torch.Tensor or LazyTensor')

    if same_set:        # toss nearest neighbor
        return topk_idx[:, 1:]
    elif k < max_k:     # extra neighbor was computed, toss last one
        return topk_idx[:, :-1]
    return topk_idx     # if k is largest possible, then don't toss anything



class RetrievalMetrics:
    def __init__(self,query_emb, query_labels,
                 ref_emb=None, ref_labels=None, device='default'):
        if device == 'default':
            device = query_emb.device
        
        if ref_emb is None or ref_labels is None:
            self.same_set = True
            ref_emb = query_emb.clone()
            ref_labels = query_labels.clone()
        else:
            self.same_set = False

        self.query_emb = query_emb.to(device)
        self.ref_emb = ref_emb.to(device)
        self.query_labels = query_labels.flatten().to(device)
        self.ref_labels = ref_labels.flatten().to(device)

        self.dist_mat = torch.cdist(query_emb, ref_emb)
        if isinstance(self.dist_mat, torch.Tensor):
            self.dist_mat = self.dist_mat.to(device)
        self.device = device

    def knn_accuracy(self, k):
        topk = find_argKmin(self.dist_mat, k, dim=1, same_set=self.same_set)
        neighbor_labels = self.ref_labels[topk]
        mode_labels, n = torch.mode(neighbor_labels, dim=1)
        return torch.mean((mode_labels == self.query_labels).float()),mode_labels,self.query_labels,n

    def precision_at_k(self, k):
        topk = find_argKmin(self.dist_mat, k, dim=1, same_set=self.same_set)
        topk_pred = self.ref_labels[topk]
        prec_per_query = torch.mean((self.query_labels.unsqueeze(1) == topk_pred).float(), dim=1)
        return torch.mean(prec_per_query)

    def mean_average_precision(self):
        n_query = len(self.query_labels)
        n_ref = len(self.ref_labels)
        topk = find_argKmin(self.dist_mat, n_ref, dim=1, same_set=self.same_set)
        topk_pred = self.ref_labels[topk]

        is_correct = (self.query_labels.unsqueeze(1) == topk_pred).float()
        cumulative_correct = torch.cumsum(is_correct, dim=1).float()

        k_idx = torch.arange(
            1, n_ref + 1 - self.same_set, device=self.device).repeat(n_query, 1)
        precision_at_ks = (cumulative_correct * is_correct) / k_idx.float()
        summed_precision_per_row = torch.sum(precision_at_ks, dim=1)

        max_possible_matches_per_row = torch.sum(is_correct, dim=1)
        max_possible_matches_per_row[max_possible_matches_per_row == 0] = 1
        accuracy_per_sample = summed_precision_per_row / max_possible_matches_per_row
        return torch.mean(accuracy_per_sample)

    def map_at_k(self, k):
        n_query = len(self.query_labels)
        topk = find_argKmin(self.dist_mat, k, dim=1, same_set=self.same_set)
        topk_pred = self.ref_labels[topk]

        is_correct = (self.query_labels.unsqueeze(1) == topk_pred).float()
        cumulative_correct = torch.cumsum(is_correct, dim=1).float()

        k_idx = torch.arange(
            1, k + 1 - self.same_set, device=self.device).repeat(n_query, 1)
        precision_at_ks = (cumulative_correct * is_correct) / k_idx.float()
        summed_precision_per_row = torch.sum(precision_at_ks, dim=1)

        max_possible_matches_per_row = torch.sum(is_correct, dim=1)
        max_possible_matches_per_row[max_possible_matches_per_row == 0] = 1
        accuracy_per_sample = summed_precision_per_row / max_possible_matches_per_row
        return torch.mean(accuracy_per_sample)

    def area_under_curve(self):
        n_query = len(self.query_labels)
        n_ref = len(self.ref_labels)
        topk = find_argKmin(self.dist_mat, n_ref, dim=1, same_set=self.same_set)
        topk_pred = self.ref_labels[topk]

        auc = torch.zeros(n_query, device=self.device)
        for i in range(n_query):
            y = self.query_labels[i]
            y_count = torch.sum(self.ref_labels == y)
            swapped_pairs = torch.sum(
                (topk_pred[i] != y) * (y_count - torch.cumsum(topk_pred[i] == y, dim=0))
            )
            auc[i] = 1 - swapped_pairs / (y_count * (n_ref - y_count))
        return torch.nansum(auc) / (n_query - torch.sum(torch.isnan(auc)))

    def topk_accuracy(self, k):
        topk = find_argKmin(self.dist_mat, k, dim=1, same_set=self.same_set)
        topk_pred = self.ref_labels[topk]

        found_per_query = torch.any(self.query_labels == topk_pred, axis=1)
        return torch.mean(found_per_query)
