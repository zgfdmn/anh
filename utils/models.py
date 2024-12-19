import torch
import torch.nn as nn
from tqdm import tqdm
from utils.intra_class_pairs import intra_loss
from utils.structure import intra_structure
def get_embeddings(dataloader, model, device='cuda',moding='train'):
    s, e = 0, 0
    data_embed_collect = []
    label_collect = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            data, label = data[0].to(device), data[1].to(device)
            q = model(data)
            if moding == 'test':
                data_embed_collect.append(q)
                label_collect.append(label)
            if label.dim() == 1:
                label = label.unsqueeze(1)
            if i == 0:
                labels = torch.zeros(
                    len(dataloader.dataset),
                    label.size(1),
                    device=device,
                    dtype=label.dtype,
                )
                all_q = torch.zeros(
                    len(dataloader.dataset),
                    q.size(1),
                    device=device,
                    dtype=q.dtype,
                )
            e = s + q.size(0)
            all_q[s:e] = q
            labels[s:e] = label
            s = e
    if moding == 'test':
        return all_q, labels,data_embed_collect,label_collect
    else :
        return all_q, labels



def train_sgd(train_loader, model, optimizers, loss_func, mining_func, device, sched, train_loss,limit_m, verbose=True):
    model.train()
    criterion = nn.CrossEntropyLoss()


    accu_loss = torch.zeros(1).to(device)
    for batch_idx, (data, labels, id) in enumerate(train_loader):
        data, labels, id = data.to(device), labels.to(device), id.to(device)
        optimizers.zero_grad()
        embeddings = model(data)

        indices_tuple = mining_func(embeddings, labels)
        Triplet_Loss = loss_func(embeddings, labels, indices_tuple)

        Intra_class_Loss = intra_loss(embeddings,labels,id,limit_m)

        structure_loss = intra_structure(embeddings,labels,id)


        classification_loss = criterion(embeddings, labels.long())

        loss = Triplet_Loss + structure_loss + Intra_class_Loss + classification_loss
        accu_loss += loss.detach()

        if len(indices_tuple[0]) > 0:
            loss.backward()
            optimizers.step()
    sched.step(loss)
    if verbose:
        print("Loss = {}, Number of mined triplets = {}".format(
            loss, mining_func.num_triplets))
    return model,accu_loss.item() / (batch_idx + 1)

