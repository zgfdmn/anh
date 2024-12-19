import numpy as np
from torch.nn import init
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score
from pytorch_metric_learning import losses, miners, reducers
from utils.cnn import cnn_Network
from utils.data import CustomDataset
from utils.models import get_embeddings, train_sgd
from utils.scoring import RetrievalMetrics
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ToTensor


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def test(train_loader, test_loader, model, device, best_acc,best_model_weights,tag=None):

    model.eval()
    with torch.no_grad():
        train_embeddings, train_labels = get_embeddings(train_loader, model, device)
        test_embeddings, test_labels = get_embeddings(test_loader, model, device)
    metrics = RetrievalMetrics(
                               test_embeddings,
                               test_labels,
                               train_embeddings,
                               train_labels
                               )
    val_acc = 0
    for k in range(1, 20):
        k_val_acc, k_pred, k_label, k_n = metrics.knn_accuracy(k)
        if k_val_acc > val_acc:
            val_acc, pred, label, n, best_k = k_val_acc, k_pred, k_label, k_n, k
            prob = n / best_k

    for i in range(len(pred)):
        if pred[i]==1:
            prob[i] =prob[i]
        if pred[i]==0:
            prob[i] = 1-prob[i]
    loss_fn = nn.CrossEntropyLoss()
    predictions = test_embeddings.softmax(dim=1)
    test_loss = loss_fn(predictions, test_labels.flatten().long())
    if val_acc >= best_acc:
        best_acc = val_acc
        best_model_weights = model.state_dict()

    print('best_k: ', best_k)
    print('val_acc: ', val_acc)
    print('MAP@10: ', metrics.map_at_k(best_k))

    pred = pred.cpu()
    label = label.cpu()
    prob = prob.cpu()

    def calculate_metrics(gt, pred):
        confusion = confusion_matrix(gt, pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        print('TP:{}; TN:{}; FP:{}; FN:{};'.format(TP, TN, FP, FN))
        print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
        print('Sensitivity:', TP / float(TP + FN))
        print('Specificity:', TN / float(TN + FP))
        print('PPV:', TP / float(TP + FP))
        print('NPV:', TN / float(TN + FN))
        print('F1 score:', 2 * TP / float(2 * TP + FP + FN))

    calculate_metrics(label, pred)
    auc = roc_auc_score(label, prob)
    print("AUC:", auc)

    return best_model_weights, best_acc, val_acc,test_loss

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# 数据增强
train_transform = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(10),
    ToTensor()
])
test_transform = Compose([
    ToTensor()
])

train_data = np.load("data/train.npz")
X_train = train_data['x']
y_train = train_data['y']
id_train =train_data['n'].astype(int)

test_data = np.load("data/test.npz")
X_test = test_data['x']
y_test = test_data['y']
id_test = test_data['n'].astype(int)


train_count_0 = np.count_nonzero(y_train)
test_count_0 = np.count_nonzero(y_test)
train_data = CustomDataset(X_train,y_train,id_train,transform=train_transform)
test_data = CustomDataset(X_test,y_test,id_test,transform=test_transform)

BATCH_SIZE = 72
EPOCHS = 200
EMB_DIM = 2
LR = 0.00002
MARGIN= 1

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

def weights_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        init.constant_(m.bias, 0)


embedding = cnn_Network(out_dim=EMB_DIM)
embedding.apply(weights_init)

reducer = reducers.ThresholdReducer(low=0)

loss_func = losses.TripletMarginLoss(margin=MARGIN, reducer=reducer)
mining_func = miners.TripletMarginMiner(margin=MARGIN, type_of_triplets='semihard')
model = embedding
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.50, patience=2)
best_acc = 0
best_model_weights=None
limit_m = 0.1
train_loss = []
test_acc = []
train_all_loss = []
CrossEntropyLoss = []
for epoch in range(EPOCHS):
    print(f'第{epoch}epoch')
    model,train_loss = train_sgd(train_loader, model, optimizer, loss_func, mining_func, device,sched,train_loss,limit_m)
    train_all_loss.append(train_loss)
    tag = f'{LR}_{MARGIN}_NBD_{epoch}'
    acc = best_acc
    best_model_weights,best_acc,val_acc,test_loss  = test(train_loader, test_loader, model, device,best_acc=best_acc,best_model_weights=best_model_weights,tag=tag)
    test_acc.append(val_acc)
    CrossEntropyLoss.append(test_loss)
    if best_acc > acc:
        torch.save(best_model_weights, f"../weights/{limit_m}_{MARGIN}_{LR}_{best_acc}_model.pth")
