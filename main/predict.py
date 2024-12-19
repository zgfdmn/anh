import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from utils.cnn import cnn_Network
import random
import torch
from utils.data import CustomDataset
from utils.models import get_embeddings
from utils.scoring import RetrievalMetrics
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ToTensor

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

train_data = np.load("data/train.npz")
X_train = train_data['x']
y_train = train_data['y']
id_train =train_data['n'].astype(int)

test_data = np.load("data/test.npz")
X_test = test_data['x']
y_test = test_data['y']
id_test = test_data['n'].astype(int)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
train_transform = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(10),
    ToTensor()
])
test_transform = Compose([
    ToTensor()
])

train_data = CustomDataset(X_train,y_train,id_train,transform=train_transform)
test_data = CustomDataset(X_test,y_test,id_test,transform=test_transform)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 72
EMB_DIM = 2

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

embedding = cnn_Network(out_dim=EMB_DIM)

model = embedding
model = model.to(device)
model.load_state_dict(torch.load('../weights/best_model.pth'))

model.eval()

train_embeddings, train_labels = get_embeddings(train_loader, model, device,moding='train')
test_embeddings, test_labels,data_embed_collect,label_collect= get_embeddings(test_loader, model, device,moding='test')

data_embed_npy = torch.cat(data_embed_collect, axis=0).cpu().numpy()
label_npu = torch.cat(label_collect, axis=0).cpu().numpy()

metric= RetrievalMetrics(
                           test_embeddings,
                           test_labels,
                           train_embeddings,
                           train_labels,
                           )
val_acc = 0
for k in range(1, 20):
    k_val_acc, k_pred, k_label, k_n = metric.knn_accuracy(k)
    if k_val_acc > val_acc:
        val_acc, pred, label, n, best_k = k_val_acc, k_pred, k_label, k_n, k
        prob = n / best_k

for i in range(len(pred)):
    if pred[i]==1:
        prob[i] =prob[i]
    if pred[i]==0:
        prob[i] = 1-prob[i]
pred=pred.cpu()
label= label.cpu()
prob = prob.cpu()
test_labels = test_labels.cpu()
print(val_acc)

calculate_metrics(label, pred)
auc = roc_auc_score(label, prob)
print('AUC:',auc)


