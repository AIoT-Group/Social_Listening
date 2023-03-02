import torch
from torch import nn
import numpy as np
import csv
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection as model_selection
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")     
from google.colab import drive
drive.mount('/content/drive')

with open("/content/drive/MyDrive/data_stopword_2.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    data_list=[row for row in reader]
# word list

with open("/content/drive/MyDrive/DataUseful.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    label_list = [row for row in reader]
    label_list = label_list
    # 0 = Ukraina and 1 = Nga
    for i in range(0,len(label_list)):
      if len(data_list[i])==0:
        label_list.pop(i)
    for i in range(0,len(label_list)):
        if label_list[i][1]=="U":
            label_list[i]=0
        else: 
            label_list[i]=1
#  label

count=0
while count<len(data_list):
  if (len(data_list[count])==0):
    data_list.pop(count)
    count=-1
  count+=1

with open("/content/drive/MyDrive/Vocab2.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    vocab = [row[0] for row in reader]

word_to_ix = {word: i for i, word in enumerate(vocab)}
# word : index
# chuyển các từ trong câu thành index
def text_to_index(X_text):
    X_index = []
    for i in X_text:
        X_index.append(torch.tensor([word_to_ix[w] for w in i], dtype=torch.long))
    return np.array(X_index)
data = text_to_index(data_list)
# đưa data về cùng kích thước của câu có số từ nhiều nhất
data_padded = pad_sequence(data,batch_first=True)

label = torch.tensor(label_list,dtype=torch.float64)

X_train_val, X_test, y_train_val, y_test = model_selection.train_test_split(data_padded, label, train_size=8/10,test_size=2/10, random_state=0)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train_val, y_train_val, train_size=7/8,test_size=1/8, random_state=0)

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, num_embeddings):
        super(RNNModel, self).__init__()
        
        self.emb = nn.Embedding(num_embeddings, input_dim)

        self.hidden_dim = hidden_dim
        
        self.layer_dim = layer_dim
        
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim)
            
        x2 = self.emb(x)

        out, hn = self.rnn(x2, h0)

        outputs = torch.sigmoid(self.fc(hn))
        return outputs

epochs = 100
# số featue của 1 từ
num_embeddings = len(vocab)
input_dim = 100
hidden_dim = 10
layer_dim = 1
output_dim = 1
learning_rate = 0.001
patience = 5
batch_size = 32
# chia batch_size
train = TensorDataset(X_train,y_train)
train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, num_embeddings)
loss_function = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []
losses_val = []

count_batch = 0
count_epoch = 0
count_stop = 0
for epoch in tqdm((range(epochs)),desc='Training Epochs'):
    count_epoch+=1
    for i, (X_mini, y_mini) in enumerate(train_loader):
        # print("X:",X_mini.shape)
        count_batch+=1
    
        optimizer.zero_grad()   
        outputs = model(X_mini) 
        # print(outputs.shape) 

        loss = loss_function(torch.squeeze(outputs), y_mini) 
        loss.backward() 
        optimizer.step() 
    
        with torch.no_grad():   # không tính gradient
            # tính loss và độ chính xác cho test_set
            outputs_val = torch.squeeze(model(X_val)) 
            loss_val = loss_function(outputs_val, y_val)
            
            predicted = outputs_val.round().detach().numpy()  # làm tròn và chuyển thành array
            total_val = y_val.size(0) # số test sample
            correct_val = np.sum(predicted == y_val.detach().numpy()) # số sample dự đoán đúng
            accuracy_val = 100 * correct_val/total_val
            losses_val.append(loss_val.item())
            
            # tính loss và độ chính xác cho training_set    
            outputs_mini = torch.squeeze(model(X_mini))   
            loss_mini = loss_function(outputs_mini, y_mini)

            total_mini = y_mini.size(0)
            correct_mini = np.sum(torch.squeeze(outputs_mini).round().detach().numpy() == y_train.detach().numpy())
            accuracy_mini = 100 * correct_mini/total_mini
            losses.append(loss_mini.item())
            
    print(f"\nVal - Loss: {loss_val.item()}. Accuracy: {accuracy_val}")
    print(f"Mini_batch -  Loss: {loss_mini.item()}. Accuracy: {accuracy_mini}")
    print("----------------------------------------------------------------------------------")

    if epoch == 0 or epoch ==1 :
        torch.save(model, 'last_model.pth')
        continue
    output=torch.squeeze(outputs)
    for i in output:
      if torch.isnan(i):
        model = torch.load('last_model.pth')
        quit()
    if ( losses_val[-1] - losses_val[-3]<=10e-6 ) :
        count_stop+=1
        if count_stop == patience :
            model = torch.load('last_model.pth')
            break
    else:
        count_stop = 0
        torch.save(model, 'last_model.pth')

a = [i for i in range(count_batch)]
# plt.plot(a,losses_val)
plt.plot(a,losses)

a = [i for i in range(count_batch)]
plt.plot(a,losses_val)

def precision_recall_f1_score(cnf_matrix):
    # dự đoán đúng là 0 / dự đoán là 0
    prec = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])
    # dự đoán đúng là 0 / thực sự là 0
    rec = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
    fs = 2*((prec*rec)/(prec+rec))
    return prec, rec, fs

with torch.no_grad():
    outputs_test = torch.squeeze(model(X_test)) # dãy xác suất dự đoán vd(0.51 0.76 0.01)
    loss_test = loss_function(outputs_test, y_test)
            
    predicted = outputs_test.round().detach().numpy()  # làm tròn và chuyển thành array
    total_test = y_test.size(0) # số test sample
    correct_test = np.sum(predicted == y_test.detach().numpy()) # số sample dự đoán đúng
    accuracy_test = 100 * correct_test/total_test

    cnf_matrix = confusion_matrix(y_test, predicted)
    print("Confusion matrix:\n",cnf_matrix)
    print("\nTest-Loss: %.4f"%loss_test.item())
    print("\nAccuracy: %.4f"%accuracy_test)
    prec, rec, fs = precision_recall_f1_score(cnf_matrix)
    print("\nPrecision: %.4f"%prec)
    print("\nRecall: %.4f"%rec)
    print("\nf1-score: %.4f"%fs)