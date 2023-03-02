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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# lấy vocabulary
with open("/content/drive/MyDrive/NLP/Vocab2.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    vc = [row[0] for row in reader]
# lấy những vector embedding của vocabulary
with open("/content/drive/MyDrive/NLP/word_embedded_list.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    word_list = [np.array(row,dtype=np.float64) for row in reader]
# lấy các câu đã được chia nhỏ thành các từ
with open("/content/drive/MyDrive/NLP/data_stopword_2.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    dt_list=[row for row in reader]
# lấy label
with open("/content/drive/MyDrive/NLP/DataUseful.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    label_list = [row for row in reader]
    label_list = label_list
    # 0 = Ukraina and 1 = Nga
    for i in range(0,len(label_list)):
      if len(dt_list[i])==0:
        label_list.pop(i)
    for i in range(0,len(label_list)):
        if label_list[i][1]=="U":
            label_list[i]=0
        else: 
            label_list[i]=1
# có một số data sau khi bỏ stop word sẽ không còn lại gì, cần loại bỏ 
count=0
while count<len(dt_list):
  if (len(dt_list[count])==0):
    dt_list.pop(count)
    count=-1
  count+=1
# tạo dictionary tra cứu vector của các từ trong vocabulary
word_embedded_dict = dict(zip(vc, word_list))
# các câu có độ dài khác nhau nên chúng ta đưa hết về độ dài 50
data_list=[]
count=0
for i in range(len(dt_list)):
    lis=[]
    # cắt bớt những câu dài hơn 50
    if len(dt_list[i]) > 50:
        for j in range(50):
            lis.append(word_embedded_dict[dt_list[i][j]])
    # những câu dưới 50 từ được bổ sung những từ "ma" biểu diễn bởi vector 0
    else:
        for j in range(len(dt_list[i])):
            lis.append(word_embedded_dict[dt_list[i][j]])
        for j in range(len(dt_list[i]),50):
            lis.append([0]*100)
    data_list.append(lis)

X_train_val, X_test, y_train_val, y_test = model_selection.train_test_split(data_list, label_list, train_size=8/10,test_size=2/10, random_state=0)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train_val, y_train_val, train_size=7/8,test_size=1/8, random_state=0)
X_train, X_val, X_test = torch.tensor(X_train,dtype=torch.float32),torch.tensor(X_val,dtype=torch.float32),torch.tensor(X_test,dtype=torch.float32)
y_train, y_val, y_test = torch.tensor(y_train,dtype=torch.float32),torch.tensor(y_val,dtype=torch.float32),torch.tensor(y_test,dtype=torch.float32)

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
    
        self.hidden_dim = hidden_dim
        
        self.layer_dim = layer_dim
        
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim)
            
        out, hn = self.rnn(x, h0)
    
        outputs = torch.sigmoid(self.fc(hn)+1e-6)
        return outputs

epochs=400
#số featue của 1 từ
input_dim=len(X_train[0][0])
hidden_dim=10
layer_dim=1
output_dim=1
learning_rate = 0.01
patience = 5
batch_size=32
# chia batch_size
train = TensorDataset(X_train,y_train)
train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []
losses_val = []

count_batch = 0
count_epoch = 0
count_stop = 0
for epoch in tqdm((range(epochs)),desc='Training Epochs'):
    count_epoch+=1
    for i, (X_mini, y_mini) in enumerate(train_loader):
        count_batch+=1
    
        optimizer.zero_grad()   
        outputs = model(X_mini)  

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