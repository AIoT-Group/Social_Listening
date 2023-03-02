import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection as model_selection
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

with open("C:/Users/User/Documents/Machine Learning/social listening project/tf_idf.csv", encoding="utf-8") as f:
    # đọc file vector, đưa về dạng array[float]
    reader = csv.reader(f)
    data_list = [row for row in reader]
    data_list = np.asarray(data_list,dtype=np.float64)

with open("C:/Users/User/Documents/Machine Learning/social listening project/DataUseful.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    label_list = [row for row in reader]
    label_list = label_list
    # 0 = Ukraina and 1 = Nga
    for i in range(0,len(label_list)):
        if label_list[i][1]=="U":
            label_list[i]=0
        else: 
            label_list[i]=1

X_train_val, X_test, y_train_val, y_test = model_selection.train_test_split(data_list, label_list, train_size=8/10,test_size=2/10, random_state=0)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train_val, y_train_val, train_size=7/8,test_size=1/8, random_state=0)

X_train, X_val, X_test = torch.tensor(X_train,dtype=torch.float32),torch.tensor(X_val,dtype=torch.float32),torch.tensor(X_test,dtype=torch.float32)
y_train, y_val, y_test = torch.tensor(y_train,dtype=torch.float32),torch.tensor(y_val,dtype=torch.float32),torch.tensor(y_test,dtype=torch.float32)

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim): # input_dim = 6329 output_dim = 1
        super(LogisticRegression, self).__init__()
        # z = w1x1 + w2x2 + ... + wnxn + b = hàm nn.linear
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # x = [5000 x 6329]
        # self.linear(x) = [5000 x 1] 
        # output = y_predict = sigmoid(self.linear(x)) = [5000 x 1]
        outputs = torch.sigmoid(self.linear(x))
        return outputs

epochs = 100
input_dim = len(X_train[0])
output_dim = 1 
learning_rate = 0.01

model = LogisticRegression(input_dim,output_dim)
loss_function = torch.nn.BCELoss() # binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []
losses_val = []
count_epoch = 0

for epoch in tqdm((range(epochs)),desc='Training Epochs'):
    count_epoch+=1
    
    optimizer.zero_grad()   
    outputs = model(X_train)  

    loss = loss_function(torch.squeeze(outputs), y_train) 
    loss.backward() 
    optimizer.step() 
    
    with torch.no_grad():   # không tính gradient
        # tính loss và độ chính xác cho test_set
        outputs_val = torch.squeeze(model(X_val)) # dãy xác suất dự đoán vd(0.51 0.76 0.01)
        loss_val = loss_function(outputs_val, y_val)
            
        predicted = outputs_val.round().detach().numpy()  # làm tròn và chuyển thành array
        total_val = y_val.size(0) # số test sample
        correct_val = np.sum(predicted == y_val.detach().numpy()) # số sample dự đoán đúng
        accuracy_val = 100 * correct_val/total_val
        losses_val.append(loss_val.item())
            
        # tính loss và độ chính xác cho training_set    
        outputs_train = torch.squeeze(model(X_train))   
        loss_train = loss_function(outputs_train, y_train)

        total_train = y_train.size(0)
        correct_train = np.sum(torch.squeeze(outputs_train).round().detach().numpy() == y_train.detach().numpy())
        accuracy_train = 100 * correct_train/total_train
        losses.append(loss_train.item())
            
        print(f"\nVal - Loss: {loss_val.item()}. Accuracy: {accuracy_val}")
        print(f"Train -  Loss: {loss_train.item()}. Accuracy: {accuracy_train}")
        print("----------------------------------------------------------------------------------")

a = [i for i in range(epoch+1)]
plt.plot(a,losses_val)
plt.plot(a,losses)

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
