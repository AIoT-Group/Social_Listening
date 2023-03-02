import numpy as np
import csv

with open("C:/Users/User/Documents/Machine Learning/social listening project/tf_idf.csv", encoding="utf-8") as f:
    # đọc file vector, đưa về dạng array[float]
    reader = csv.reader(f)
    data_list = [row for row in reader]
    data_list = np.asarray(data_list, dtype=np.float64)
    # chia thành các tập 
    vector_training_set = np.array([i for i in data_list[:4000]])
    vector_validation_set = np.array([i for i in data_list[4001:5500]])
    vector_test_set = np.array([i for i in data_list[5501:]])
f.close()

with open("C:/Users/User/Documents/Machine Learning/social listening project/DataUseful.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    label_list = [row for row in reader]
    # 0 = Ukraina and 1 = Nga
    for i in range(0,len(label_list)):
        if label_list[i][1]=="U":
            label_list[i]=0
        else: 
            label_list[i]=1
    # chia thành các tập
    label_training_set = [i for i in label_list[:4000]]
    label_validation_set = [i for i in label_list[4001:5500]]
    label_test_set = [i for i in label_list[5501:]]
f.close()

X_train = vector_training_set
# thêm hệ số bias
X_train = np.concatenate((np.ones((X_train.shape[0],1)), X_train), axis = 1)
y_train = label_training_set

X_val = vector_validation_set
# thêm hệ số bias
X_val = np.concatenate((np.ones((X_val.shape[0],1)), X_val), axis = 1)
y_val = label_validation_set

X_test = vector_test_set
# thêm hệ số bias
X_test = np.concatenate((np.ones((X_test.shape[0],1)), X_test), axis = 1)
y_test = label_test_set

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def loss_function(w,X_val,y_val):
    loss = 0
    for i in range(0,len(y_val)):
        z = sigmoid(np.dot(w, X_val[i].T))
        loss = loss - (y_val[i]*np.log(z+1e-5)+(1-y_val[i])*np.log(1-z+1e-5))
    return loss

def logistic_sigmoid_regression(X_train, y_train, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]    # vector trọng số
    it = 0
    N = X_train.shape[0]  # số đối tượng
    d = X_train.shape[1]  # số feature trong một đối tượng
    count = 0   # đến số vong lặp
    check_w_after = 20  # check lại w sau mỗi 20 lần lặp
    while count < max_count:
        # mix data 
        mix_id = np.random.permutation(N)
        # cập nhật w sau mỗi vòng lặp
        for i in mix_id:
            xi = X_train[i]
            yi = y_train[i]
            zi = sigmoid(np.dot(w[-1], xi.T))
            w_new = w[-1] + eta*(yi - zi)*xi
            count += 1
            # điều kiện dừng vòng lặp
            if count%check_w_after == 0:  
                count2.append(count)
                loss_train_new = loss_function(w_new, X_train,y_train)
                loss_train.append(loss_train_new)
                loss_val_new = loss_function(w_new, X_val,y_val)
                loss_val.append(loss_val_new)
                if np.linalg.norm(w_new -  w[-check_w_after])<tol:
                    return w[-1]
            w.append(w_new)
    return w[-1]     

count2 = []
loss_train = []
loss_val = [] 
eta = .05   
d = X_train.shape[1]
w_init = np.random.randn(1, d)  # lấy random một vector w
w = logistic_sigmoid_regression(X_train, y_train, w_init, eta).reshape((1,-1))

a = [i for i in count2]
plt.plot(a,loss_train)
plt.plot(a,loss_val)

predict = sigmoid(np.dot(w, X_test.T))
count = 0
for i in range(0,len(predict[0])):
    x = predict[0][i]
    y = y_test[i]
    s = x + y
    if (x + y >=1.5) or (x + y <0.5):
        count+=1
    # print("%.4f" %predict[0][i],y_test[i])
print("Correct label",count)
print("Number of label",len(y_test))
print("Accuracy: ",count/len(y_test))
