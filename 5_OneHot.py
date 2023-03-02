from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import csv

with open("C:/Users/User/Documents/Machine Learning/social listening project/Vocab.csv",encoding="utf-8") as f:
    reader = csv.reader(f)
    l = [row for row in reader]
f.close()
vocab = []
for i in range(0,len(l)):
    vocab.append(l[i][0])

# in từng biến ra để hiểu code nhé
label_encoder = LabelEncoder()
# tạo mảng số nguyên chứa số thứ tự các từ trong V
integer_label_encoder = label_encoder.fit_transform(vocab)
# chuyển mảng trên thành ma trận cột
label_encoded = integer_label_encoder.reshape(len(integer_label_encoder),1)

onehot_encoder = OneHotEncoder(sparse=False)
# tạo ma trận với các hàng là các onehot vector
onehot_encoded = onehot_encoder.fit_transform(label_encoded)
print(onehot_encoded)