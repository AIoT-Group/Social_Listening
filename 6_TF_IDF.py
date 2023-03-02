import csv
import numpy as np

with open("C:/Users/User/Documents/Machine Learning/social listening project/DataStopwordLemma.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    data_list = [row for row in reader]
f.close()

with open("C:/Users/User/Documents/Machine Learning/social listening project/Vocab.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    vocab_set = []
    for i in reader:
        vocab_set.append(i[0])
f.close()

def tf(doc_list):
    tf_dict = {}
    # loại các từ giống nhau để đếm
    doc_set = set(doc_list)
    doc_dict = {}
    for i in doc_set:
        doc_dict[i] = 0
    for i in doc_list:
        doc_dict[i]+=1
    # tìm số lần xuất hiện nhiều nhất
    present_max = 0
    for i in doc_set:
        if doc_dict[i]>present_max:
            present_max = doc_dict[i]
    # tính tf = số lần xuất hiện/số lần xuất hiện của từ xuất hiện nhiều nhất
    for i in doc_list:
            tf_dict[i] = doc_dict[i]/present_max
    return tf_dict

# idf
idf_dict = {}
for i in vocab_set:
    present_vocab = 0
    for j in data_list:
        if i in j:
            present_vocab +=1
    idf_dict[i] = np.log(len(data_list)/present_vocab)  

def tf_idf(doc_list):
    tf_dict = tf(doc_list)
    tf_idf_list = [i for i in vocab_set]
    for i in range(0,len(vocab_set)):
        if vocab_set[i] not in doc_list:
            tf_idf_list[i] = 0
        else:
            tf_idf_list[i] = tf_dict[vocab_set[i]]*idf_dict[vocab_set[i]]
    return tf_idf_list

f = open("C:/Users/User/Documents/Machine Learning/social listening project/tf_idf.csv",mode='w',encoding='utf-8')
f.close()
with open("C:/Users/User/Documents/Machine Learning/social listening project/tf_idf.csv",mode="a",encoding="utf-8") as out:
    writer = csv.writer(out,delimiter=',',lineterminator='\n')
    for i in data_list:
        a = tf_idf(i)
        writer.writerow(a)
out.close()