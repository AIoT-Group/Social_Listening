import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('stopwords')
import csv
from nltk.corpus import brown
import re

# lấy danh sách stopword
stop = set(stopwords.words('english'))

def del_stopword(list_sentence):
    last_sentence = []
    for word in list_sentence:
        if (word not in stop) and (word in wordlist_lowercased):
            last_sentence.append(word)
    return last_sentence

# tạo kho từ tiếng anh + một số tên riêng quan trọng
wordlist_lowercased = set(i.lower() for i in brown.words())|{"biden","ukraine","kiev","kyiv","zelensky","putin"}
# tạo list lưu vocaburary
vocabulary=[]
# tạo file lưu data
f = open("C:/Users/User/Documents/Machine Learning/social listening project/data_stopword_2.csv",mode='w',encoding='utf-8')
f.close()
# mở file data và lấy dữ liệu dưới dạng list l
with open("C:/Users/User/Documents/Machine Learning/social listening project/DataUseful.csv",encoding="utf-8") as f:
    reader = csv.reader(f)
    l = [row for row in reader]

# lưu những rows dùng được đồng thời thêm từ mới vào vocab
with open("C:/Users/User/Documents/Machine Learning/social listening project/data_stopword_2.csv",mode="a",encoding='utf-8') as out:
    writer = csv.writer(out,delimiter=',',lineterminator='\n')
    count = 0
    for i in range(0,len(l)):
        sentence = re_clean(l[i][2])
        list_sentence = list(sentence.split())
        last_sentence = del_stopword(list_sentence)
        writer.writerow(last_sentence)
        print(i)
        # thêm những từ mới vào vocab
        for vocab in last_sentence:
            if vocab not in vocabulary:
                vocabulary.append(vocab)

f = open("C:/Users/User/Documents/Machine Learning/social listening project/Vocab2.csv",mode='w',encoding='utf-8')
f.close()
with open("C:/Users/User/Documents/Machine Learning/social listening project/Vocab2.csv",mode="a",encoding="utf-8") as out:
    writer = csv.writer(out,lineterminator='\n')
    for i in vocabulary:
        writer.writerow([i])