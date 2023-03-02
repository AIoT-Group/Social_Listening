import torch
import gensim.downloader as api
import csv
# import model
from gensim.models.word2vec import Word2Vec

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# phần code này trở đi chúng tôi dùng colab 
# 2 dòng sau có chức năng liên kết colab với drive
from google.colab import drive
drive.mount('/content/drive')
# tải kho data text8
corpus = api.load('text8')
# huấn luyện model bằng dữ liệu text8
model = Word2Vec(corpus)
# tải kho từ của chúng ta
with open("/content/drive/MyDrive/Vocab.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    vocab = [row[0] for row in reader]
vocab_text8 = model.wv.vocab

# mỗi từ biểu diễn bằng vector kích thước 100 nên 
# phải tạo một vector 100 con 0 để biểu diễn cho 
# những từ không có trong kho từ của họ
zero_list = [0]*100
f = open("/content/drive/MyDrive/word_embedded_list.csv",mode='w',encoding='utf-8')
f.close()
with open("/content/drive/MyDrive/word_embedded_list.csv",mode="a",encoding='utf-8') as out:
    writer = csv.writer(out,delimiter=',',lineterminator='\n')
    for i in vocab:
        if i in vocab_text8:
            writer.writerow(list(model.wv.get_vector(i)))
        else:
            # từ không có trong text8 thì là vector 0
            writer.writerow(zero_list)