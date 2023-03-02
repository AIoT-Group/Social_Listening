import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
import csv
# kho từ tiếng anh
import re
from nltk.corpus import brown

# lấy danh sách stopword
stop = set(stopwords.words('english'))
# gắn thẻ các loại từ
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None

# đưa từ về dạng gốc
def lemmatization(sentence):
    # gắn thẻ các từ 
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence)) 
    #print(pos_tagged)
    # chuyển các thẻ sang dạng dễ hiểu hơn, chi tiết thì print cái trên vs cái dưới rồi so sánh
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    #print(wordnet_tagged)
 
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # nếu từ đã có dạng nguyên thể và không phải stopword thì cứ thêm vào
            word_lemmed = word
        else:       
            # từ chưa có dạng nguyên thể và không phải stopword thì chuyển về nguyên thể rồi thêm
            word_lemmed = lemmatizer.lemmatize(word, tag)
        if (word_lemmed not in stop) and (word_lemmed in wordlist_lowercased):
            lemmatized_sentence.append(word_lemmed)
    return lemmatized_sentence

# tạo kho từ tiếng anh + một số tên riêng quan trọng
wordlist_lowercased = set(i.lower() for i in brown.words())|{"biden","ukraine","kiev","kyiv","zelensky","putin"}
# tạo list lưu vocaburary
vocabulary=[]
# tạo file lưu data mới
f = open("C:/Users/User/Documents/Machine Learning/social listening project/DataStopwordLemma.csv",mode='w',encoding='utf-8')
f.close()
# mở file data và lấy dữ liệu dưới dạng list 
with open("C:/Users/User/Documents/Machine Learning/social listening project/DataUseful.csv",encoding="utf-8") as f:
    reader = csv.reader(f)
    l = [row for row in reader]

# lưu những rows dùng được đồng thời thêm từ mới vào vocab
with open("C:/Users/User/Documents/Machine Learning/social listening project/DataStopwordLemma.csv",mode="a",encoding='utf-8') as out:
    writer = csv.writer(out,delimiter=',',lineterminator='\n')
    count = 0
    for i in range(0,len(l)):
        a=lemmatization(cham(l[i][2]))
        writer.writerow(a)
        print(i)
        # thêm những từ mới vào vocab
        for j in a:
            if j not in vocabulary:
                vocabulary.append(j)
# tạo và lưu file vocab
f = open("C:/Users/User/Documents/Machine Learning/social listening project/Vocab.csv",mode='w',encoding='utf-8')
f.close()
with open("C:/Users/User/Documents/Machine Learning/social listening project/Vocab.csv",mode="a",encoding="utf-8") as out:
    writer = csv.writer(out,lineterminator='\n')
    for i in vocabulary:
        writer.writerow([i])