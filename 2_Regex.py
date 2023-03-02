import re 
import csv

def clean(chuoi):
    # chuyển các từ viết tắt về dạng đầy đủ
    a="(’|')ve"
    b=' have'
    chuoi = re.sub(a, b, chuoi)
    a="y(’|')all"
    b='you all'
    chuoi = re.sub(a, b, chuoi)
    a="(’|')ll"
    b=' will'
    chuoi = re.sub(a, b, chuoi)
    a="(’|')m"
    b=' am'
    chuoi = re.sub(a, b, chuoi)
    a="(’|')d"
    b=' would'
    chuoi = re.sub(a, b, chuoi)
    a="(’|')re"
    b=' are'
    chuoi = re.sub(a, b, chuoi)
    a="n(’|')t"
    b=' not'
    chuoi = re.sub(a, b, chuoi)
    # xóa những ký tự không phải chữ
    pattern = "[^(\w| )]|[\d]"
    replace = ''
    chuoi = re.sub(pattern, replace, chuoi)
    return chuoi
    
# tạo file chứa data cleaned
f=open("C:/Users/User/Documents/Machine Learning/social listening project/DataCleaned.csv",mode='w',encoding='utf-8')
f.close()
# đọc file data raw
with open('C:/Users/User/Documents/Machine Learning/social listening project/DataCrawl.csv','r',encoding='utf-8') as f:
    reader=csv.reader(f)
    l=[row for row in reader]
# clean data và viết vào file data cleaned
with open('C:/Users/User/Documents/Machine Learning/social listening project/DataCleaned.csv',mode="a",encoding='utf-8') as out:
    writer=csv.writer(out,delimiter=',',lineterminator='\n')
    count = 0
    for i in range(1,len(l)):
        a = clean(l[i][1])
        if a.strip():
            count+=1
            writer.writerow([count,a])
        print(i)