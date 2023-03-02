import csv

# tạo file Data dùng được.csv
f = open("C:/Users/User/Downloads/DataLabel.csv",mode='w',encoding='utf-8')
f.close()
# mở file data và lấy dữ liệu dưới dạng list l
with open("C:/Users/User/Downloads/DataUseful.csv",encoding="utf8") as f:
    reader = csv.reader(f)
    l = [row for row in reader]
f.close()
# lưu những rows dùng được
with open('C:/Users/User/Downloads/DataUseful.csv',mode="a",encoding='utf-8') as out:
    writer = csv.writer(out,delimiter=',',lineterminator='\n')
    count = 0
    for i in range(1,len(l)):
        a = l[i][5]
        # lưu những data phe Nga
        if l[i][2]=="1":
            count+=1
            writer.writerow([count,"N",a.lower()])
        # lưu những data phe Ukraine
        if l[i][3]=="1":
            count+=1
            writer.writerow([count,"U",a.lower()])
        print(i)
