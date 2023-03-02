from selenium import webdriver
import time
import csv
import chromedriver_binary
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from datetime import date,timedelta
from urllib.parse import quote
from googletrans import Translator

# cài đặt thời gian lấy dữ liệu từ ngày hôm nay 
until=date.today()
since= until-timedelta(days=1)
# nhập từ khóa muốn tìm kiếm trên twitter
keyword = input("keyword = ")
# chọn browser là Chrome
browser = webdriver.Chrome()
# tạo file data crawl
f=open("C:/Users/User/Documents/Machine Learning/social listening project/DataCrawl.csv",mode='w',encoding='utf-8')
f.close()
len_file = 0
tran=Translator()
# vòng lặp lưu data vào file
with open("C:/Users/User/Documents/Machine Learning/social listening project/DataCrawl.csv",mode='a',encoding = 'utf-8') as out:
    writer=csv.writer(out,delimiter=',',lineterminator='\n')
    writer.writerow(['stt','noi dung'])
    # lấy khoảng 50.000 tweet
    while(len_file<50000):
        url = "https://twitter.com/search?q={}%20until%3A{}%20since%3A{}&src=typed_query&f=live".format(quote(keyword), until, since)
        browser.get(url)
        htmlElem = browser.find_element(by=By.TAG_NAME, value='html')
        # tạo list chứa 20 tweets ngay trước để so sánh, tránh lặp 
        contented = [] 
        len_day = 0
        # vòng lặp lấy dữ liệu theo mỗi 2 ngày, mỗi lần lấy khoảng 500 data
        while(len_day<500):
            try:
                # tìm tweets trong trang hiện tại
                local_contents = browser.find_elements(by=By.CSS_SELECTOR,value='[data-testid="tweetText"]')
                # so sánh với data trước đó để tránh lặp dữ liệu
                local_contents = [i.text for i in local_contents if i.text not in contented] 
            except:
                browser.implicitly_wait(3)
                continue
            # cập nhật list các tweet trước
            if len(contented)>20:
                del contented[0:len(local_contents)]
            contented.extend(local_contents)
            for i in local_contents:
                len_file+=1
                len_day+=1
                # dịch các tweet không phải tiếng anh
                writer.writerow([len_file,tran.translate(str(i)).text])
                # writer.writerow([len_file,i])
                print(len_file)
            # lệnh cuộn trang web xuống
            htmlElem.send_keys(Keys.PAGE_DOWN)
            htmlElem.send_keys(Keys.PAGE_DOWN)
            htmlElem.send_keys(Keys.PAGE_DOWN)
        #cập nhật 1 ngày trước đó
        until = until - timedelta(days=1)
        since = since - timedelta(days=1)  
