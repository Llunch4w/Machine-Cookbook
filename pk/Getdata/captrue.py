#获取网页
import requests
from queue import Queue
from bs4 import BeautifulSoup
import re
import sys
q = Queue()
response = requests.get("https://baidu.com")
maxNum = 0#结点数
pages = []#最多maxNum个元素
visited = []#已经进行过搜索的结点
curPageNum = 0
index = 1#此时文件尾号
#开始
def start(url):
    global q,pages,curPageNum,maxNum,visited
    curPageNum = 0
    pages = []
    visited = []
    #清空队列
    while not q.empty():
        q.get()
    q.put(url)
    pages.append(url)
    curPageNum += 1
    while not q.empty():
        temp = q.get()#当前队首元素
        if curPageNum >= maxNum:
            for i in range(maxNum):
                item = pages[i]
                if item not in visited:
                    oneLevelSearch(item)
                    visited.append(item)
            return
            
        html = cap_html(temp)
        parse_html(html,temp)
        visited.append(temp)
#获取网页
def cap_html(url):
    global response
    try:
        response = requests.get(url,timeout=5)
    except:
        print("果神tql")
    if response.status_code != 200:
        return ""
    response.encoding = "utf-8"
    html = response.text
    return html   
#解析网页
def parse_html(html,source):
    global maxNum,index,q,curPageNum
    if curPageNum >= maxNum:
        return
    pattern = re.compile(r'http[s]?://*')#不是网址不进队列
    file = "../data/url_" + str(index) + ".txt"
    f = open(file,'a',encoding='utf-8')
    soup = BeautifulSoup(html,from_encoding='utf-8',features="lxml")
    #超链接存在于a标签中的href属性下
    links = soup.find_all('a')
    #print("width:",width)
    width = len(links)
    for i in range(width):
        if 'href' not in links[i].attrs.keys():
            continue
        s = links[i]['href']
        flag = re.search(pattern,s)
        if flag:
            #print(s)
            sentence = source + "," + s + "\n"
            f.write(sentence)
            if s not in pages:
                q.put(s)
                #print(s)
                curPageNum += 1
                pages.append(s)
    f.close()
def oneLevelSearch(source):
    html = cap_html(source)
    pattern = re.compile(r'http[s]?://*')#不是网址不进队列
    file = "../data/url_" + str(index) + ".txt"
    f = open(file,'a',encoding='utf-8')
    soup = BeautifulSoup(html,from_encoding='utf-8',features="lxml")
    #超链接存在于a标签中的href属性下
    links = soup.find_all('a')
    width = len(links)
    #print("width:",width)
    for i in range(width):
        #print("i:",i)
        if 'href' not in links[i].attrs.keys():
            continue
        s = links[i]['href']
        flag = re.search(pattern,s)
        if flag:
            sentence = source + "," + s + "\n"
            #print(sentence)
            f.write(sentence)
    f.close()
    
if __name__ == "__main__":
    for i in range(46):
        if i <= 29:
            maxNum += 10
        if i > 29 and i <= 36:
            maxNum += 100
        if i > 36:
            maxNum += 1000
        start("https://baidu.com")
        file = "../data/url/url_" + str(index) + ".txt"
        f = open(file,'a',encoding='utf-8')
        s = "Nodes:"+str(maxNum)+"\n"
        f.write(s)
        f.close()
        print("finish...")
        index += 1