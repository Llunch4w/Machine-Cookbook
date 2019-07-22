import numpy as np
from numpy import zeros
import time
import operator
import os,psutil
import csv

def buildGraph(file):
    all_x = []
    all_y = []
    pages = []
    f = open(file,encoding='UTF-8')
    line = f.readline()
    while line:
        tempx,tempy = line.split(',')
        all_x.append(int(tempx))
        all_y.append(int(tempy))
        if int(tempx) not in pages:
            pages.append(int(tempx))
        if int(tempy) not in pages:
            pages.append(int(tempy))
        line = f.readline()
    return all_x,all_y,pages

def setM(pageNum,all_x,all_y):#构造pageNum*pageNum的二维数组M
    M = []
    for _ in range(pageNum):
        temp = []
        for _ in range(pageNum):
            temp.append(0)
        M.append(temp)
    #求出各结点出度
    count_dic = {}#字典
    for item in all_x:
        count_dic[item] = count_dic.get(item,0)+1  #get函数，返回键item的值，如果不在coint_dic中返回0
    for i in range(len(all_x)):
        M[all_y[i]-1][all_x[i]-1] = 1/count_dic[all_x[i]]
    return M

def pageRank(pageNum,all_x,all_y):
    dd_num = 0
    a = 0.85
    M = setM(pageNum,all_x,all_y)
    pr0=1/pageNum
    pr=zeros((pageNum,1),dtype=float)+pr0#构造pr值的矩阵
    #i = 0
    while(((pr==(a*np.dot(M,pr)+(1-a)*pr0)).all())==False):   #判断pr值是否收敛
        #i=i+1
        pr=a*np.dot(M,pr)+(1-a)*pr0  #np.dot 矩阵对应相乘
        dd_num += 1
    temp = {}
    for i in range(len(pr)):
        temp[i+1] = float(pr[i])
    return temp,dd_num

def test(n):
    testResult = []
    for i in range(1,n+1):       
        iFile = f"data/std/data_{i}.txt"
        all_x,all_y,pages = buildGraph(iFile)
        #记录xx数据规模下的时间空间
        theResult = {}
        pageNum = len(pages)
        theResult["node_num"] = pageNum#规模
        start = time.time()
        nodes,dd_num = pageRank(pageNum,all_x,all_y)
        end = time.time()
        theResult["time"] = end - start#时间
        process = psutil.Process(os.getpid())
        theResult["memory"] = process.memory_info().rss/1024#空间
        theResult["dd_num"] = dd_num#迭代次数
        testResult.append(theResult)#记录这次循环的结果
        #将排名写入文件
        nodes = sorted(nodes.items(),key=lambda x:x[1],reverse=True)
        oFile = f"rank/base/base_rank_{i}.csv"
        with open(oFile,mode='w',newline='') as f:
            rank_writer = csv.writer(f,delimiter=',')
            rank_writer.writerow(['index','weight'])
            for item in nodes:
                rank_writer.writerow(list(item))
    #将不同规模下的时间空间写入文件
    oFile = f"base_res.csv"
    with open(oFile,mode='a',newline='') as f:
        fieldnames = ["node_num","time","memory","dd_num"]
        res_writer = csv.DictWriter(f,fieldnames = fieldnames)
        res_writer.writeheader()
        for item in testResult:
            res_writer.writerow(item)

if __name__ == "__main__":
    test(46)