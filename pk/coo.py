import math
import matplotlib.pyplot as plt
import time
import os,psutil
import csv

def buildGraph(filename):
    linkFrom = {}#{i:[...]}
    linkTo = {}#{i:[...]}
    nodes = {}#{i:pi}
    with open(filename,"r",encoding='utf-8') as f:
        line = f.readline()
        while line:
            src,des = [int(x) for x in line.split(",")]
            if src not in nodes.keys():
                nodes[src] = 0
            if des not in nodes.keys():
                nodes[des] = 0
            if not des in linkFrom.keys():
                linkFrom[des] = []
            if not src in linkTo.keys():
                linkTo[src] = []
            linkFrom[des].append(src)
            linkTo[src].append(des)
            line = f.readline()
    #初始权重1/N
    initWeight = 1/len(nodes)
    for x in nodes.keys():
        nodes[x] = 1/initWeight

    return linkFrom,linkTo,nodes

def isOK(nodes,nextNodes,allowedError):
    for index in nodes.keys():
        if math.fabs(nodes[index]-nextNodes[index]) > allowedError:
            return False
    return True

def pageRank(linkFrom,linkTo,nodes):
    dd_num = 0#迭代次数
    alpha = 0.8
    N = len(nodes.keys())
    allowedError = 1/(N*10000.0)
    oldNodes = {}
    while 1:
        for x in nodes.keys():
            oldNodes[x] = nodes[x]
        for index in nodes.keys():
            temp = 0
            if index in linkFrom.keys():
                for node in linkFrom[index]:
                    temp += alpha*nodes[node]*(1/len(linkTo[node]))
            nodes[index] = temp + (1-alpha)/N
        dd_num += 1
        if isOK(nodes,oldNodes,allowedError):
            break
    return nodes,dd_num

def test(n):
    testResult = []
    for i in range(1,n+1):       
        iFile = f"data/std/data_{i}.txt"
        linkFrom,linkTo,nodes = buildGraph(iFile)
        #记录xx数据规模下的时间空间
        theResult = {}
        theResult["node_num"] = len(nodes)#规模
        start = time.time()
        nodes,dd_num = pageRank(linkFrom,linkTo,nodes)
        end = time.time()
        theResult["time"] = end - start#时间
        process = psutil.Process(os.getpid())
        theResult["memory"] = process.memory_info().rss/1024#空间
        theResult["dd_num"] = dd_num#迭代次数
        testResult.append(theResult)#记录这次循环的结果
        #将排名写入文件
        nodes = sorted(nodes.items(),key=lambda x:x[1],reverse=True)
        oFile = f"rank/coo/rank_{i}.csv"
        with open(oFile,mode='w',newline='') as f:
            rank_writer = csv.writer(f,delimiter=',')
            rank_writer.writerow(['index','weight'])
            for item in nodes:
                rank_writer.writerow(list(item))
    #将不同规模下的时间空间写入文件
    oFile = f"res.csv"
    with open(oFile,mode='a',newline='') as f:
        fieldnames = ["node_num","time","memory","dd_num"]
        res_writer = csv.DictWriter(f,fieldnames = fieldnames)
        res_writer.writeheader()
        for item in testResult:
            res_writer.writerow(item)
    
    
if __name__ == "__main__":
    test(46)