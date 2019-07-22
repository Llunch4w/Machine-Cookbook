import re
import csv
def url_to_num(tail):
    nn_dict = {}
    pages = []
    if tail <= 30:
        maxNum = tail*10+1#标号的结点数+1
    if tail > 30 and tail <= 37:
        maxNum = 300 + (tail-30)*100+1
    if tail > 37 and tail <= 46:
        maxNum = 1000 + (tail-37)*1000+1
    noIndex = 0#不需要再编号标志
    index = 1
    filename = "../data/url/url_" + str(tail) + ".txt"
    f = open(filename,"r",encoding='utf-8')
    filename = "../data/std/data_" + str(tail) + ".txt"    
    output = open(filename,'w')
    filename = "../data/dict/dict_tabel_" + str(tail) + ".csv"
    dictFile = open(filename,mode='w',newline='',encoding='utf-8')
    fieldnames = ['index','address']
    dictWriter = csv.DictWriter(dictFile,fieldnames=fieldnames)
    dictWriter.writeheader()
    line = f.readline()
    while line:
        line = line.strip()
        sour = ""
        des = ""
        try:
            sour,des = line.split(',')
            if des == '':
                line = f.readline()
                continue
        except:
            print("果神tql")
            line = f.readline()
            continue
        
        if sour.endswith('/'):
            sour = sour[:-1]
        if des.endswith('/'):
            des = des[:-1]
        pattern = re.compile(r'www.')
        sour = re.sub(pattern,"", sour)
        des = re.sub(pattern, "", des)
        if noIndex == 0 :
            if sour not in pages:
                #print("sour not in pages:")
                pages.append(sour)
                nn_dict[sour] = index
                dictWriter.writerow({'index':index,'address':sour})
                index += 1
                if(index == maxNum):
                    noIndex = 1
            if des not in pages:
                #print("des not in pages:")
                pages.append(des)
                nn_dict[des] = index
                dictWriter.writerow({'index':index,'address':des})
                index += 1
                if(index == maxNum):
                    noIndex = 1
        if sour in nn_dict and des in nn_dict:
            s = str(nn_dict[sour]) + "," + str(nn_dict[des]) + "\n"
            output.write(s)
        line = f.readline()
        #print(pages)
    f.close()
    output.close()
    dictFile.close()

if __name__ == "__main__":
    for i in range(1,47):
        url_to_num(i)