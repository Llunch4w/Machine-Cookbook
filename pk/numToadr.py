import csv
def NTA(i,Ifilename,Ofilename):
    #base_
    rank_dict = {}
    index_dict = {}
    with open(Ifilename,mode='r',encoding='utf-8') as f:
        reader = csv.reader(f,delimiter=',')
        fieldnames = next(reader)
        reader = csv.DictReader(f,fieldnames=fieldnames,delimiter=',')
        rank = 1
        for row in reader:
                rank_dict[rank] = row['index']
                rank += 1
    filename = f'data/dict/dict_tabel_{i}.csv'
    with open(filename,mode='r',encoding='utf-8') as f:
        reader = csv.reader(f,delimiter=',')
        fieldnames = next(reader)
        reader = csv.DictReader(f,fieldnames=fieldnames,delimiter=',')
        for row in reader:
                index_dict[row['index']] = row['address']
    with open(Ofilename,mode='w',newline='',encoding='utf-8') as f:
        fieldnames = ['rank','index','url']
        writer = csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        for rank,index in rank_dict.items():
            writer.writerow({'rank':rank,'index':index,'url':index_dict[index]})

if __name__ == "__main__":
    for i in range(1,47):
        Ifilename = f'rank/coo/rank_{i}.csv'
        Ofilename = f'rank/coo/final_{i}.csv'
        NTA(i,Ifilename,Ofilename)
    for i in range(1,47):
        Ifilename = f'rank/base/base_rank_{i}.csv'
        Ofilename = f'rank//base/base_final_{i}.csv'
        NTA(i,Ifilename,Ofilename)