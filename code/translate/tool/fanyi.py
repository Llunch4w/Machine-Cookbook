import configparser
import json
import random
import requests
import urllib.parse
from hashlib import md5
import os
import time

class FanYi(object):
    def __init__(self):
        target = configparser.ConfigParser()
        target.read('pwd.ini')
        self.appId = target.get('FanyiSdk','APPID')
        self.secretKey = target.get('FanyiSdk','SecretKey')

    def getUrl(self,text):
        base_url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
        q = text
        fromLang = 'en'
        toLang = 'zh'
        salt = random.randint(32768, 65536)
        sign = self.appId + q + str(salt) + self.secretKey
        m1 = md5()
        m1.update(sign.encode("utf-8"))
        sign = m1.hexdigest()

        myurl = (base_url+'?appid='+ self.appId 
        + '&q='+ urllib.parse.quote(q) + '&from='
        + fromLang + '&to=' + toLang+'&salt='+ str(salt) + '&sign=' + sign)

        return myurl

    def translate(self):
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        files = os.listdir(self.indir)
        cnt = 0
        for file in files:
            cnt += 1
            outfile = open('{}/{}.txt'.format(self.outdir,cnt),'a')
            filePath = self.indir + '/' + file

            infile = open(filePath,'r')
            for line in infile:
                text = line.strip()
                myurl = self.getUrl(text)
                response = requests.get(myurl)
                rans_result = json.loads(response.text)['trans_result'][0]['dst']
                outfile.write(rans_result+'\n')
                time.sleep(1)
            infile.close()

            outfile.close()
        

    def setPath(self,infile,outfile):
        self.indir = infile
        self.outdir = outfile


if __name__ == "__main__":
    test = FanYi()
    test.setPath('recogs','translates')
    test.translate()
