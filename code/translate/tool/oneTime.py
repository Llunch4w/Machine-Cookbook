import keyboard
import time
from PIL import ImageGrab
import sys

from aip import AipOcr
import configparser

import json
import random
import requests
import urllib.parse
from hashlib import md5

target = configparser.ConfigParser()
target.read('pwd.ini')
AppId = target.get('RecogSdk','APPID')
APIKey = target.get('RecogSdk','APIKey')
SecretKey = target.get('RecogSdk','SecretKey')
client = AipOcr(AppId,APIKey,SecretKey)

target_f = configparser.ConfigParser()
target_f.read('pwd.ini')
appId_f = target_f.get('FanyiSdk','APPID')
secretKey_f = target_f.get('FanyiSdk','SecretKey')

def screenShot():
    if keyboard.wait(hotkey='ctrl+s') == None:
        # 读取剪切板里的图片
        time.sleep(1)
        img = ImageGrab.grabclipboard()
        # 保存
        img.save('temp/grab.jpg')

def recog():
    with open("temp/grab.jpg",'rb') as f:
        img = f.read()
        # 识别图片
        outfile = open('temp/recog.txt','w')
        outfile.close()
        outfile = open('temp/recog.txt','a')
        text = client.basicGeneral(img)
        for item in text['words_result']:
            outfile.write(item['words'] + '\n')
        

def fanyi(text):
    base_url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    q = text
    fromLang = 'en'
    toLang = 'zh'
    salt = random.randint(32768, 65536)
    sign = appId_f + q + str(salt) + secretKey_f
    m1 = md5()
    m1.update(sign.encode("utf-8"))
    sign = m1.hexdigest()

    myurl = (base_url+'?appid='+ appId_f
    + '&q='+ urllib.parse.quote(q) + '&from='
    + fromLang + '&to=' + toLang+'&salt='+ str(salt) + '&sign=' + sign)

    response = requests.get(myurl)
    rans_result = json.loads(response.text)['trans_result'][0]['dst']
    return rans_result

if __name__ == "__main__":
    while True:
        print("截图并用Ctrl+s保存")
        screenShot()
        recog()
        readfile = open('temp/recog.txt','r')
        outfile = open('temp/fanyi.txt','w')
        outfile.close()
        outfile = open('temp/fanyi.txt','a')
        for line in readfile:
            text = line.strip()
            res = fanyi(text)
            time.sleep(1)
            outfile.write(res+'\n')
        readfile.close()
        outfile.close()







