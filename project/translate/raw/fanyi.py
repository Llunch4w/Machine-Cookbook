import configparser
import json
import random
import requests
import urllib.parse
from hashlib import md5

class BaiDuAPI(object):
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

    def translate(self,text):
        myurl = self.getUrl(text)
        response = requests.get(myurl)
        rans_result = json.loads(response.text)['trans_result'][0]['dst']
        return rans_result


if __name__ == "__main__":
    baidu_api = BaiDuAPI()
    res = baidu_api.translate('you are a really cool man!')
    print(res)

