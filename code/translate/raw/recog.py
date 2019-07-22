from aip import AipOcr
import configparser


class BaiDuAPI(object):
    def __init__(self):
        target = configparser.ConfigParser()
        target.read('pwd.ini')
        AppId = target.get('MySdk','APPID')
        APIKey = target.get('MySdk','APIKey')
        SecretKey = target.get('MySdk','SecretKey')

        # 类内都可用
        self.client = AipOcr(AppId,APIKey,SecretKey)

    def picture2Text(self,filePath):
        outfile = open('out.txt','w')
        img = self.getPicture(filePath)
        # 识别图片
        text = self.client.basicGeneral(img)
        for item in text['words_result']:
            outfile.write(item['words']+'\n')
        outfile.close()

    # 读取图片
    @staticmethod # 静态方法
    def getPicture(filePath):
        with open(filePath,'rb') as f:
            return f.read()

baidu_api = BaiDuAPI()
baidu_api.picture2Text('image/1.jpg')
