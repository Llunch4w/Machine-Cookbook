from aip import AipOcr
import configparser
import os

class Recoger(object):
    def __init__(self):
        target = configparser.ConfigParser()
        target.read('pwd.ini')
        AppId = target.get('RecogSdk','APPID')
        APIKey = target.get('RecogSdk','APIKey')
        SecretKey = target.get('RecogSdk','SecretKey')

        # 类内都可用
        self.client = AipOcr(AppId,APIKey,SecretKey)

    def setPath(self,infile,outfile):
        self.indir = infile
        self.outdir = outfile

    def picture2Text(self):
        if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)

        files = os.listdir(self.indir)
        cnt = 0
        for file in files:
            cnt += 1
            outfile = open('{}/{}.txt'.format(self.outdir,cnt),'w')
            filePath = self.indir + '/' + file
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


if __name__ == "__main__":
    test = Recoger()
    test.setPath('images','recogs')
    test.picture2Text()