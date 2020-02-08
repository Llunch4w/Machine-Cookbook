import keyboard
import time
from PIL import ImageGrab
import sys
import os

class ScreenShot(object):
    def __init__(self):
        self.cnt = 1

    def setSavePath(self,outfile,cnt):
        self.outdir = outfile
        self.cnt = cnt

    def cut(self):
        index = self.cnt
        self.cnt += 1
        if keyboard.wait(hotkey='ctrl+q') == None:
            # 读取剪切板里的图片
            time.sleep(1)
            img = ImageGrab.grabclipboard()
            # 保存
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)
            img.save('{}/{}.jpg'.format(self.outdir,index))

    def run(self):
        while True:
            print("截取一张图片并用ctrl+q保存")
            self.cut()
    

if __name__ == "__main__":
    test = ScreenShot()
    test.setSavePath('images',1)
    test.run()


    