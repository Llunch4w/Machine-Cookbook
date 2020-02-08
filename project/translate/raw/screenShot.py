import keyboard
import time
from PIL import ImageGrab
import sys

def screenShot(index):
    # if keyboard.wait(hotkey='c') == None:
    if keyboard.wait(hotkey='ctrl+s') == None:
        # 读取剪切板里的图片
        time.sleep(1)
        img = ImageGrab.grabclipboard()
        # 保存
        img.save('image/{}.jpg'.format(index))

if __name__ == "__main__":
    for i in range(1,sys.maxsize):
        print("截取一张图片")
        screenShot(i)