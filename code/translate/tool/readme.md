**提前需要安装的库**
- keyboard
- pillow
- baidu-aip


**运行方法**  
- screenShot.py用于保存屏幕截图，运行此程序后，用任何截图软件截图，然后ctrl+s保存，图片即可保存到images文件夹中
- 运行recoger.py可对images文件夹中的所有文件进行识别，识别结果保存到recogs文件夹中
- 运行translates文件可对recogers文件夹中的所有文件进行翻译，默认是英译汉
- 其中oneTime.py是一个一次性使用的，即只翻一张图片，结过在temp文件夹中

**原理**    
基本是调用百度提供的API，其中pwd.ini文件中保存的是我百度云上申请接口的分配的id啥的

**参考资料**    
[b站视频--截屏+识别](https://www.bilibili.com/video/av29341898?from=search&seid=4947967871438546134)                   
[知乎回答--翻译](https://zhuanlan.zhihu.com/p/34659267) 

