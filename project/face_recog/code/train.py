import os

import cv2
import numpy as np
from sklearn import preprocessing

# 读取中文名字问题
def cv_imread(file_path = ""):
    img_mat=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),0)
    return img_mat

# 定义一个类来处理与类标签编码相关的所有任务
class LabelEncoder(object):
    # 将单词转换成数字的编码方法
    def encode_labels(self, label_words):
        self.labels = label_words
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label_words)
    # 将输入单词转换成数字
    def word_to_num(self, label_word):
        if label_word not in self.labels:
            return -1
        return int(self.le.transform([label_word])[0])
    # 定义一个方法，用于将数字转换为原始单词
    def num_to_word(self, label_num):
        if label_num >= len(self.labels):
            return -1
        return self.le.inverse_transform([label_num])[0]
    

# 用于从输入文件夹中提取图像和标签
def get_images_and_labels(input_path):
    label_words = []

    # 对输入文件夹做递归迭代并追加文件
    dirs = os.listdir(input_path)
    for dir_name in dirs:
        dir_name = input_path + '/' + dir_name
        files = os.listdir(dir_name)
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = dir_name + '/' + filename
            label_words.append(filepath.split('/')[-2]) 
            
    # 初始化变量
    images = []
    le = LabelEncoder()
    le.encode_labels(label_words)
    labels = []

    # 解析输入目录
    dirs = os.listdir(input_path)
    for dir_name in dirs:
        print(dir_name)
        dir_name = input_path + '/' + dir_name
        files = os.listdir(dir_name)
        for filename in (x for x in files if x.endswith(('.jpg','.jpeg','.png'))):
            filepath = dir_name + '/' + filename
            print(filepath)
            # 将当前图像读取成灰度格式
            # image = cv2.imread(filepath,0)
            image = cv_imread(filepath)
            # 提取标签
            name = filepath.split('/')[-2]
            try:
                faceCascade.detectMultiScale(image, 1.1, 2, minSize=(100,100))
            except:
                continue
            else:
                # 对该图像进行人脸检测
                faces = faceCascade.detectMultiScale(image, 1.1, 2, minSize=(100,100))
                # 如果未检测到人脸
                if len(faces) == 0:
                    print("error:",filepath)
                    continue
                # 循环处理每一张脸
                for (x, y, w, h) in faces:
                    images.append(image[y:y+h, x:x+w])
                    labels.append(le.word_to_num(name))
            
            
    return images, labels, le    

if __name__=='__main__':
    cascade_path = "cascade_files/haarcascade_frontalface_alt.xml"
    path_train = '../images'
    path_test = '../test'
    # 加载人脸级联文件
    faceCascade = cv2.CascadeClassifier(cascade_path)
    # 生成局部二值模式直方图人脸识别器对象
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 从训练数据集中提取图像、标签和标签编码器
    images, labels, le = get_images_and_labels(path_train)
    # 训练人脸识别器
    print("\nTraining...")
    recognizer.train(images, np.array(labels))
    # 用未知数据检测人脸识别器
    print('\nPerforming prediction on test images...')
    stop_flag = False
    for root, dirs, files in os.walk(path_test):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = root + '/' + filename
            # 读取图像
            predict_image = cv2.imread(filepath, 0)
            # 检测人脸
            faces = faceCascade.detectMultiScale(predict_image, 1.1, 
                    2, minSize=(100,100))
            if len(faces) == 0:
                continue
            # 循环处理每一张脸
            for (x, y, w, h) in faces:
                # 预测输出
                predicted_index, conf = recognizer.predict(
                        predict_image[y:y+h, x:x+w])
                print(conf)
                if(conf < 30):
                    predicted_person = "unknown"
                else:
                # 将标签转换成单词
                    predicted_person = le.num_to_word(predicted_index)
                # 在输出图像中叠加文字，并显示图像
                cv2.putText(predict_image, ': ' + predicted_person, 
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)
                cv2.imshow("Recognizing face", predict_image)

            c = cv2.waitKey(0)
            if c == 27:
                stop_flag = True
                break

        if stop_flag:
            break