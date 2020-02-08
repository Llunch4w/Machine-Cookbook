import cv2
import os
import time

face_cascade = cv2.CascadeClassifier(
'cascade_files/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_nose.xml')
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')
if nose_cascade.empty():
    raise IOError('Unable to load the nose cascade classifier xml file')


def save(name, frame):
    root_dir = "../images"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    dir_path = root_dir + '/' + name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = str(int(time.time()))
    file = dir_path + '/' + filename + '.jpg'
    cv2.imwrite(file, frame)


def capture(name):
    # 0输入参数指定网络摄像头的ID
    cap = cv2.VideoCapture(0)
    # 定义网络摄像头采集图像的比例系数
    scaling_factor = 1
    # 启动一个无限循环采集帧，直到按下Esc键
    while True:
        success, frame = cap.read()
        if success is not True:
            print("这一帧未捕捉到图片")
            continue
        # 调整帧的大小
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,
                            interpolation=cv2.INTER_AREA)
        # 将捕捉到的图像的灰度图保存下来
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 图像灰化
        save(name, frame)
        # 画图
        draw(frame)
        # 等待1ms，然后采集下一帧
        c = cv2.waitKey(1)
        if c == 27:
            break
    # 释放视频采集对象
    cap.release()
    # 在结束代码之前关闭所有活动窗体
    cv2.destroyAllWindows()


def draw(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 参数1.3是指每个阶段的乘积系数。参数5是指每个候选矩形应该拥有的最小近邻数量。候选矩形是指人脸可能被检测到的候选区域
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # 画出人脸
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,155,100), 3)
        # 从彩色和灰度图中提取人脸ROI信息
        roi_gray = gray[int(y+h/4):int(y+0.55*h), int(x+0.13*w):int(x+w-0.13*w)]
        roi_color_eye = frame[int(y+h/4):int(y+0.55*h), int(x+0.13*w):int(x+w-0.13*w)]
        # 从灰度图ROI信息中检测眼睛
        eye_rects = eye_cascade.detectMultiScale(roi_gray)
        # 检测鼻子
        roi_gray = gray[y:y+h, x:x+w]
        roi_color_nose = frame[y:y+h, x:x+w]
        nose_rects = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
        # 在眼睛周围画绿色的圆
        for (x_eye, y_eye, w_eye, h_eye) in eye_rects:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (255, 255, 0)
            thickness = 3
            cv2.circle(roi_color_eye, center, radius, color, thickness)
        # 在鼻子周围画矩形
        for (x_nose, y_nose, w_nose, h_nose) in nose_rects:
            cv2.rectangle(roi_color_nose, (x_nose, y_nose), (x_nose+w_nose,
                                                y_nose+h_nose), (0, 255, 0), 3)
            break
    cv2.imshow('Eye and nose detector', frame)


if __name__ == "__main__":
    capture("mg")
