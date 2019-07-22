import numpy as np
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_classifier(classifier,x,y,title):
    # 定义图形的取值范围
    x_min,x_max = min(x[:,0])-1.0,max(x[:,0])+1.0
    y_min,y_max = min(x[:,1])-1.0,max(x[:,1])+1.0
    # 预测值表示我们在图形中想要使用的数值范围，我们增加了一些余量，例如上述代码中的1.0
    
    # 为了画出边界，还需要利用一组网格数据求出方程的值，然后把边界画出来
    step_size = 0.01 # 网格步长 
    x_values,y_values = np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size)) # 定义网格
    
    # 计算出分类器对所有数据点的分类结果
    mesh_output = classifier.predict(np.c_[x_values.ravel(),y_values.ravel()]) # 分类器输出结果
    mesh_output = mesh_output.reshape(x_values.shape) # 数组维度变形
    ## np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
    ## np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
    ## ravel()将多维数组降位一维
    
    # 用彩色区域画出各个类型的边界
    plt.figure()
    plt.pcolormesh(x_values,y_values,mesh_output,cmap=plt.cm.gray) # 选择配色方案
    
    # 接下来再把训练数据点画在图上
    plt.scatter(x[:,0],x[:,1],c=y,s=80,edgecolors='blue',linewidth=1,cmap=plt.cm.Paired)
    # c=y表示颜色使用顺序（y对应的是label），用目标标记映射cmap的颜色表
    plt.xlim(x_values.min(),x_values.max())
    plt.ylim(y_values.min(),y_values.max())
    
    plt.xticks(np.arange(int(x_min),int(x_max),1.0))
    plt.yticks(np.arange(int(y_min),int(y_max),1.0))

    plt.title(title)
    plt.show()

def plot_2class(x_data,y_data):
    class_0 = np.array([x_data[i] for i in range(len(x_data)) if y_data[i]==0])
    class_1 = np.array([x_data[i] for i in range(len(x_data)) if y_data[i]==1])
    plt.figure()
    plt.scatter(class_0[:,0],class_0[:,1],facecolors='r',s=5)
    plt.scatter(class_1[:,0],class_1[:,1],facecolors='y',s=5)
    plt.show()


def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Paired)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def count_for_report(y_test,y_test_pred):
    # accuracy = 100.0 * (y_test==y_test_pred).sum()/y_test.shape[0]
    # print('Accuracy of the classifier = ',round(accuracy,2),'%')
    from sklearn.model_selection import cross_val_score

    num_validation = 5
    accuracy = cross_val_score(classifier_guassiannb,x_data,y_data,scoring='accuracy',cv=num_validation)
    print('Accuracy:',round(100*accuracy.mean(),2),'%')

    # 用前面的方程分别计算精度，召回率和F1得分
    precision = cross_val_score(classifier_guassiannb,x_data,y_data,scoring='precision_weighted',cv=num_validation)
    print('Precision:',round(100*precision.mean(),2),'%')

    recall = cross_val_score(classifier_guassiannb,x_data,y_data,scoring='recall_weighted',cv=num_validation)
    print('Recall:',round(100*recall.mean(),2),'%')

    f1 = cross_val_score(classifier_guassiannb,x_data,y_data,scoring='f1_weighted',cv=num_validation)
    print('F1:',round(100*f1.mean(),2),'%')


def report(y_true,y_pred,target_names,title):
    from sklearn.metrics import classification_report
    # target_names = ['Class-' + str(int(i)) for i in set(y_data)]
    print('\n','#'*30)
    print('\n',title,'\n')
    print('\n',classification_report(y_true,y_pred,target_names=target_names),'\n')
    print('\n','#'*30)


def data_split(x_data,y_data,size):
    x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=size,random_state=5)
    return x_train,x_test,y_train,y_test


def simple_classify(x,y):
    # 初始花一个逻辑回归分类器
    classifier = linear_model.LogisticRegression(solver='liblinear',C=10)
    # solver用于设置求解系统方程的算法类型
    # 参数C表示正则化强度，数值越小，正则化强度越高

    # 训练分类器
    classifier.fit(x,y)
    return classifier


def bayes_classify(x,y):
    # 建立一个朴素贝叶斯分类器
    classifier_guassiannb = GaussianNB() # GaussianNB()指定了正态分布朴素贝叶斯模型
    classifier_guassiannb.fit(x,y)
    return classifier_guassiann

def forest_classify(x,y):
    # 建立随机森林分类器
    params = {'n_estimators':200,'max_depth':8,'random_state':7}
    classifier = RandomForestClassifier(**params)
    classifier.fit(x,y)
    return classifier

def testone(classifier,input_data):
    count = 0
    input_data_encoded = [-1]*len(input_data)
    for i,item in enumerate(input_data):
        contain = []
        contain.append(item)
        if item.isdigit():
            input_data_encoded[i] = int(item) # 注意，如果不化为int型则会报错
        else:    
            input_data_encoded[i] = int(label_encoder[count].transform(contain)) 
            count += 1
        
    input_data_encoded = np.array(input_data_encoded).reshape(1,-1)

    output_class = classifier.predict(input_data_encoded)
    print('Output class:',label_encoder[-1].inverse_transform(output_class))


def encode(x_data):
    # 将字符串转化为数值
    label_encoder = []
    x_encoded = np.empty(x_data.shape)
    for i,item in enumerate(x_data[0]):
        if item.isdigit():
            x_encoded[:,i] = x_data[:,i]
        else:
            label_encoder.append(preprocessing.LabelEncoder())
            x_encoded[:,i] = label_encoder[-1].fit_transform(x_data[:,i])
            
    x_values = x_encoded[:,:-1].astype(int)
    y_values = x_encoded[:,-1].astype(int)
    print(y_values)
    return x_values,y_values,label_encoder


def valid_n_estimate():
    from sklearn.model_selection import validation_curve

    classifier = RandomForestClassifier(max_depth=4,random_state=7)
    parameter_grid = np.linspace(25,200,8).astype(int) # [25,200]分成8等分
    train_scores,validation_scores = validation_curve(classifier,x_values,y_values,'n_estimators',parameter_grid,cv=5)
    print('###### VALIDATIONCURVES ######')
    print('Param:n_estimators\nTraining scores:\n',train_scores)
    print('Param:n_estimators\nValidation scores:\n',validation_scores)

    plt.figure()
    plt.plot(parameter_grid,100*np.average(train_scores,axis=1))
    plt.title('Training curve')
    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy')
    plt.show()


def valid_max_depth():
    from sklearn.model_selection import validation_curve

    classifier = RandomForestClassifier(n_estimators=20,random_state=7)
    parameter_grid = np.linspace(2,10,5).astype(int)
    train_scores,validation_scores = validation_curve(classifier,x_values,y_values,'max_depth',parameter_grid,cv=5)
    print('###### VALIDATIONCURVES ######')
    print('Param:max_depth\nTraining scores:\n',train_scores)
    print('Param:max_depth\nValidation scores:\n',validation_scores)
    plt.figure()
    plt.plot(parameter_grid,100*np.average(validation_scores,axis=1))
    plt.title('Validation curve')
    plt.xlabel('Maximum depth of the tree')
    plt.ylabel('Accuracy')
    plt.show()


def valid_learn_curve():
    from sklearn.model_selection import learning_curve

    classifier = RandomForestClassifier(random_state=7)
    parameter_grid = np.linspace(200,1100,4).astype(int)
    train_sizes,train_scores,validation_scores = learning_curve(classifier,x_values,y_values,train_sizes=parameter_grid,cv=5)
    print('##### LEARING CURVE ####')
    print('Training scores:\n',train_scores)
    print('Validation scaores\n',validation_scores)
    plt.figure()
    plt.plot(parameter_grid,100*np.average(train_scores,axis=1))
    plt.title('Learning curve')
    plt.xlabel('Number of training sample')
    plt.ylabel('Accuracy')
    plt.show()


