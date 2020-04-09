
# coding:utf-8

# 卷积核尺寸，卷积核个数，池化层尺寸，全连接层的节点数，学习率，权重，偏置

import copy
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from public.public_function import *

from public.cnn_single_keras_tensorflow import *
# from multi_part.first_part import *
from keras.models import load_model

from _1_first_part.first_part import Constraints,control


class Multi_Individual():

   # x:包含个体全部信息，第一二卷积层卷积核尺寸以及卷积核个数，第一二层池化层尺寸和步长，全连接层节点数以及学习率
   # train_average_y_list:保存每一代模型在训练集上的输出矩阵
   # net_num:该代的第几个个体，用于确认保存的位置以及保存的模型名称
   def __init__(self, x, net_num,initial_p=None):
       print('开始创建个体')
       if initial_p == None:
           # 种群的个体信息
           self.x = x
           # 用于保存个体对应的两个目标函数值
           self.f = []
           # 记录该个体是此代中第几个个体
           self.net_num = net_num
           # 模型的保存位置以及保存的模型名称
           self.model_save_path = temppath
           self.model_name = 'model_' + str(self.net_num) + '.h5'
           # 根据个体，模型保存位置和模型名称构建网络

           # 返回来三个结果：
           # 1、训练集上的准确度
           # 2、训练集得到的分类矩阵
           print("即将进入CNN")
           c = cnn(self.x, model_save_path=self.model_save_path, model_name=self.model_name)
           acc, y_prediction =c.cnn_run(1)
           if acc > 0.3:
               acc, y_prediction = c.cnn_run(10)
           print("网络训练完毕")

           print('验证集集准确率：', acc)
           self.y_pre = y_prediction

           # 将训练误差作为第一个目标函数
           f1 = 1 - acc
           self.f.append(f1)

           # 初始化的第二个目标函数，这里先用100占位
           f2 = 100
           self.f.append(f2)

           with open(os.path.join(self.model_save_path, "second.txt"), 'w')as f:
               f.write(str(self.f[1]) + ",")


       else:
           # 种群的个体信息
           self.x = initial_p.x
           # 用于保存个体对应的两个目标函数值
           self.f = []
           # 记录该个体是此代中第几个个体
           self.net_num = net_num
           # 模型的保存位置以及保存的模型名称
           self.model_save_path = os.path.join(cluster_path,str(self.net_num))
           self.model_name = 'model_' + str(self.net_num) + '.h5'
           self.acc=initial_p.acc
           self.y_pre=initial_p.y_pre

           # 将训练误差作为第一个目标函数
           f1 = 1 - self.acc
           self.f.append(f1)

           # 初始化的第二个目标函数，这里先用0占位
           f2 = 100
           self.f.append(f2)

           with open(os.path.join(self.model_save_path,"second.txt"),'w')as f:
               f.write(str(self.f[1])+",")



def matrix_to_number(x):
    x_num=np.argmax(x,axis=1)
    return x_num

def calculate_P(p):
    p_l = []
    for i in range(len(p)):
        p_l.append(p[i].y_pre)
    p_sum = 0
    for i in range(len(p_l)):
        p_sum += p_l[i]

    P=matrix_to_number(p_sum)

    return P

def firstPart_single_or_semeble_SecondFunction(p, k=None, y=None):
    # 非变异个体的第二个目标函数的计算。。包括一代每个个体的第二个目标函数计算，以及对邻域个体的第二个目标函数的重新计算
    if y==None:
        P = calculate_P(p)
        p_l = []
        for i in range(len(p)):
            p_l.append(matrix_to_number(p[i].y_pre))

        # 计算一个集合中所有个体的第二个目标函数
        if k == None:
            for i in range(len(p_l)):
                pi = np.sum(p_l[i] == P) / len(P)
                pj_sum = 0
                for j in range(len(p_l)):
                    if i == j:
                        continue
                    else:
                        pj_sum += np.sum(p_l[j] == P) / len(P)
                pj_sum_average = pj_sum
                p[i].f[1] = pi*pj_sum_average
            return p

        # 计算单个个体的第二个目标函数
        else:
            pk = np.sum(p_l[k] == P) / len(P)
            pj_sum = 0
            for j in range(len(p_l)):
                if j==k:
                    continue
                pj_sum += np.sum(p_l[j] == P) / len(P)
            pj_sum_average = pj_sum
            return pk * pj_sum_average

    # 计算处于第一阶段的变异个体的第二个目标函数
    else:
        p[k] = y
        P = calculate_P(p)
        p_l = []
        for i in range(len(p)):
            p_l.append(matrix_to_number(p[i].y_pre))

        pk = np.sum(p_l[k] == P) / len(P)
        pj_sum = 0
        for j in range(len(p_l)):
            if j == k:
                continue
            pj_sum += np.sum(p_l[j] == P) / len(P)
        pj_sum_average = pj_sum
        return pk * pj_sum_average

# 初始化种群及对应的权重向量
# 输入 N：种群个体数量
# 返回：种群p,和对应的权重向量Lamb
def Initial(N,initial_p):

    # p用来保存种群个体
    p = []
    # Lamb保存的是对应的权重向量
    Lamb = []

    for i in range(N):
        temp = []
        # 卷积核尺寸，卷积核个数，池化层尺寸，全连接层的节点数，学习率，权重，偏置
        l = initial_p[i].x
        net_num = i

        # 清空模型保存的一手文件夹
        # 所有模型都是先保存在该文件夹中，然后根据一定规则将该文件夹中的模型文件复制到指定文件夹，后清空该文件夹。等待接受下一个模型。
        if os.path.exists(temppath):
            deletefile(temppath)
        else:
            os.makedirs(temppath)

        # # 根据个体信息，创建对应的CNN
        net_i = Multi_Individual(l,net_num,copy.deepcopy(initial_p[i]))  # 初始化第一代网络

        print_l(net_i.x)

        p.append(net_i)

        # 为个体随机创建权重向量，向量里包含元素的个数与目标函数个数相同，此处为2个。
        mmm = i/N
        temp.append(mmm)
        temp.append(1.0 - mmm)

        # 用列表Lamb保存权重向量，个体的坐标与对应的权重向量坐标是对应的。
        Lamb.append(temp)

    print('第一代网络模型保存位置')
    for i in range(len(p)):
        print(p[i].model_save_path)

    return p, Lamb   #返回种群，及对应的权重向量


def rndG(a,b):
    max_ = max(a,b)
    min_ = min(a,b)
    return np.random.normal(0, (1 / 20) * (max_-min_))

def mutation_Gaussian(l):
    conv1_size_Gaussian = rndG(28/2,1)
    conv1_numbers_Gaussian = rndG(2,64)
    pool1_size_Gaussian = rndG(2,28/2)
    pool1_stride_Gaussian = rndG(2,l[2])

    p1 = pictureSize_afterPool(28,l[3])

    conv2_size_Gaussian = rndG(2,int(p1/2))
    conv2_numbers_Gaussian = rndG(l[1],128)
    pool2_size_Gaussian = rndG(2,int(p1/2))
    pool2_stride_Gaussian = rndG(2,l[6])
    p2=pictureSize_afterPool(p1,l[7])
    fullconnection_Gaussian = rndG(10,p2*p2*l[5])
    learning_rate_Gaussian=rndG(0,1)
    dropout_Gaussian=rndG(0,1)
    return np.array([conv1_size_Gaussian,conv1_numbers_Gaussian,pool1_size_Gaussian,pool1_stride_Gaussian,conv2_size_Gaussian,conv2_numbers_Gaussian,pool2_size_Gaussian,pool2_stride_Gaussian,fullconnection_Gaussian,learning_rate_Gaussian,dropout_Gaussian])

# 变异获得新的个体
# c为原有个体，a,b为它邻域里面的两个个体，
def GeneticOperation(a, b, c, k):
    # 此处F设置为0.5
    F = 0.5

    # 在0，1范围内生成一个服从均匀分布的随机数
    j = np.random.uniform(0, 1)
    print("变异控制参数"+str(j))

    # 如果随机变量小于等于0.5，在原有个体加上控制参数乘以邻域个体的差，再加上后面的随机变量e

    a_array=np.array(a.x)
    b_array=np.array(b.x)
    c_array=np.array(c.x)
    print("邻域个体1:",a_array)
    print("邻域个体2:",b_array)
    print("本体：",c_array)

    l_array = c_array + F *(a_array  - b_array)

    if j <= 0.5:
        l_array = Constraints(l_array,a_array,b_array,c_array)
        Gaussian = mutation_Gaussian(l_array)
        l_array+=Gaussian

    # 卷积核尺寸，卷积核个数，池化层尺寸，全连接层的节点数，学习率，权重，偏置
    l_array=Constraints(l_array,a_array,b_array,c_array)
    print_l(l_array)
    return Multi_Individual(l_array,k)

# 计算邻域
# 输入：权重向量Lamb，邻域个数T
# 返回：距离每个向量最近的T个向量的索引的列表
def Neighbor(Lamb, T):

    #为每个权重向量，计算对应的T个邻域
    B = []
    for i in range(len(Lamb)):
        temp = []
        for j in range(len(Lamb)):
            distance = np.sqrt((Lamb[i][0]-Lamb[j][0])**2+(Lamb[i][1]-Lamb[j][1])**2)
            temp.append(distance)

        # temp中存放的是种群中第i个个体与其他个体之间的距离
        # 对距离进行排序，并且将其对应个体的坐标存放在l列表中
        l = np.argsort(temp)

        # 取前T个个体   B中存放的是距离每个个体最近的T个个体的坐标
        B.append(l[:T])
    return B  #得到每个权重向量的T个邻域

def min_distance(p, l):
    d = []
    for i in range(len(p)):
        d.append(p[i].f[0]*l[0]+p[i].f[1]*l[1])
    return np.argmin(d)

def BestValue(p):
    # 获取每个目标函数的最优值
    best = []
    for i in range(len(p[0].f)):
        best.append(p[0].f[i])
    for i in range(1,  len(p)):
        for j in range(len(p[i].f)):
            if p[i].f[j] < best[j]:
                best[j] = p[i].f[j]
    return best


def max_rPlace(l, z_1, z):
    l_1 = l[0] * np.abs(z_1[0] - z[0])
    l_2 = l[1] * np.abs(z_1[1] - z[1])
    return max(l_1, l_2)

def update_bestvalue(z,y,min_value):
    # step 2.7) Updating
    # 更新到目前为止的两个目标函数的最优值
    flag = False
    if (1-y.f[0]) > min_value:
        for j in range(len(z)):
            if y.f[j] < z[j]:
                z[j] = y.f[j]
                if flag == False:
                    flag = True
    else:
        pass
    return flag

# 输入为N,T,G
# N 种群个体数量
# T 邻域个数
# G 进化的代数
def MOEAD(N, T, G, initial_p, path, min_value):  # 种群数量和邻域个数

    # step 1)

    # step 1.1)
    # 初始化种群及对应的权重向量，以及“候选集成模型个体集合”
    p, Lamb = Initial(N,initial_p)
    print('种群数量', len(p))
    print('种群初始化完毕')

    # 计算初代所有个体第二个目标函数值
    p=firstPart_single_or_semeble_SecondFunction(copy.deepcopy(p))
    update_second(p)
    functions_print(p)

    # step 1.2)
    # 获取当前两个目标函数的最小值，参考点
    z = BestValue(p)
    with open(os.path.join(path,"bestvalue.txt"),'w')as f:
        f.write(str(z)+",")

    print('当前BestValue：', z)

    # step 1.3)
    # 根据权重向量计算对应的T个邻域
    B = Neighbor(Lamb, T)
    # step 1.4) 标准化部分   没有
    # step 2)
    # 进化G代
    t = 0
    while (t < G):

        # step 2.1) 标准化部分   没有
        t += 1
        for i in range(len(p)):
            if update_bestvalue(z, p[i],min_value):
                print("参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新"
                      "参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新")
                with open(os.path.join(path,"bestvalue.txt"),'a')as f:
                    f.write(str(z)+",")

        for i in range(N):
            # step 2.2) Reproduction
            # step 2.3) Repairing
            # step 2.4) Evaluation 这三部分包含在i中
            # 为个体i在其邻域随机选取两个个体
            k = random.randint(0, T - 1)
            l = random.randint(0, T - 1)
            print('从第'+str(t)+"代" + str(i) + '个个体选取邻域为:' + str(k) + ', ' + str(l))
            # 根据原有个体i,以及随机选取的它邻域的两个个体变异出一个新的个体
            y = GeneticOperation(p[B[i][k]], p[B[i][l]], p[i], i)

            y.f[1] = firstPart_single_or_semeble_SecondFunction(copy.deepcopy(p), i, y)
            with open(os.path.join(y.model_save_path, 'second.txt'), 'a')as f:
                f.write(str(y.f[1]) + ",")

            bianyigeti_print(y, t, i)

            print('变异结束')

            # step 2.5)标准化部分  没有

            print(B[i])

            # step 2.6) Replacement
            # 此处进行replacement，对个体的邻域的每个个体，如果变异出来的个体满足条件，则用变异个体将邻域个体进行替换
            for j in B[i]:

                # 获取邻域元素的权重向量
                Lamb_j = Lamb[j]

                # 重新计算邻域中待更新个体的第二个目标函数
                p[j].f[1] = firstPart_single_or_semeble_SecondFunction(copy.deepcopy(p),j)

                # 用变异个体代替待更新个体，计算变异个体的第二个目标函数
                # 计算变异个体与除待更新个体外的其他个体之间的联系
                y.f[1] = firstPart_single_or_semeble_SecondFunction(copy.deepcopy(p), j, y)

                # 获取变异个体两个目标函数距离最优值的最大值
                y_ = max_rPlace(Lamb_j, y.f, z)

                # 获取当前 邻域个体两个目标函数距离最优值的最大值
                j_ = max_rPlace(Lamb_j, p[j].f, z)

                # 如果变异个体小，则对邻域个体进行替换
                if y_ <= j_ and (1-y.f[0]) > min_value:

                    # 用变异个体模型文件对原有文件进行替换
                    deletefile(p[j].model_save_path)
                    print("权重向量",Lamb_j)
                    print("变异个体目标函数值",y.f)
                    print("邻域目标函数值", p[j].f)

                    print("===================邻域个体替换=============================="
                          "===================邻域个体替换=============================="
                          "===================邻域个体替换=============================="
                          "===================邻域个体替换=============================="
                          "===================邻域个体替换=============================="
                          "===================邻域个体替换=============================="
                          "===================邻域个体替换=============================="
                          "===================邻域个体替换==============================")
                    print('第' +str(t)+"代第"+ str(i) + '个个体的邻域：第' + str(j) + '个个体 模型文件删除成功')
                    movefiles(y.model_save_path, p[j].model_save_path)
                    print('第' +str(t)+'代用变异个体模型文件对第' + str(i) + '个个体的邻域：第' + str(j) + '个个体 模型文件替换成功')

                    # 用变异个体对原有个体的邻域个体进行替换，但是模型的保存位置不变
                    # 将保存种群个体的列表进行同步更新，将对应位置的个体替换为变异个体，并且修改变异个体的model_save_path属性
                    p[j] = copy.deepcopy(y)
                    p[j].model_save_path = path + str(j) + "\\"
                else:
                    print('第' +str(t)+'代' + str(j) + '个体不满足替换要求')

            if update_bestvalue(z,y,min_value):
                print('第' +str(t)+"代参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新"
                      "参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新参考点更新")
                with open(os.path.join(path,"bestvalue.txt"),'a')as f:
                    f.write(str(z)+",")

            # 变异个体操作全部完成，则将变异个体保存的模型文件进行删除
            deletefile(y.model_save_path)

            print('变异个体文件删除==================================')

            # step 2.8）在以上步骤中包含，并没有保存每一代的每一个个体，都是在第一代中不断进行替换，直到最后一代
    return p

def move_first_part_to_multi_part(multi_path, first_path):
    for i in os.listdir(first_path):
        old_path=os.path.join(first_path,str(i))
        new_path=os.path.join(multi_path,str(i))
        if os.path.exists(new_path):
            os.rmdir(new_path)
        shutil.copytree(old_path,new_path)
    print("文件复制完毕")

class p():
    def __init__(self, i):
        path = os.path.join(cluster_path, str(i))
        model_path = os.path.join(path, getFilename(path))
        model = load_model(model_path)
        last_layer = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
        y_prediction = last_layer.predict(validation_x_all)
        self.y_pre = y_prediction
        self.net_num = i
        l = os.listdir(path)
        # print(l)
        with open(os.path.join(path, l[1]), 'r')as f:
            acc = float(f.read().strip())
        self.acc = acc

        with open(os.path.join(path, l[2]), 'r')as f:
            l_content = f.read().strip()[1:-1].split(",")
            for i in range(len(l_content) - 2):
                l_content[i] = int(l_content[i])
            l_content[-2] = float(l_content[-2])
            l_content[-1] = float(l_content[-1])

        self.x = l_content

def read_first_part():
    l_p=[]
    for i in range(20):
        p_i=p(i)
        l_p.append(p_i)
    return l_p

if __name__ == "__main__":
    start = time.clock()
    print(cluster_path)
    folder_create_or_clear(cluster_path)
    # 将文件复制过来
    move_first_part_to_multi_part(cluster_path, "E:\pc_model\\new\\"+date_str+"\population")

    single_p=read_first_part()
    print("第一阶段单目标完毕")

    # （1）训练程序从此处进入
    # 进入MOEAD多目标程序  位于train_moea_cnn的MOEAD
    # 第一个参数：种群个体数，第二个参数：邻域数量，第三个参数：进化代数，第四个参数：模型保存位置，外部存储集合的模型保存位置
    p = MOEAD(20, 6, 3, single_p, cluster_path ,0.98)
    end = time.clock()

    multi_time = str(end-start)
    with open(cluster_path+"time.txt", 'w')as f:
        f.write(multi_time)

    with open(cluster_path+"all_outep_variables.txt", "w")as f:
        for i in range(len(p)):
            if i==(len(p)-1):
                f.write(str(p[i].x))
            else:
                f.write(str(p[i].x)+"$$$")

    print('多目标训练部分耗时' + multi_time + '秒')