# coding:utf-8

import random
import time
from public.public_function import *
from public.cnn_single_keras_tensorflow import *
import copy

def get_start(a,b):         #返回一个（a,b）间的随机数
    a_min=min(a,b)
    a_max=max(a,b)
    return a_min+np.random.random()*(a_max-a_min)

def pictureSize_afterPool(picture_size,stride):  #计算池化后的图片尺寸
    return math.ceil(float(picture_size)/float(stride))

def startVariable():
    while True:
        # 第一个卷积层设置为[2，原图片尺寸的一半]中随机一个值
        conv1_kernel_size = int(get_start(2, 28 / 2))

        # 卷积层都设置成为步长为1，0填充边界，经过卷积层后输出的图片尺寸与原图片尺寸相同，只起到提取特征的目的
        # conv1_kernel_stride = random.choice(list(range(1, conv1_kernel_size + 1)))

        # 第一个卷积核的数目设置为[2,64]随机取一个
        conv1_kernel_numbers = int(get_start(2, 64))

        # 池化层用来进行降维，因为经过第一层卷积层后图片尺寸没变
        # 第一个池化层的尺寸设置为[2,原图片尺寸一半]随机取一个
        pool1_size = int(get_start(2, 28 / 2))

        # 池化层的步长设置为[2,池化层尺寸]中随机取一个，因为采取same，边界填充模式，不必担心窗口滑出界
        pool1_stride = int(get_start(2,pool1_size + 1))

        # 通过图片尺寸以及池化层的滑动步长，计算经过池化层降维之后，输出的图片的尺寸
        pictureSize_afterPool1 = pictureSize_afterPool(28, pool1_stride)

        # 这里进行防止报错判断，第二层卷积核尺寸设置为[2,输入图片尺寸的一半]
        a = int(pictureSize_afterPool1 / 2)
        if a <= 2:
            conv2_kernel_size = 2
        else:
            conv2_kernel_size = int(get_start(2, a))
        # conv2_kernel_stride = random.choice(list(range(1, conv2_kernel_size + 1)))

        # 第二层卷积核个数设为[第一个卷积核个数，128]随机取一个
        conv2_kernel_numbers = int(get_start(conv1_kernel_numbers, 128))

        # 第二个池化层尺寸设置为[2,输入图片尺寸的一半]随机取一个
        pool2_size = int(get_start(2, int(pictureSize_afterPool1 / 2 + 1)))

        # 第二个池化层步长设置为[2,池化层尺寸]中随机取一个
        pool2_stride = int(get_start(2, pool2_size + 1))

        # 通过第一层池化层的输出和第二个池化层的步长获得第二个池化层输出的图片尺寸
        pictureSize_afterPool2 = pictureSize_afterPool(pictureSize_afterPool1, pool2_stride)

        if pictureSize_afterPool2<=0:
            continue


        nodes_num=pictureSize_afterPool2 * pictureSize_afterPool2 * conv2_kernel_numbers
        if nodes_num>100:
            # 全连接层节点数设置为[100,第二个池化层输出图片的所有节点]
            fullconnection_numbers = int(get_start(100, nodes_num))
        else:
            fullconnection_numbers=nodes_num

        # 学习率设置为[0,0.01]中随机取一个，学习率较高极容易出现过拟合现象
        learning_rate = get_start(0,1)
        while learning_rate == 0:
            learning_rate = get_start(0, 1)

        dropout= get_start(0,1)
        while dropout == 0:
            dropout = get_start(0, 1)


        l = [conv1_kernel_size, conv1_kernel_numbers, pool1_size, pool2_stride, conv2_kernel_size, conv2_kernel_numbers,
             pool2_size, pool2_stride, fullconnection_numbers, learning_rate,dropout]

        break
    return l

class Firstpart_Individual():

   # x:包含个体全部信息，第一二卷积层卷积核尺寸以及卷积核个数，第一二层池化层尺寸和步长，全连接层节点数以及学习率
   # train_average_y_list:保存每一代模型在训练集上的输出矩阵
   # net_num:该代的第几个个体，用于确认保存的位置以及保存的模型名称

   def __init__(self, x, net_num,first_model_save_path):
       print('开始创建个体')

       # 种群的个体信息
       self.x = x

       # 记录该个体是此代中第几个个体
       self.net_num = net_num
       # 模型的保存位置以及保存的模型名称
       self.model_save_path = first_model_save_path
       self.model_name = 'model_' + str(self.net_num) + '.h5'

       # 根据个体，模型保存位置和模型名称构建网络

       # 返回来三个结果：
       # 1、训练集上的准确度
       # 2、训练集得到的分类矩阵
       print("即将进入CNN")
       c = cnn(self.x, model_save_path=self.model_save_path, model_name=self.model_name)
       self.acc, self.y_pre =c.cnn_run(1)
       if self.acc > 0.3:
           self.acc, self.y_pre = c.cnn_run(10)

       print('验证集准确率：', self.acc)


       # 将训练误差作为第一个目标函数

def create_folds(single_path,net_num):
    # 为每个个体创建对应的文件夹，以供后续保存模型
    if os.path.exists(os.path.join(single_path ,str(net_num))):
        print('文件夹已存在')
        deletefile(os.path.join(single_path,str(net_num)))
        print('文件夹内文件清空完毕')
    else:
        os.mkdir(os.path.join(single_path, str(net_num)))
        print('文件夹创建成功')


def Constraints(l_array,*x):
    # 第一卷积核尺寸

    for i in range(len(l_array)-2):
        l_array[i]=int(l_array[i])
    print("对除学习率进行取整")
    print(l_array)

    if l_array[0] < 1:
        l_array[0]=control(l_array[0],(28/2),1,0)
    print(l_array)
    if l_array[0] > (28 / 2):
        l_array[0] = control(l_array[0], (28 / 2), 1, 1)

    # 第一卷积核数量
    if l_array[1] < 6:
        l_array[1] = control(l_array[1], 64,6, 0)
    if l_array[1]>64:
        l_array[1] = control(l_array[1], 64, 6, 1)

    # 第一池化层尺寸
    if l_array[2] < 2:
        l_array[2] = control(l_array[2], (28 / 2), 2, 0)

    if l_array[2] > (28 / 2):
        l_array[2] = control(l_array[2], (28 / 2), 2, 1)

    # 第一池化层步长
    if l_array[3] < 2:
        l_array[3] = control(l_array[3], l_array[2], 2, 0)

    if l_array[3] > l_array[2]:
        l_array[3] = control(l_array[3], l_array[2], 2, 1)

    afterpool1=pictureSize_afterPool(28, l_array[3])

    size = int(afterpool1/ 2)

    # 第二卷积核尺寸
    if l_array[4] < 1:
        l_array[4] = control(l_array[4], size, 1, 0)

    if l_array[4] > size:
        l_array[4] = control(l_array[4], size, 1, 1)

    # 第二卷积核个数
    if l_array[5] < l_array[1]:
        l_array[5] = control(l_array[5], 128, l_array[1], 0)

    if l_array[5]>128:
        l_array[5] = control(l_array[5], 128, l_array[1], 1)

    # 第二池化层尺寸
    if l_array[6] < 2:
        l_array[6] = control(l_array[6], size, 2, 0)

    if l_array[6] > size:
        l_array[6] = control(l_array[6], size, 2, 1)

    # 第二个池化层步长
    if l_array[7] < 2:
        l_array[7] = control(l_array[7], l_array[6], 2, 0)

    if l_array[7] > l_array[6]:
        l_array[7] = control(l_array[7], l_array[6], 2, 1)

    afterpool2=pictureSize_afterPool(afterpool1,l_array[7])

    # 全连接层节点数
    if l_array[8] < 10:
        l_array[8] = control(l_array[8],min(afterpool2*afterpool2*l_array[5],3000),10,0)
    if l_array[8] > 3000:
        l_array[8] = control(l_array[8], min(afterpool2 * afterpool2 * l_array[5],3000), 10,1)

    l_x = []
    for i in x:
        print(i[9])
        l_x.append(np.abs(i[9]))
    learning_average = np.average(np.array(l_x))
    # 学习率
    if l_array[9] >= learning_average:

        l_array[9]=control(l_array[9], learning_average, 0, 1)
    if l_array[9] <= 0:
        l_array[9] = control(l_array[9], learning_average, 0, 0)
    if l_array[10] <= 0:
        l_array[10] = control(l_array[10], 1, 0, 0)
    if l_array[10] >= 1:
        l_array[10] = control(l_array[10], 1, 0, 1)

    # for i in range(len(l_array)):
    #     l_array[i]=int(l_array[i])
    l=[]
    for i in range(len(l_array)-2):
        l.append(math.floor(l_array[i]))
    l.append(l_array[9])
    l.append(l_array[10])
    print("加入限制之后：")
    print(l)
    return l

class Population():
    def __init__(self, first_single_path, single_temp_path):
        self.NP = 20       # 种群规模
        self.F = 0.5       # 变异的控制参数
        self.CR = 0.8      # 杂交的控制参数
        self.X = []        # 个体
        self.p=[]
        self.XMutation = []
        self.XCrossOver = []
        self.fitness_x = []
        self.model_save_path = first_single_path
        self.single_temp_path = single_temp_path

    def initialize(self):
        xTemp = []
        FitnessTemp = []
        pTemp=[]
        for i in range(self.NP):
            # 卷积核尺寸，卷积核个数，池化层尺寸，全连接层的节点数，学习率，权重，偏置
            l = startVariable()
            net_num = i
            create_folds(self.model_save_path,net_num)
            # 根据个体信息，创建对应的DBN
            net_i = Firstpart_Individual(l, net_num, self.single_temp_path)  # 初始化第一代网络
            print_l(net_i.x)
            # 将网络模型文件转移到对应的文件夹下，并将网络个体的模型文件位置属性进行修改
            Move_and_Alter_Model_Save_Path(net_i, self.model_save_path)
            xTemp.append(l)
            pTemp.append(net_i)
            FitnessTemp.append(1-net_i.acc)

        # xtemp 对应的权重  即单目标里面的个体
        # fitnesstemp 个体对应的适应度的值
        return xTemp, FitnessTemp,pTemp

    # 变异操作
    def Mutation(self,l_index):
        xTemp = self.X
        XMutationTemp = []

        for i in l_index:
            r1 = random.randint(0, self.NP - 1)
            r2 = random.randint(0, self.NP - 1)
            r3 = random.randint(0, self.NP - 1)
            XMutationTemp_row = []
            while (r1 == i or r2 == i or r3 == i or r1 == r2 or r1 == r3 or r2 == r3):
                r1 = random.randint(0, self.NP - 1)
                r2 = random.randint(0, self.NP - 1)
                r3 = random.randint(0, self.NP - 1)

            for j in range(len(self.X[0])):
                a = xTemp[r1][j] + self.F * (xTemp[r2][j] - xTemp[r3][j])
                XMutationTemp_row.append(a)
            XMutationTemp_row = Constraints(XMutationTemp_row,xTemp[r1],xTemp[r2],xTemp[r3])
            XMutationTemp.append(XMutationTemp_row)
        return XMutationTemp

    # 交叉操作
    def crossOver(self,l_index):
        XTemp = self.X
        XMutationTemp = self.XMutation
        XCrossOverTemp = []
        for i in range(len(l_index)):
            XCrossOverTemp_row = []
            for j in range(len(self.X[0])):
                rTemp = random.random()
                if (rTemp <= self.CR):
                    XCrossOverTemp_row.append(XMutationTemp[i][j])
                else:
                    XCrossOverTemp_row.append(XTemp[l_index[i]][j])
            XCrossOverTemp_row=Constraints(XCrossOverTemp_row,XMutationTemp[i],XTemp[l_index[i]])
            XCrossOverTemp.append(XCrossOverTemp_row)
        return XCrossOverTemp

    # 选择操作  贪婪选择策略
    def selection(self,l_index):
        XTemp = self.X
        pTemp=self.p
        XCrossOverTemp = self.XCrossOver  # 交叉变异后的个体
        FitnessTemp = self.fitness_x
        FitnessCrossOverTemp = []

        for i in range(len(l_index)):
            net_i = Firstpart_Individual(XCrossOverTemp[i], l_index[i], self.single_temp_path)
            print_l(net_i.x)
            FitnessCrossOverTemp.append(1-net_i.acc)
            if (FitnessCrossOverTemp[i] < FitnessTemp[l_index[i]]):
                print("第" + str(l_index[i]) + "个变异交叉后的个体的适应度为:" + str(FitnessCrossOverTemp[i]))
                print("原" + str(l_index[i]) + "个个体的适应度为:" + str(FitnessTemp[l_index[i]]))
                pTemp[l_index[i]] = copy.deepcopy(net_i)
                XTemp[l_index[i]] = XCrossOverTemp[i]
                FitnessTemp[l_index[i]] = FitnessCrossOverTemp[i]
                folder_create_or_clear(os.path.join(self.model_save_path,  str(l_index[i])))
                Move_and_Alter_Model_Save_Path(pTemp[l_index[i]], self.model_save_path)

                print(pTemp[l_index[i]].model_save_path)

                print("种群中第" + str(l_index[i]) + "个个体被变异交叉后的个体替换========================================"
                      "种群中第" + str(l_index[i]) + "个个体被变异交叉后的个体替换========================================"
                      "种群中第" + str(l_index[i]) + "个个体被变异交叉后的个体替换========================================"
                      )

            folder_create_or_clear(self.single_temp_path)

        # 变异交叉之后，经过选择之后的个体和适应度
        return XTemp, FitnessTemp, pTemp

    # 保存每一代最优值
    def saveBest(self):
        FitnessTemp = self.fitness_x
        a = min(FitnessTemp)
        a_index=FitnessTemp.index(a)
        print('第' + str(a_index) + '个体取得最小适应度为' + str(a))
        return a_index,a



def first_single(single_path,temp_path,min_acc):
    # 获取最后一代个体在训练集上的分类矩阵

    folder_create_or_clear(first_single_path)
    folder_create_or_clear(temp_path)

    print('开始创建个体')
    p = Population(single_path,temp_path)

    print('种群初始化')

    # 生成个体，并计算对应适应度的值
    p.X, p.fitness_x,p.p = p.initialize()
    # print(len(p.X))
    # a = []
    t1 = time.clock()
    dai = 1
    while True:

        l_p_index = []

        for i in range(len(p.p)):
            if p.p[i].acc < min_acc:
                l_p_index.append(i)

        print("第"+str(dai)+"代==============================================="
              "第" + str(dai) + "代==============================================="
              "第" + str(dai) + "代==============================================="
              "第" + str(dai) + "代==============================================="
              "第" + str(dai) + "代==============================================="
              "第" + str(dai) + "代==============================================="
              "第" + str(dai) + "代==============================================="
              "第" + str(dai) + "代==============================================="
              "第" + str(dai) + "代==============================================="
              )
        p.XMutation = p.Mutation(l_p_index)
        print("第"+str(dai)+"代变异完成")
        p.XCrossOver = p.crossOver(l_p_index)
        print("第"+str(dai)+"代交叉完成")
        p.X, p.fitness_x, p.p = p.selection(l_p_index)
        print("第"+str(dai)+"代选择完成")
        dai += 1

        num = 0
        for i in range(len(p.fitness_x)):
            if p.fitness_x[i] <= 1-min_acc:
                num += 1

        if num == len(p.fitness_x):
            break

    t2=time.clock()
    with open(single_path + "第一部分单目标.txt", 'w')as f:
        f.write(str(t2 - t1))
    print('======================')

    a,min_fitness = p.saveBest()
    variable = p.X[a]
    print('获得最高准确率的权重组合为', variable)

    return variable,min_fitness,p


# def get_first_part():
#     first_single_path = "E:\pc_model\\new\\"+date_str+"\population"
#     temp_path = "E:\pc_model\\new\\"+date_str+"\\temp"
#
#     best_variable, min_error, p = first_single(first_single_path, temp_path,0.99)
#
#     print("最优参数variable:", best_variable)
#     print("最高准确率:", (1 - min_error))
#
#     for i in range(len(p.p)):
#         print("参数:", p.p[i].x)
#         print("准确率:", p.p[i].acc)
#
#     with open(os.path.join(first_single_path,"最优参数variable.txt"),'w')as f:
#         f.write(str(best_variable))
#
#     with open(os.path.join(first_single_path,"all_variable.txt"),'w')as f:
#         f.write("参数:"+str(p.p[i].x)+","+"准确率:"+str(p.p[i].acc)+"!!!")
#
#     return p.p


if __name__ == "__main__":

    first_single_path="E:\pc_model\\new\\"+date_str+"\population"
    temp_path="E:\pc_model\\new\\"+date_str+"\\temp"
    best_variable,min_error,p=first_single(first_single_path,temp_path,0.90)
    print("最优参数variable:",best_variable)
    print("最高准确率:", (1-min_error))

    for i in range(len(p.p)):
        print("参数:",p.p[i].x)
        print("准确率:",p.p[i].acc)
