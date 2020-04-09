# coding:utf-8

import random

import numpy as np
from public.public_function import *
from public.keras_load_data import *
import time
import pandas as pd
from keras.models import load_model,Model
from _2_multi_moead.multi_moead_cluster import firstPart_single_or_semeble_SecondFunction

# 计算两个矩阵的相似程度
def cal_acc(my_y, y):
    t1=time.clock()
    my_y_array = np.argmax(my_y,axis=1)
    t2=time.clock()
    # print("取最大值索引耗时"+str(t2-t1))
    y_array = np.argmax(y,axis=1)
    t3 = time.clock()
    # print("取最大值索引耗时" + str(t3 - t2))
    acc = np.sum(my_y_array == y_array)/len(y_array)
    t4 = time.clock()
    # print("求准确率耗时" + str(t4 - t3))
    return 1-acc

# l 权重
# l_y 分类矩阵列表
def calculateFitness(l, l_y, flag):
    # t1=time.clock()

    for i in range(len(l)):
        l[i]=float(l[i]/sum(l))

    final_y = 0

    for i in range(len(l)):
        final_y += l[i] * l_y[i]

    final_y_to = to_one_zero(final_y)

    if flag == 'train':
        return cal_acc(final_y_to, train_y_all)
    elif flag == 'test':
        return cal_acc(final_y_to, test_y_all)
    elif flag == 'validation':
        return cal_acc(final_y_to, validation_y_all)
    elif flag == "ensemble":
        return cal_acc(final_y_to, train_validation_y_all)


class Population():
    def __init__(self, l,way):
        self.NP = 10       # 种群规模
        self.size = len(l) #最后一代分类矩阵个数
        self.xMin = 0      # 最小值
        self.xMax = 1      # 最大值
        self.F = 0.5       # 变异的控制参数
        self.CR = 0.8      # 杂交的控制参数
        self.f = l         # 最后一代个体的训练集得到的分类矩阵
        self.X = []        # 个体
        self.XMutation = []
        self.XCrossOver = []
        self.fitness_x = []
        self.way=way

    def initialize(self):
        xTemp = []
        FitnessTemp = []
        l_random=[]
        for i in range(self.NP):

            # 创建一个个体
            l_final_w = []
            for j in range(self.size-1):
                w = np.random.randint(1, 100)
                l_final_w.append(w)
            wi=np.random.randint(sum(l_final_w),self.size*100)
            l=list(set(list(range(0,self.size)))^set(l_random))
            if l!=[]:
                x=np.random.choice(l)
                l_random.append(x)
                l_final_w.insert(x,wi)
            else:
                l_final_w.append(wi)
            # l_final_w.append(wi)
            xTemp.append(l_final_w)
            # 计算适应值
            FitnessTemp.append(calculateFitness(xTemp[i], self.f, self.way))

        # xtemp 对应的权重  即单目标里面的个体
        # fitnesstemp 个体对应的适应度的值
        return xTemp, FitnessTemp

    # 变异操作

    def Mutation(self):
        xTemp = self.X
        XMutationTemp = []
        for i in range(self.NP):
            r1 = random.randint(0, self.NP - 1)
            r2 = random.randint(0, self.NP - 1)
            r3 = random.randint(0, self.NP - 1)
            XMutationTemp_row = []
            while (r1 == i or r2 == i or r3 == i or r1 == r2 or r1 == r3 or r2 == r3):
                r1 = random.randint(0, self.NP - 1)
                r2 = random.randint(0, self.NP - 1)
                r3 = random.randint(0, self.NP - 1)

            for j in range(self.size):
                a = np.abs(xTemp[r1][j] + self.F * (xTemp[r2][j] - xTemp[r3][j]))
                XMutationTemp_row.append(a)
            XMutationTemp.append(XMutationTemp_row)
        return XMutationTemp

    # 交叉操作
    def crossOver(self):
        XTemp = self.X
        XMutationTemp = self.XMutation
        XCrossOverTemp = []
        for i in range(self.NP):
            XCrossOverTemp_row = []
            for j in range(self.size):
                rTemp = random.random()
                if (rTemp <= self.CR):
                    XCrossOverTemp_row.append(XMutationTemp[i][j])
                else:
                    XCrossOverTemp_row.append(XTemp[i][j])
            XCrossOverTemp.append(XCrossOverTemp_row)
        return XCrossOverTemp

    # 选择操作  贪婪选择策略
    def selection(self):
        XTemp = self.X
        XCrossOverTemp = self.XCrossOver  # 交叉变异后的个体
        FitnessTemp = self.fitness_x
        FitnessCrossOverTemp = []
        for i in range(self.NP):
            FitnessCrossOverTemp.append(calculateFitness(XCrossOverTemp[i], self.f, self.way))
            if (FitnessCrossOverTemp[i] < FitnessTemp[i]):
                XTemp[i] = XCrossOverTemp[i]
                FitnessTemp[i] = FitnessCrossOverTemp[i]
        # 变异交叉之后，经过选择之后的个体和适应度
        return XTemp, FitnessTemp

    # 保存每一代最优值
    def saveBest(self):
        FitnessTemp = self.fitness_x
        a = min(FitnessTemp)
        a_index = FitnessTemp.index(a)
        print('第' + str(a_index) + '个体取得最小适应度为' + str(a))
        return a_index, a

def to_one_zero(l):
    a = pd.get_dummies(l.argmax(1))
    return np.array(a)

def single_moead_w(P, g,way):
    # 获取最后一代个体在训练集上的分类矩阵
    final_generation_matrix = []
    for i in range(len(P)):
        final_generation_matrix.append(to_one_zero(P[i].y_pre))

    # print('开始创建个体')
    p = Population(final_generation_matrix,way)
    gen = 0
    maxCycle = g
    # print('种群初始化')

    # 生成个体，并计算对应适应度的值
    p.X, p.fitness_x = p.initialize()
    print("初始fitness",p.fitness_x)

    while gen < maxCycle:
        p.XMutation = p.Mutation()
        # print('变异完成')
        p.XCrossOver = p.crossOver()
        # print('交叉完成')
        p.X, p.fitness_x = p.selection()
        # print('选择完成')
        # print('第%s代' % (gen))
        gen += 1

    print(p.fitness_x)
    a,min_fitness = p.saveBest()
    zuhe_w = p.X[a]
    print('获得最高准确率的权重组合为', zuhe_w)
    # zuhe_train_acc = calculateFitness(zuhe_w, final_generation_matrix, 'train')
    # print('组合权重后训练集得到的(1-准确率)为:', zuhe_train_acc)
    return zuhe_w, min_fitness

class geti_ea():
    def __init__(self,model_save_path):
        self.model_save_path = model_save_path
        model = load_model(self.model_save_path)
        last_layer = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
        y_prediction = last_layer.predict(validation_x_all)
        self.y_pre = y_prediction
        self.f = []

        with open(os.path.join(os.path.dirname(self.model_save_path), "train_acc.txt"), "r")as f:
            acc = float(f.read().strip())
        f1 = 1-acc
        self.f.append(f1)
        f2 = 0
        self.f.append(f2)

def get_P(model_save_path):
    # print("进入outep函数")
    l = []
    for file in model_save_path:
        model_name = getFilename(file)
        # print("获取文件名称")
        model_path = os.path.join(file, model_name)
        l.append(geti_ea(model_path))
        # print("添加个体")
    l = firstPart_single_or_semeble_SecondFunction(l)
    # print("计算目标函数值")
    return l

if __name__ == "__main__" :
    while True:
        select = input("s开始,q退出")
        if select=="s":
            path = []
            for i in range(20):
                path.append(os.path.join(cluster_path,str(i)))
            ea_p = get_P(path)
            file_path = os.path.join(cluster_path, "final_generation")

            zuhe_final_w, min_fitness = single_moead_w(ea_p, 1000, "validation")

            folder_create_or_clear(file_path)

            with open(os.path.join(file_path, "final_zuhe_w.txt"), "w")as f:
                f.write(str(zuhe_final_w))
        elif select=='q':
            break
        else:
            continue