#coding:utf-8


import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import load_model
from public.public_function import *

from _3_combine.zuhe_w_single_ea import calculateFitness
from public.keras_load_data import *


def read_get_w(path):
    with open(path, 'r') as f:
        zuhe_test=f.read()
        zuhe=zuhe_test[1:-1]
        zuhe_w=zuhe.split(',')
        for i in range(len(zuhe_w)):
            zuhe_w[i]=float(zuhe_w[i])
        return zuhe_w


def read_model(path,trainORtest):
    l_y=[]
    l_acc=[]
    paths=[]
    for i in range(20):
        paths.append(os.path.join(path,str(i)))
    for i in range(len(paths)):
        model_path=os.path.join(paths[i],getFilename(paths[i]))
        model=load_model(model_path)
        if trainORtest.strip() in ["A", 'a']:
            loss, accuracy = model.evaluate(train_x_all, train_y_all)
            last_layer = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
            y_prediction = last_layer.predict(train_x_all)
        elif trainORtest.strip() in ["b", 'B']:
            loss, accuracy = model.evaluate(validation_x_all, validation_y_all)
            last_layer = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
            y_prediction = last_layer.predict(validation_x_all)
        elif trainORtest.strip() in ["c", 'C']:
            loss, accuracy = model.evaluate(test_x_all, test_y_all)
            last_layer = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
            y_prediction = last_layer.predict(test_x_all)

        print('\nloss: ', loss)
        print('\naccuracy: ', accuracy)
        l_acc.append(accuracy)
        l_y.append(y_prediction)
    return l_y,l_acc


def view_single_semeble_acc(x1,y1):

    plt.plot(x1, y1, label='all', linewidth=2, color='red', marker='o',
             markerfacecolor='blue', markersize=5)

    plt.xlabel('number')
    plt.ylabel('acc')
    plt.title('single_and_resemeble_acc')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    while True:
        select=input("s开始,q退出:")
        if select=='s':
            path="E:\pc_model\small\dai_"+date_str+"\\"+date_str+"_cluster\\"

            zuhe_w = read_get_w(os.path.join(path, "final_generation", 'final_zuhe_w.txt'))

            while True:
                train_OR_test=input("选择集成模型在训练集还是测试集合上的结果？（训练集：A，验证集：B,测试集：C）(q/Q退出):")
                if train_OR_test.strip() in ['A','a']:
                    str1="train"
                    l_y, l_acc = read_model(path,train_OR_test)
                    out="训练"
                elif train_OR_test.strip() in ['c','C']:
                    str1="test"
                    l_y, l_acc = read_model(path,train_OR_test)
                    out="测试"
                elif train_OR_test.strip() in ['B','b']:
                    str1="validation"
                    l_y,l_acc=read_model(path,train_OR_test)
                    out="验证"

                elif train_OR_test in ['q','Q']:
                    break
                else:
                    print("请输入正确指令")
                    continue
                final_train = calculateFitness(zuhe_w, l_y, str1)
                final_acc = (1 - final_train)

                print("组合权重后"+out+"集得到的准确率为:", final_acc)
                l_acc.append(final_acc)
                view_single_semeble_acc(range(0,len(l_acc)),l_acc)


                view_acc=[]
                for i in range(len(l_acc)):
                    if l_acc[i]>0.99:
                        view_acc.append(l_acc[i])
                view_single_semeble_acc(range(0,len(view_acc)),view_acc)
        elif select=='q':
            break
        else:
            continue
