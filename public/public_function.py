
#coding:utf-8
import os
import shutil

date_str = "2018_6_31"

temppath = 'E:\pc_model\small\\bian'
cluster_path = "E:\pc_model\small\dai_"+date_str+"\\"+date_str+"_cluster\\"

# 将sourcepath路径下的文件转移到targetpath路径下
def movefiles(sourcepath,targetpath):
    for file in os.listdir(sourcepath):
        files=os.path.join(sourcepath,file)
        shutil.copy(files,targetpath)

def print_l(l):
    print("==========================================")
    print("第一卷积核尺寸："+str(l[0]))
    print("第一卷积核数量："+str(l[1]))
    print("第一池化层尺寸:"+str(l[2]))
    print("第一池化层步长:"+str(l[3]))

    print("第二卷积核尺寸：" + str(l[4]))
    print("第二卷积核数量：" + str(l[5]))
    print("第二池化层尺寸:" + str(l[6]))
    print("第二池化层步长:" + str(l[7]))

    print("全连接层节点数:"+str(l[8]))
    print("学习率:"+str(l[9]))
    print("dropout:" + str(l[10]))
    print("==========================================")


def control(a,b,c,upOrdown=1):
    ma=max(b,c)
    mi=min(b,c)
    while True:
        # 上边界
        if ma == mi:
            a = ma
            break
        else:
            if upOrdown == 1:
                a = a - (ma-mi)
            else:
                a = a + (ma - mi)
            if a <= ma and a >= mi:
                break
    return a

def Move_and_Alter_Model_Save_Path(net_i, path):
    # 将第一代个体对应的模型，移动到对应的个体文件夹中
    movefiles(net_i.model_save_path, os.path.join(path,str(net_i.net_num)))
    # 将原保存模型的文件夹里面的模型文件删除
    deletefile(net_i.model_save_path)
    # 将个体的模型保存路径，修改为转移后的路径
    net_i.model_save_path = os.path.join(path,str(net_i.net_num))


# 将targetpath路径下的所有文件删除
def deletefile(targetpath):
    for file in os.listdir(targetpath):
        targetfile=os.path.join(targetpath,file)
        if os.path.isfile(targetfile):
            os.remove(targetfile)

def getFilename(path):
    list_name=os.listdir(path)
    return list_name[0]

def folder_create_or_clear(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('文件夹创建成功')
    else:
        file_list = os.listdir(path)
        for f in file_list:
            f_path = os.path.join(path, f)
            if os.path.isfile(f_path):
                os.remove(f_path)
            elif os.path.isdir(f_path):
                shutil.rmtree(f_path)
        print('文件夹清空成功')

def p_path_print_after_cluster(p):
    for i in range(len(p)):
        print("个体"+str(i)+"model_save_path:"+p[i][0].model_save_path)


def p_path_print(p):
    for i in range(len(p)):
        print("个体"+str(i)+"model_save_path:"+p[i].model_save_path)

def bianyigeti_print(y,t,i):
    print("第" + str(t) + "代的第" + str(i) + "个变异个体的model_save_path:"+y.model_save_path)


def single_newgeti_print(y,t,i,j):
    print("第" + str(t) + "代的第" + str(i) + "个用于替换第"+str(j)+"邻域的变异个体的model_save_path:" + y.model_save_path)
    print("第" + str(t) + "代的第" + str(i) + "个用于替换第"+str(j)+"邻域的变异个体的ep_model_save_path:" + y.ep_model_save_path)


def before_newgeti(newgeti, before=True):
    if before == True:
        s = "转移之前"
    else:
        s = "转移之后"
    for i in range(len(newgeti)):
        print(s+"个体" + str(i) + "model_save_path:" + newgeti[i].model_save_path)
        print(s+"个体" + str(i) + "ep_model_save_path:" + newgeti[i].ep_model_save_path)

def lamb_print(lamb):
    for i in range(len(lamb)):
        print("第"+str(i)+"个个体对应的权重向量"+str(lamb[i]))

def functions_print(p):
    for i in range(len(p)):
        print("第"+str(i)+"个个体的第一个目标函数值为："+str(p[i].f[0])+"，第二个目标函数值为："+str(p[i].f[1]))

def update_second(p):
    for i in range(len(p)):
        with open(os.path.join(p[i].model_save_path, "second.txt"), 'a')as f:
            f.write(str(p[i].f[1]) + ",")