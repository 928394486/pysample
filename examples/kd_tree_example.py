# -*- coding: utf-8 -*-

from imylu.utils.kd_tree import KDTree
from imylu.utils.load_data import gen_data
from imylu.utils.utils import get_euclidean_distance
import numpy as np
import pandas as pd
from numpy import *
import numpy.linalg as lg
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from collections import Counter
import random
import json
import csv

def loadFont():
    f = open("D:\\sample\\pysample\\test\\imylu-master\\examples\\cars.json", encoding='utf-8')  #//设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    setting = json.load(f)
    return setting




def loadDataSet(fileName, delim=','):
    y=[]
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr=[]
    for i in range(len(stringArr)):
        datArr_one=[]
        for j in range(len(stringArr[i])-2):
            datArr_one.append(float(stringArr[i][j]))
        datArr.append(datArr_one)
        y.append(stringArr[i][len(stringArr[i])-1])
    return datArr,y


def main():
    print("Testing KD Tree...")
    #X,y=loadDataSet('D:\\sample\\pysample\\code\\imylu-master\\examples\\iris.data')
    t = loadFont()
    X = []
    y = []
    for i in range(len(t)):
        if (t[i]['Horsepower'] == None or t[i]['Miles_per_Gallon'] == None or t[i]['Acceleration'] == None):
            continue
        data_one = [t[i]['Horsepower'], t[i]['Miles_per_Gallon'], t[i]['Acceleration']]
        X.append(data_one)
        y.append(t[i]['Origin'])

    data_num=len(X)
    print(data_num)
    input()
    diff_class_num=Counter(y)

    x_number=10
    y_number=10
    z_number=10
    data_array=np.array(X)
    data_df = pd.DataFrame(data_array)
    data_max=data_df.max()
    data_min=data_df.min()
    x_width=(data_max[0]-data_min[0])/(x_number+1)
    y_width=(data_max[1]-data_min[1])/(y_number+1)
    z_width=(data_max[2]-data_min[2])/(z_number+1)
    box={}
    for i in range(data_num):
        x_order=str(int((X[i][0]-data_min[0])/x_width))
        #print(X[i][1])
        y_order=str(int((X[i][1]-data_min[1])/y_width))
        z_order=str(int((X[i][2]-data_min[2])/z_width))
        ke=[x_order,y_order,z_order]
        str1 = ','
        kes=str1.join(ke)
        if(kes not in box.keys()):
            box[kes]=[]
        box[kes].append([X[i][0],X[i][1],X[i][2]])
    total_num=(x_number+1)*(y_number+1)*(z_number+1)


    diff_class_point={}
    for i in range(len(y)):
        if(y[i] not in diff_class_point.keys()):
            diff_class_point[y[i]]=[]
        diff_class_point[y[i]].append(X[i])
    #print(diff_class_point)
    diff_class_density={}
    for i in diff_class_point.keys():
        diff_class_density[i]={}
        for j in range(len(diff_class_point[i])):
            x_order1=int((diff_class_point[i][j][0]-data_min[0])/x_width)
            y_order1=int((diff_class_point[i][j][1]-data_min[1])/y_width)
            z_order1=int((diff_class_point[i][j][2]-data_min[2])/z_width)
            ke = [x_order, y_order, z_order]
            str1 = ','
            kes = str1.join(ke)
            if (kes not in diff_class_density[i].keys()):
                diff_class_density[i][kes] = []
            diff_class_density[i][kes].append([diff_class_point[i][j][0],diff_class_point[i][j][1],diff_class_point[i][j][2]])






    #print(diff_class_density)

    #print(X_min)
    #print(X_df)


    Cate=sort(list(set(y)))
    class_num=len(Cate)

    #print(Cate)
    color = ['r', 'b', 'c', 'g', 'y', 'k', 'm']
    Categ={}

    for i in range(len(Cate)):
        Categ[Cate[i]]=color[i]

    # Build KD Tree
    node=[]
    tree = KDTree(5,data_num,class_num)
    tree.build_tree(X, y, node)
    #leafs=tree.getAllleaves()

    leaves_num=len(node)
    class_node={}
    node_num=[]
    for i in range(len(node)):
        classes=[]
        for j in range(len(node[i])):
            classes.append(node[i][j][1])
        classeses=dict(Counter(classes))
        node_num.append(classeses)

    for i in range(len(node_num)):
        for j in node_num[i].keys():
            if(j not in class_node.keys()):
                class_node[j]=[]
            class_node[j].append(i)

    class_node_num={}
    for i in class_node.keys():
        class_node_num[i]=len(class_node[i])
    #print(class_node)
    node_class={}
    for i in class_node.keys():
        for j in class_node[i]:
            if(j not in node_class.keys()):
                node_class[j]=[]
            node_class[j].append(i)
    #print(node_class)
    num_node={}
    for i in node_class.keys():
        if(len(node_class[i]) not in num_node.keys()):
            num_node[len(node_class[i])]=[]
        num_node[len(node_class[i])].append(i)
    #print(num_node)
    #print(node_class)
    node_label=[]
    for i in range(class_num):
        k=i+1
    sample=[]

    for i in range(len(node)):
        le=len(node[i])
        if(le==1):
            sample.append(node[i][0])
            continue
        ran=random.randint(1,le-1)
        sample.append(node[i][ran])

    out=open('Stu.csv','a',newline='')
    csvw=csv.writer(out,dialect='excel')
    key_={'USA':1,'Europe':2,'Japan':3}
    print(len(sample))
    input()
    for i in range(len(sample)):
        csvw.writerow([sample[i][0][0],sample[i][0][1],sample[i][0][2],key_[sample[i][1]]])
    print('write over')
    #sample
    #print(len(sample))







if __name__ == "__main__":
    main()
