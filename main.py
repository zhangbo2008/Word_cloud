'''
做聚类算法.

'''
##
# 如果骚年,你用的是小词库,那么就把词库这个txt的第一行改成 词汇数量,200 即可.词汇数量就是文本行数-1


'''
首先是计数
'''
data=[i for i in open('computer.txt',encoding='utf-8').readlines()]
print(data[0])
out=[]
for i in data:
    for j in i.split(';'):
        j=j.strip("\n")
        j=j.strip(' ')
        out.append(j)
print(out[:10])
print(len(out))

from collections import defaultdict

dict1 = defaultdict(int)

for i in out:
    dict1[i]+=1

print(len(dict1))

a=[]
for i in dict1:
    a.append((i,dict1[i]))
a=sorted(a,key=lambda x:x[1])
a=a[::-1]
print(a[:1000])

# 我们只处理最大前1千个词语
a=a[:1000]




import json
from flask import Flask, request
from gensim.models import KeyedVectors
from flask import jsonify
import argparse
import sys
import socket
import time
import logging



model=KeyedVectors.load_word2vec_format('Tencent_AILab_ChineseEmbedding_Min.txt', binary=False)

def isNoneWords(word):
    if word is None or len(word)==0 or word not in model.vocab:
        return True
    else:
        return False

def vec_route(word):

    if isNoneWords(word):
        return None
    else:
        return {'word':word,'vector': model.word_vec(word).tolist()}


def similarity_route():
    word1 = request.args.get("word1")
    word2 = request.args.get("word2")
    if isNoneWords(word1) or isNoneWords(word2):
        return None
    else:

        # 从下面代码看出来,腾讯词向量的度量是Cosine
        return {'word1':word1,'word2':word2,'similarity':float(model.similarity(word1, word2))}

def similarity(a,b):
    word1 = a
    word2 = b
    if isNoneWords(word1) or isNoneWords(word2):
        return None
    else:

        # 从下面代码看出来,腾讯词向量的度量是Cosine
        return float(model.similarity(word1, word2))


# 下面调用这个函数即可vec_route


print("                    ")
print("                    ")
print("                    ")
print("                    ")
print("                    ")
print("                    ")
print("                    ")
print("                    ")
print("                    ")
print("                    ")
print("                    ")
print("                    ")
print("                    ")

# b里面放的是词语,词频,词向量
b=[]
for i in a:
    tmp=vec_route(i[0])
    if tmp:
         b.append([i[0],i[1],tmp['vector']])

# 下面进行降维
from matplotlib.font_manager import _rebuild

_rebuild() #reload一下
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim
import matplotlib as mpl



label=[]
for i in b:
    label.append(i[0])
vectors=[]
for i in b:
    vectors.append(i[2])
cipin=[]
for i in b:
    cipin.append(i[1])





uuuuu=str(b)
with open("chuanshu999.json",mode='w') as f:
    f.write(uuuuu)






tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')

low_dim_embs = tsne.fit_transform(vectors) # 需要显示的词向量，一般比原有词向量文件中的词向量个数少，不然点太多，显示效果不好


for i in range(len(b)):
    b[i][2]=low_dim_embs[i]

uuuuu=str(b)



import json
#chuanshu22288.json  这个数据就是给前端用的!!!!!!!!!!!!!!!!!1
bb=b
for i in range(len(bb)):
    bb[i][2]=list(bb[i][2])


import json


# define A.class
class node:
    def __init__(self, id,label,x,y,size,color):
        self.id = id
        self.label = label
        self.x = x
        self.y = y
        self.size = size
        self.color=color




##








import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics



#选择聚类数K=2  聚类小于=8,因为颜色就写了8个
n_clusters=4
y_pred=KMeans(n_clusters=n_clusters).fit_predict(low_dim_embs)

with open("y_pred.json",mode='w') as f:
    f.write(str(y_pred))



colorlist=['red','black','yellow','greenyellow','blue','brown','coral','cyan']








list1=[]
for i,j in enumerate(bb):
    list1.append(node(i,j[0],float((float(j[2][0]))),float((float(j[2][1]))),float(j[1]**0.5),colorlist[y_pred[i]]).__dict__)




nodes={}



tmp=json.dumps(nodes,ensure_ascii=False)







# 接口:https://github.com/xukuanzhuo/xukuanzhuo.github.io/issues/8
#下面把每一个聚类里面的距离算一下.
#similarity


class edge:
    def __init__(self, sourceID,targetID,size):
        self.sourceID = sourceID
        self.targetID = targetID

        self.size = size

list3=[]
for i in range(n_clusters):
    dexlist=[]
    for j in range(len(y_pred)):
         if y_pred[j]==i:
             dexlist.append(j)
    for i1 in range(len(dexlist)):
        for i2 in range(i1+1,len(dexlist)):
                left=dexlist[i1]  # 得到索引.
                right=dexlist[i2]
                similar=similarity(bb[left][0],bb[right][0])
                list3.append(edge(bb[left][0],bb[right][0],similar).__dict__)

all3={"nodes":list1,"edges":list3}



tmp2=json.dumps(all3,ensure_ascii=False)

with open("chuanshu22288.json",mode='w',encoding='utf-8') as f:

    f.write(tmp2)











def plot_with_labels(low_dim_embs=low_dim_embs, labels=label, filename="output.png"):   # 绘制词向量图
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    print('绘制词向量中......')
    plt.figure(figsize=(10, 10))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y,s=cipin[i],c=colorlist[y_pred[i]])	# 画点，对应low_dim_embs中每个词向量
        plt.annotate(label,	# 显示每个点对应哪个单词
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)



# plot_with_labels()







print("allover!!!!!!!!!!")


##

