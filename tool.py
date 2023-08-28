from asyncio import SafeChildWatcher
from posixpath import dirname
from pprint import pp
from pstats import Stats
from re import T
from time import time_ns
import numpy as np
import torch
import random
import os 
import torch.utils.data as Data  
from IPython import display
from matplotlib import  pyplot as plt
import matplotlib_inline
from torch.autograd import Variable
from torch.nn import init
import torch.nn as nn  
import torch.optim as optim
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
device='cuda:3'
seed=100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def setload(path):
    with open(path, encoding='utf-8') as file_obj:
        lines=file_obj.readlines()
    data=[]
    for i in range(0,len(lines)):
        item=lines[i].split(",")
        item[len(item)-1]=item[len(item)-1][:-1]
        data.append(item)
    title=data[0]
    del data[0]
    return data,title

def labelencode(label,encode):
    l=[i[2] for i in label]
    ll=[encode[i] if i in encode else i for i in l]
    return ll

def featureNormalize_std(X,type='f'):
    if type=='f':
        X_norm = X.reshape(X.shape[0]*X.shape[1],X.shape[2])
    elif type=='b':
        X_norm=X
    mu=torch.mean(X_norm,dim=0)
    sigma=torch.std(X_norm,dim=0)
    x=X
    for i in range(len(x)):
        x[i]  = (X[i] - mu) / (sigma+1e-8)
    return x,mu,sigma

def unonehot(data):
    d=data.view(data.shape[0]*data.shape[1],data.shape[2]).cpu().numpy()
    dd=[]
    for i in d:
        dd.append(np.where(i==1)[0][0])
    return dd

def vexclass(p):
    pt=p
    for i in range(len(p)):
        mask=(p[i]==p[i].max(dim=1,keepdim=True)[0]).to(p.device)
        pt[i]=mask
    return pt

def decode(p):
    p=vexclass(p)
    pt=p.flatten(start_dim=0,end_dim=1)
    pp=pt.cpu().numpy().tolist()
    c=[]
    for i in pp:
        c.append(i.index(1.))
    return c

def acc(p,r):
    num=0
    for i in range(len(p)):
        if r[i]==p[i]:
            num=num+1
    return num/len(p)

def tpfn(pred,real,c):
    p=[]
    r=[]
    for i in pred:
        if i!=c:
            p.append(-1)
        else:
            p.append(c)
    for i in real:
        if i!=c:
            r.append(-1)
        else:
            r.append(c)
    tp,tn,fp,fn=0,0,0,0
    for i in range(len(p)):
        if p[i]==r[i] and p[i]==c:
            tp=tp+1
        if p[i]==r[i] and r[i]==-1:
            tn=tn+1
        if p[i]!=r[i] and p[i]==c:
            fp=fp+1
        if p[i]!=r[i] and p[i]==-1:
            fn=fn+1
    if tp==0:
        pre=0
        rec=0
        sen=0
        f1=0
    else:
        pre=tp/(tp+fp)
        rec=tp/(tp+fn)
        sen=tp/(tp+fn)
        f1=(pre*rec)/(pre+rec)
    if tn==0:
        spe=0
    else:
        spe=tn/(tn+fp)

    return pre,rec,spe,sen,f1

def data_10scl(data,label,n,zh):
    idx=[i for i in range(len(data))]
    random.shuffle(idx)
    cdata=[ [] for i in range(zh)]
    for i in range(len(idx)):
        cdata[i%zh].append(idx[i])
    trainid=torch.tensor(list(set(idx)-set(cdata[n]))).to(data.device)
    train_data=torch.index_select(data,dim=0,index=trainid)
    train_label=torch.index_select(label,dim=0,index=trainid)
    testid=torch.tensor(cdata[n]).to(data.device)
    test_data=torch.index_select(data,dim=0,index=testid)
    test_label=torch.index_select(label,dim=0,index=testid)
    return train_data,train_label,test_data,test_label

def data_scaler(data,label,id,gate=0.7):
    a=int(len(data)*gate)
    trainid=torch.tensor(id[0:a]).to(data.device)
    testid=torch.tensor(list(set(id)-set(id[0:a]))).to(data.device)
    train_data=torch.index_select(data,dim=0,index=trainid)
    train_label=torch.index_select(label,dim=0,index=trainid)
    test_data=torch.index_select(data,dim=0,index=testid)
    test_label=torch.index_select(label,dim=0,index=testid)
    return train_data,train_label,test_data,test_label

def w_row(filename,data):
    with open (filename,'w') as f:
        w=csv.writer(f)
        w.writerow(data)
        # for i in data:
        #     w=csv.writer(f)
        #     w.write(i)
    f.close()



