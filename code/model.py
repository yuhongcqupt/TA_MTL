from asyncio import SafeChildWatcher
import math
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

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
device='cuda:3'
seed=50
random.seed(seed) 
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def obj_func(theta,rol1=0.1,rol2=0.1,rol3=0.1):
    tim=theta.shape[2]
    R=torch.zeros(tim,tim-1,dtype=torch.float64).to(device)
    for i in range(len(R)):
        for j in range(len(R[i])):
            if i==j:
                R[i][j]=1.
            elif i==j+1:
                R[i][j]=-1.
    pan1=torch.norm(theta,p=1)
    pan2mat=torch.zeros(3,1,dtype=torch.float64).to(device)
    for i in range(len(pan2mat)-1):
        pan2mat[i]=torch.norm(torch.mm(theta[i],R),p='fro')
    pan2=torch.sum(pan2mat,dim=0)
    # pan2=torch.norm(torch.mm(theta,R),p='fro')
    pan3mat=torch.zeros(3,1,dtype=torch.float64).to(device)
    for i in range(len(pan3mat)-1):
        pan3mat[i]=torch.sum(torch.sqrt(torch.sum(torch.square(theta[i]),dim=1)),dim=0)
    pan3=torch.sum(pan3mat,dim=0)
    # pan3=torch.sum(torch.sqrt(torch.sum(torch.square(theta),dim=1)),dim=0)
    return rol1*pan1+rol2*pan2+rol3*pan3


grouplist=[]
ROI=np.array([i for i in range(120)]).reshape(40,3).tolist()
AREA=np.array([i for i in range(120,678)]).reshape(62,9).tolist()
PLASMA=[i for i in range(678,680)]
SCALA=[i for i in range(680,689)]
PSY=[i for i in range(689,720)]
DEMO=[i for i in range(720,733)]
NERU=[i for i in range(733,747)]
BASE=[i for i in range(747,775)]
for i in ROI:
    grouplist.append(i)
for i in AREA:
    grouplist.append(i)
grouplist.append(PLASMA)
grouplist.append(SCALA)
grouplist.append(PSY)
grouplist.append(DEMO)
grouplist.append(NERU)
grouplist.append(BASE)

def GSGL_obj_func(theta,grouplist,rol1=0.1,rol2=0.1,rol3=0.1):
    pan1mat=torch.zeros(3,1,dtype=torch.float64).to(device)
    for i in range(len(pan1mat)):
        n=torch.norm(theta[i],dim=1,keepdim=False).to(device)
        gmat=torch.zeros(len(grouplist),1,dtype=torch.float64).to(device)
        for j in range(len(grouplist)):
            gmat[j]=torch.sqrt(torch.sum(torch.index_select(n,dim=0,index=torch.tensor(grouplist[j]).to(device))))*math.sqrt(len(grouplist[j]))
        pan1mat[i]=torch.sum(gmat,dim=0)
    pan1=torch.sum(pan1mat,dim=0)

    pan2mat=torch.zeros(3,1,dtype=torch.float64).to(device)
    for i in range(len(pan2mat)):
        gmat=torch.zeros(len(grouplist),theta.shape[2],dtype=torch.float64).to(device)
        for j in range(len(grouplist)):
            gmat[j]=math.sqrt(len(grouplist[j]))*torch.norm(torch.index_select(theta[i],dim=0,index=torch.tensor(grouplist[j]).to(device)),dim=0,keepdim=False)
        pan2mat[i]=torch.sum(gmat)
    pan2=torch.sum(pan2mat,dim=0)
    pan3mat=torch.zeros(3,1,dtype=torch.float64).to(device)
    for i in range(len(pan3mat)-1):
        pan3mat[i]=torch.sum(torch.sqrt(torch.sum(torch.square(theta[i]),dim=1)),dim=0)
    pan3=torch.sum(pan3mat,dim=0)
    return rol1*pan1+rol2*pan2+rol3*pan3


grouplist2=[]
ROI=[i for i in range(120)]
AREA=[i for i in range(120,678)]
PLASMA=[i for i in range(678,680)]
SCALA=[i for i in range(680,689)]
PSY=[i for i in range(689,720)]
DEMO=[i for i in range(720,733)]
NERU=[i for i in range(733,747)]
BASE=[i for i in range(747,775)]

grouplist2.append(ROI)
grouplist2.append(AREA)
grouplist2.append(PLASMA)
grouplist2.append(SCALA)
grouplist2.append(PSY)
grouplist2.append(DEMO)
grouplist2.append(NERU)
grouplist2.append(BASE)

def Joint_obj_func(theta,grouplist2,rol1=0.1,rol2=0.1,rol3=0.1):
    theta1=theta.transpose(0,1)
    norm1=[torch.norm(i) for i in theta1]
    pan1=sum(norm1)

    pant2=[]
    for i in grouplist2:
        pant2.append(torch.index_select(theta1,dim=0,index=torch.tensor(i).to(device)))
    norm2=[torch.norm(i) for i in pant2]
    pan2=sum(norm2)

    u,s,v=torch.svd(theta1)
    pan3=torch.sum(s)
    return rol1*pan1+rol2*pan2+rol3*pan3

def Exc_obj_func(theta,rol1=0.1,rol2=0.1):
    theta1=torch.abs(theta.transpose(0,2))
    pan1=torch.sum(theta1,dim=0)
    pan1=torch.sum(torch.pow(torch.sum(pan1,dim=1),2))/3

    pan2=0
    for i in theta:
        omega=torch.cov(i.T)
        o=torch.mm(i.T,i)
        val,vec=torch.eig(o,eigenvectors=True)
        val1=torch.eye(val.shape[0],dtype=torch.float64).to(device)
        for j in range(val.shape[0]):
            val1[j][j]=torch.sqrt(val[j][0])
        sq=torch.mm(torch.mm(vec,val1),torch.inverse(vec))
        omega=sq/torch.trace(sq).type(torch.float64)
        p2=torch.trace(torch.mm(torch.mm(i,torch.inverse(omega)),i.T))
        pan2=pan2+p2

    return rol1*pan1+rol2*pan2

class FLmodel(nn.Module):

    def __init__(self,n,m,t,c):
        super(FLmodel,self).__init__()
        w=torch.randn((c,m,t),dtype=torch.float64,requires_grad=True)
        # w=prama
        self.w = nn.Parameter(w)
        self.n=n
        self.m=m
        self.t=t
        self.c=c
        self.r=t-1

    def create_Smat(self,t):
        self.S=torch.eye(t)
        for i in range(len(self.S)):
            for j in range(len(self.S[i])):
                if j<i:
                    self.S[i][j]=0.1
        return self.S

    def create_Cmat(self,t,r):
        self.C=torch.eye(t).to(device)
        for i in range(len(self.C)):
            if i<=r:
                self.C[i][i]=1-0.1*i
                for j in range(1,i+1):
                    self.C[i][i-j]=0.1
            else:
                self.C[i][i]=1-0.1*r                                                                                                
                for j in range(1,r+1):
                    self.C[i][i-j]=0.1
        return self.C

# 分类
    def forward(self,x):
        CC=torch.tensor([[1,0,0],[0.1,0.9,0],[0.1,0.1,0.8]]).to(device)
        # CC=torch.tensor([[1,0,0],[0.1,0.9,0],[0,0.1,0.9]]).to(device)
        # CC=torch.tensor([[1,0,0],[0.2,0.8,0],[0,0.2,0.8]]).to(device)
        # CC=torch.tensor([[1,0,0],[0.2,0.8,0],[0.1,0.1,0.8]]).to(device)
        # CC=torch.tensor([[1,0,0],[0.1,0.9,0],[0.03,0.07,0.8]]).to(device)
        # CC=torch.tensor([[1,0,0],[0.2,0.8,0],[0.05,0.15,0.8]]).to(device)
        # CC=torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).to(device)  # no TAO
        y=torch.zeros((x.shape[0],self.t,self.c),dtype=torch.float64)
        for i in range(len(y)):
            yy=torch.zeros((self.c,self.t),dtype=torch.float64)
            for j in range(len(self.w)):
                t1=torch.mm(x[i],self.w[j])
                t2=t1*CC
                t3=torch.sum(t2,dim=1)
                yy[j]=t3
            y[i]=torch.nn.functional.softmax(yy.T,dim=1)
        return y

def Loss(y_hat,y):
    return ((y_hat-y)*(y_hat-y)).sum()/(y.shape[0]*y.shape[1])
