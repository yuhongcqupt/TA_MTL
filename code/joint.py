from asyncio import SafeChildWatcher
from imp import find_module
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
import tool as tl
from model import FLmodel,Loss,obj_func,Joint_obj_func
import svm

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
device='cuda:3'
seed=100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

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
lpath="label.csv"
fdpath="data.csv"

num_data=114
fl_tim=3
num_fl_fea=703
num_class=3
num_fea=775

#############################
# 调参：
#     1. 特征选择方法（皮尔森、最大互信息、距离相关稀疏）
#     2. 归一化方法（最大最小、标准差）
#     3. 优化参数（学习率、衰减率、动量）
#     4. 正则化参数
#     5. 时间累积矩阵
###############################

data,featitle=tl.setload(fdpath)
del featitle[0]
for i in data:
    del i[0]

Dat=np.array(data,np.float64).reshape(num_data,fl_tim,num_fea)
data=list(Dat)
Df=torch.tensor(Dat,dtype=torch.float64).to(device)

clas,clastitle=tl.setload(lpath)
clas_encode={'CN':0,'MCI':1,'AD':2}
Clas=tl.labelencode(clas,clas_encode)
clas_one_hot=torch.nn.functional.one_hot(torch.tensor(Clas),num_classes=num_class)
Cf=clas_one_hot.view(num_data,fl_tim,num_class).to(device)

Df_norm,Df_mu,Df_std=tl.featureNormalize_std(Df)

ACC1=[]
CN_PRE1=[]
CN_REC1=[]
CN_SPE1=[]
CN_SEN1=[]
CN_F11=[]
MCI_PRE1=[]
MCI_REC1=[]
MCI_SPE1=[]
MCI_SEN1=[]
MCI_F11=[]
AD_PRE1=[]
AD_REC1=[]
AD_SPE1=[]
AD_SEN1=[]
AD_F11=[]
ACC2=[]
CN_PRE2=[]
CN_REC2=[]
CN_SPE2=[]
CN_SEN2=[]
CN_F12=[]
MCI_PRE2=[]
MCI_REC2=[]
MCI_SPE2=[]
MCI_SEN2=[]
MCI_F12=[]
AD_PRE2=[]
AD_REC2=[]
AD_SPE2=[]
AD_SEN2=[]
AD_F12=[]
ACC3=[]
CN_PRE3=[]
CN_REC3=[]
CN_SPE3=[]
CN_SEN3=[]
CN_F13=[]
MCI_PRE3=[]
MCI_REC3=[]
MCI_SPE3=[]
MCI_SEN3=[]
MCI_F13=[]
AD_PRE3=[]
AD_REC3=[]
AD_SPE3=[]
AD_SEN3=[]
AD_F13=[]
P=[]
R=[]

for i in range(10):
    train_Df,train_Cf,test_Df,test_Cf=tl.data_10scl(Df_norm,Cf,i,10)

    flmodel=FLmodel(len(train_Df),num_fea,fl_tim,num_class).to(device)
    loss1=Loss
    optimizer1=optim.SGD(flmodel.parameters(),lr=0.01,weight_decay=0.1,momentum=0.1,nesterov=True)

    iterations = 5000
    for epoch in range(iterations):
        F_output=flmodel(train_Df).to(device)
        optimizer1.zero_grad()
        for param in flmodel.parameters():
            l=loss1(F_output,train_Cf) 
            pant=Joint_obj_func(param,grouplist2,rol1=0.01,rol2=0.01,rol3=0.01)
            ll=l+pant
            w=param
        ll.backward()
        optimizer1.step()
        if epoch % 1000 == 0:
            print('epoch %d | loss:%f' %(epoch,l.item()))

    test_Df1=test_Df[:,0:1,:]
    test_Df2=test_Df[:,1:2,:]
    test_Df3=test_Df[:,2:3,:]
    pred_F1=flmodel(test_Df1).data.to(device)
    pred_F2=flmodel(test_Df2).data[:,1:3,:].to(device)
    pred_F3=flmodel(test_Df3).data[:,2:3,:].to(device)
    p1=tl.decode(pred_F1)
    p2=tl.decode(pred_F2)
    p3=tl.decode(pred_F3)
    r1=tl.decode(test_Cf)
    r2=tl.decode(test_Cf[:,1:3,:])
    r3=tl.decode(test_Cf[:,2:3,:])

    

    test1_pre0,test1_rec0,test1_spe0,test1_sen0,test1_f10=tl.tpfn(p1,r1,0)
    test1_pre1,test1_rec1,test1_spe1,test1_sen1,test1_f11=tl.tpfn(p1,r1,1)
    test1_pre2,test1_rec2,test1_spe2,test1_sen2,test1_f12=tl.tpfn(p1,r1,2)
    ACC1.append(tl.acc(p1,r1))
    CN_PRE1.append(test1_pre0),CN_REC1.append(test1_rec0),CN_SPE1.append(test1_spe0),CN_SEN1.append(test1_sen0),CN_F11.append(test1_f10)
    MCI_PRE1.append(test1_pre1),MCI_REC1.append(test1_rec1),MCI_SPE1.append(test1_spe1),MCI_SEN1.append(test1_sen1),MCI_F11.append(test1_f11)
    AD_PRE1.append(test1_pre2),AD_REC1.append(test1_rec2),AD_SPE1.append(test1_spe2),AD_SEN1.append(test1_sen2),AD_F11.append(test1_f12)

    test2_pre0,test2_rec0,test2_spe0,test2_sen0,test2_f10=tl.tpfn(p2,r2,0)
    test2_pre1,test2_rec1,test2_spe1,test2_sen1,test2_f11=tl.tpfn(p2,r2,1)
    test2_pre2,test2_rec2,test2_spe2,test2_sen2,test2_f12=tl.tpfn(p2,r2,2)
    ACC2.append(tl.acc(p2,r2))
    CN_PRE2.append(test2_pre0),CN_REC2.append(test2_rec0),CN_SPE2.append(test2_spe0),CN_SEN2.append(test2_sen0),CN_F12.append(test2_f10)
    MCI_PRE2.append(test2_pre1),MCI_REC2.append(test2_rec1),MCI_SPE2.append(test2_spe1),MCI_SEN2.append(test2_sen1),MCI_F12.append(test2_f11)
    AD_PRE2.append(test2_pre2),AD_REC2.append(test2_rec2),AD_SPE2.append(test2_spe2),AD_SEN2.append(test2_sen2),AD_F12.append(test2_f12)

    test3_pre0,test3_rec0,test3_spe0,test3_sen0,test3_f10=tl.tpfn(p3,r3,0)
    test3_pre1,test3_rec1,test3_spe1,test3_sen1,test3_f11=tl.tpfn(p3,r3,1)
    test3_pre2,test3_rec2,test3_spe2,test3_sen2,test3_f12=tl.tpfn(p3,r3,2)
    ACC3.append(tl.acc(p3,r3))
    CN_PRE3.append(test3_pre0),CN_REC3.append(test3_rec0),CN_SPE3.append(test3_spe0),CN_SEN3.append(test3_sen0),CN_F13.append(test3_f10)
    MCI_PRE3.append(test3_pre1),MCI_REC3.append(test3_rec1),MCI_SPE3.append(test3_spe1),MCI_SEN3.append(test3_sen1),MCI_F13.append(test3_f11)
    AD_PRE3.append(test3_pre2),AD_REC3.append(test3_rec2),AD_SPE3.append(test3_spe2),AD_SEN3.append(test3_sen2),AD_F13.append(test3_f12)

print("time1")
print("acc:",ACC1)
print("CN_PRE:",CN_PRE1)
print("CN_REC:",CN_REC1)
print("CN_SPE:",CN_SPE1)
print("CN_SEN:",CN_SEN1)
print("CN_F1:",CN_F11)
print("MCI_PRE:",MCI_PRE1)
print("MCI_REC:",MCI_REC1)
print("MCI_SPE:",MCI_SPE1)
print("MCI_SEN:",MCI_SEN1)
print("MCI_F1:",MCI_F11)
print("AD_PRE:",AD_PRE1)
print("AD_REC:",AD_REC1)
print("AD_SPE:",AD_SPE1)
print("AD_SEN:",AD_SEN1)
print("AD_F1:",AD_F11)
print("finall acc:",np.mean(ACC1))
print("CN finall pre,rec,spe,sen and f1:",np.mean(CN_PRE1),np.mean(CN_REC1),np.mean(CN_SPE1),np.mean(CN_SEN1),np.mean(CN_F11))
print("MCI finall pre,rec,spe,sen and f1:",np.mean(MCI_PRE1),np.mean(MCI_REC1),np.mean(MCI_SPE1),np.mean(MCI_SEN1),np.mean(MCI_F11))
print("AD pre,rec,spe,sen and f1:",np.mean(AD_PRE1),np.mean(AD_REC1),np.mean(AD_SPE1),np.mean(AD_SEN1),np.mean(AD_F11))
print("====================================================================================================")

print("time2")
print("acc:",ACC2)
print("CN_PRE:",CN_PRE2)
print("CN_REC:",CN_REC2)
print("CN_SPE:",CN_SPE2)
print("CN_SEN:",CN_SEN2)
print("CN_F1:",CN_F12)
print("MCI_PRE:",MCI_PRE2)
print("MCI_REC:",MCI_REC2)
print("MCI_SPE:",MCI_SPE2)
print("MCI_SEN:",MCI_SEN2)
print("MCI_F1:",MCI_F12)
print("AD_PRE:",AD_PRE2)
print("AD_REC:",AD_REC2)
print("AD_SPE:",AD_SPE2)
print("AD_SEN:",AD_SEN2)
print("AD_F1:",AD_F12)
print("finall acc:",np.mean(ACC2))
print("CN finall pre,rec,spe,sen and f1:",np.mean(CN_PRE2),np.mean(CN_REC2),np.mean(CN_SPE2),np.mean(CN_SEN2),np.mean(CN_F12))
print("MCI finall pre,rec,spe,sen and f1:",np.mean(MCI_PRE2),np.mean(MCI_REC2),np.mean(MCI_SPE2),np.mean(MCI_SEN2),np.mean(MCI_F12))
print("AD pre,rec,spe,sen and f1:",np.mean(AD_PRE2),np.mean(AD_REC2),np.mean(AD_SPE2),np.mean(AD_SEN2),np.mean(AD_F12))
print("====================================================================================================")

print("time3")
print("acc:",ACC3)
print("CN_PRE:",CN_PRE3)
print("CN_REC:",CN_REC3)
print("CN_SPE:",CN_SPE3)
print("CN_SEN:",CN_SEN3)
print("CN_F1:",CN_F13)
print("MCI_PRE:",MCI_PRE3)
print("MCI_REC:",MCI_REC3)
print("MCI_SPE:",MCI_SPE3)
print("MCI_SEN:",MCI_SEN3)
print("MCI_F1:",MCI_F13)
print("AD_PRE:",AD_PRE3)
print("AD_REC:",AD_REC3)
print("AD_SPE:",AD_SPE3)
print("AD_SEN:",AD_SEN3)
print("AD_F1:",AD_F13)
print("finall acc:",np.mean(ACC3))
print("CN finall pre,rec,spe,sen and f1:",np.mean(CN_PRE3),np.mean(CN_REC3),np.mean(CN_SPE3),np.mean(CN_SEN3),np.mean(CN_F13))
print("MCI finall pre,rec,spe,sen and f1:",np.mean(MCI_PRE3),np.mean(MCI_REC3),np.mean(MCI_SPE3),np.mean(MCI_SEN3),np.mean(MCI_F13))
print("AD pre,rec,spe,sen and f1:",np.mean(AD_PRE3),np.mean(AD_REC3),np.mean(AD_SPE3),np.mean(AD_SEN3),np.mean(AD_F13))
print("====================================================================================================")
