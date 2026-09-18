import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

#MLP模型定义
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(3,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,2)
        self.projection1=nn.Linear(3,128)
        self.relu=nn.ReLU()
    def forward(self,x):
        h1=self.fc1(x)
        h1=self.relu(h1)
        h1=self.fc2(h1)
        h1=self.relu(h1)
        h2=self.projection1(x)
        h3=h1+h2
        h3=self.fc3(h3)
        return h3

#初始化-定义loss和操作器
model=MLP()
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
best_accurate=0

#封装
def generate_data(amount,type):#这里代表固定还是变化的种子，填0则为随机种子
    if type==0:
        seeds=torch.randint(0,100,()).item()
        torch.manual_seed(seeds)
        xb=torch.rand(amount,3)
        xb=-2+xb*4
        yb=(abs(xb[:,0])+abs(xb[:,1])+abs(xb[:,2])>3).long()#这里定义学习目标
        dataset=TensorDataset(xb,yb)
        print("this time the random seed is",seeds)
        return dataset
    else:
        seeds=type
        torch.manual_seed(seeds)
        xb=torch.rand(amount,3)
        xb=-2+xb*4
        yb=(abs(xb[:,0])+abs(xb[:,1])+abs(xb[:,2])>3).long()#这里定义学习目标
        dataset=TensorDataset(xb,yb)
        print("this time the selected seed is",seeds)
        return dataset

def model_train(amount,epoch):
    dataset=generate_data(amount,0)
    loader=DataLoader(dataset,batch_size=64,shuffle=True)
    for i in range(epoch):
        correct,total=0,0
        for xb,yb in loader:
            optimizer.zero_grad()
            logits=model(xb)
            loss=loss_fn(logits,yb)
            loss.backward()
            optimizer.step()
            total+=yb.size(0)
            correct+=(yb==logits.argmax(dim=1)).sum().item()
        model_validation(model)
        print("the training epoch ",i,"accuaracy is ",correct/total)
    print("the train is end")
    return

def model_validation(model):
    correct,total=0,0
    with torch.no_grad():
        dataset=generate_data(100,918)
        loader=DataLoader(dataset,batch_size=64)
        for xb,yb in loader:
            pred=model(xb).argmax(dim=1)
            total+=yb.size(0)
            correct+=(yb==pred).sum().item()
        if correct/total>best_accurate:
            best_accurate=correct/total
            print("in validation,the accurate new record is ",correct/total)

def model_test(model):
    correct,total=0,0
    with torch.no_grad():
        dataset=generate_data(1000,919)
        loader=DataLoader(dataset,batch_size=64)
        for xb,yb in loader:
            pred=model(xb).argmax(dim=1)
            total+=yb.size(0)
            correct+=(yb==pred).sum().item()
        print("in test,the accurate is ",correct/total)
    return

#------对比基线--------#
def random_baseline(dim_number):
    dataset=generate_data(1000,0)
    total,correct=0,0
    loader=DataLoader(dataset,batch_size=64)
    for xb,yb in loader:
        pred=torch.randint(0,dim_number,(yb.size(0),))
        total+=yb.size(0)
        correct+=(yb==pred).sum().item()
    print("the random predicting baseline accuracy is",correct/total)

def majority_baseline(dim_number):
    dataset=generate_data(1000,0)
    total,correct=0,0
    y_all=dataset.tensors[1]
    major=torch.bincount(y_all).argmax().item()
    pred=torch.full_like(y_all,major)
    testing_dataset=generate_data(1000,0)
    y_test_label=testing_dataset.tensors[1]
    acc=(pred==y_test_label).float().mean().item()
    print("the majority predicting baseline training accuracy is",acc)



#主程序         
majority_baseline(2)
random_baseline(2)
model_train(2000,20)
model_test(model)



    


