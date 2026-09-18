import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader


x=torch.rand(10000,2)
y=(x[:,0]**2+x[:,1]**2+abs(x[:,0])<2).long()
dataset=TensorDataset(x,y)
loader=DataLoader(dataset,batch_size=64,shuffle=True)
model=nn.Sequential(nn.Linear(2,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,2))
losser=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)


for epoch in range(21):
    for xb,yb in loader:
        optimizer.zero_grad()
        logits=model(xb)
        loss=losser(logits,yb)
        loss.backward()
        optimizer.step()



with torch.no_grad():
    pred=model(x).argmax(dim=1)
    acc=(pred==y).float().mean()
    print(acc)



