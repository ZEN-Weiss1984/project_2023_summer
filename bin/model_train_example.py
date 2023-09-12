import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datetime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl

s_input_size=4
s_output_size=4
s_hidden_size=40
s_num_layers=2
s_lr=0.01

num_time_steps=10

class Net(nn.Module):
    def __init__(self,_input_size,_output_size,_hidden_size):
        super(Net,self).__init__()
        self.rnn=nn.RNN(
            input_size=_input_size,
            output_size=_output_size,
            hidden_size=_hidden_size,
            batch_first=True
        )
        for p in self.rnn.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)
        
        self.linear = nn.Linear(_hidden_size, _output_size)
    

    def forward(self, x, hidden_prev):
       out, hidden_prev = self.rnn(x, hidden_prev)
       out = out.view(-1, s_hidden_size)
       out = self.linear(out)
       out = out.unsqueeze(dim=0)
       return out, hidden_prev

def getdata():
    x1 = np.linspace(1,10,30).reshape(30,1)
    y1 = (np.zeros_like(x1)+2)+np.random.rand(30,1)*0.1
    z1 = (np.zeros_like(x1)+2).reshape(30,1)
    tr1 =  np.concatenate((x1,y1,z1),axis=1)
    return tr1


def tarin_RNN(data):
    model = Net(s_input_size, s_hidden_size, s_num_layers)
    print('model:\n',model)
    criterion = nn.MSELoss()    #损失函数
    optimizer = optim.Adam(model.parameters(), s_lr)    #优化器
    hidden_prev = torch.zeros(1, 1, s_hidden_size)  #初始化上一步的隐藏状态
    l = []  #存储损失值

    #开始训练
    for iter in range(3000):
        start = np.random.randint(10, size=1)[0]    #获得一个随机值
        end = start + num_time_steps    #结束值增加一个步长
        x = torch.tensor(data[start:end]).float().view(1, num_time_steps - 1, 3)
        y = torch.tensor(data[start + 5:end + 5]).float().view(1, num_time_steps - 1, 3)
        output, hidden_prev = model(x, hidden_prev) #调用向前传播方法
        hidden_prev = hidden_prev.detach()                 

        loss = criterion(output, y)
        model.zero_grad()   #梯度清零
        loss.backward()     #反向传播计算梯度
        optimizer.step()    #利用优化器更新参数

        if iter % 100 == 0:
            print("Iteration: {} loss {}".format(iter, loss.item()))
            l.append(loss.item())
        #每一百次记录一次损失

    plt.plot(l,'r')
    plt.xlabel('训练次数')
    plt.ylabel('loss')
    plt.title('RNN损失函数下降曲线')
    return hidden_prev,model


def RNN_pre(model,data_test,hidden_prev):
    hidden_prev=torch.zeros(1,1,s_hidden_size)






def main():
    data = getdata()
    start = datetime.datetime.now()
    hidden_pre, model = tarin_RNN(data)
    end = datetime.datetime.now()
    print('The training time: %s' % str(end - start))
    plt.show()
    RNN_pre(model, data, hidden_pre)



if __name__ == '__main__':
    main()