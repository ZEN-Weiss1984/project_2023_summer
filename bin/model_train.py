import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json

import os

BATCH_SIZE = 5
DEVICE = torch.device("cuda")
EPOCHS =40

INPUT_SIZE=9
OUTPUT_SIZE=4
HIDDEN_SIZE=50
NUM_LAYERS=2
SEQ_LEN=64
LR=0.001

train_set=[]
train_target=[]
test_set=[]
test_target=[]

def get_data():
    global train_set,train_target,test_set,test_target
    fs=open(r"C:/Users/k/Desktop/body_pre/data_detected/train_set.json","r")
    content=fs.read()
    train_set=json.loads(content)
    fs.close()
    fs=open(r"C:/Users/k/Desktop/body_pre/data_detected/train_target.json","r")
    content=fs.read()
    train_target=json.loads(content)
    fs.close()
    fs=open(r"C:/Users/k/Desktop/body_pre/data_detected/test_set.json","r")
    content=fs.read()
    test_set=json.loads(content)
    fs.close()
    fs=open(r"C:/Users/k/Desktop/body_pre/data_detected/test_target.json","r")
    content=fs.read()
    test_target=json.loads(content)
    fs.close()

class Net(nn.Module):
    def __init__(self,_input_size,_output_size,_hidden_size):
        super(Net,self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.rnn=nn.LSTM(
            input_size=_input_size,
            hidden_size=_hidden_size,
            num_layers=NUM_LAYERS,
            bidirectional=False
        )
        for p in self.rnn.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)
        
        self.linear = nn.Linear(_hidden_size, _output_size)
    

    def forward(self, x, hidden_prev):
        #torch.autograd.set_detect_anomaly(True)
        out, hidden_prev = self.rnn(x, hidden_prev)
        out = self.linear(out)
        #out = nn.functional.relu(out)
        return out,hidden_prev
    

def train_model(train_data,train_target):
    model = Net(INPUT_SIZE,OUTPUT_SIZE,HIDDEN_SIZE)
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR) 
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200, eta_min=1e-6)
    print('model:\n',model)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1,1.0,1.0,1.0]).to(torch.float32).cuda() ,size_average=True)
    optimizer = optim.Adam(model.parameters(), LR)

    train_data=torch.tensor(train_data).cuda()
    train_target=torch.tensor(train_target).cuda()

    #train_data=train_data.cuda()
    #train_target=train_target.cuda()

    for num_epochs in range(EPOCHS):
        total_loss=0.0
        
        for iter in range(0,len(train_data),BATCH_SIZE):
            batch_input=train_data[iter:iter+BATCH_SIZE]
            batch_input=batch_input.permute(1,0,2).cuda()
            batch_target=train_target[iter:iter+BATCH_SIZE]
            batch_target=batch_target.permute(1,0,2).cuda()
            #print(batch_input.size())
            #print(batch_target.size())

            hidden_prev = torch.zeros(NUM_LAYERS,BATCH_SIZE,HIDDEN_SIZE).cuda()
            c_hint = torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE).cuda()

            for iter in range(SEQ_LEN):
                output,(hidden_prev,c_hint)=model(batch_input[iter].unsqueeze(dim=0),(hidden_prev,c_hint))

                c_hint=c_hint.detach()
                hidden_prev = hidden_prev.detach()

                batch_target=batch_target.to(torch.float32)
                loss=criterion(output[0],batch_target[iter])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss+=loss.item()

            scheduler.step()

        print(f"Epoch [{num_epochs}], Loss: {total_loss:.4f}")
        scheduler.step()
    print("Training finished")
    return model


def model_pre(model,pre_data,hidden_prev,c_hint):
    model.eval()
    pre_data=torch.tensor(pre_data).unsqueeze(dim=0).unsqueeze(dim=0)
    out,(hidden_prev,c_hint)=model(pre_data,(hidden_prev,c_hint))
    index=out[0][0].argmax().item()
    accuracy=out[0][0][index].item()
    return index,accuracy,hidden_prev,c_hint


def test_model(model,test_data,test_target):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    total_act = 0
    detected_act=0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5,1.0,1.0,1.0]).to(torch.float32).cuda() ,size_average=True)
    test_data=torch.tensor(test_data).permute(1,0,2).cuda()
    test_target=torch.tensor(test_target).permute(1,0,2).to(torch.float32).cuda()
    with torch.no_grad():
        hidden_prev=torch.zeros(NUM_LAYERS,1,HIDDEN_SIZE).cuda()
        c_hint = torch.zeros(NUM_LAYERS,1, HIDDEN_SIZE).cuda()
        for i in range(300):
            pre_target,accuracy=model(test_data[i].unsqueeze(dim=0),(hidden_prev,c_hint))
            test_loss+=criterion(pre_target[0],test_target[i]).item()
            if test_target[i].argmax()!=0:
                total_act+=1
            if pre_target[0].argmax()!=0:
                detected_act+=1
            if pre_target[0].argmax()==test_target[i].argmax() and test_target[i].argmax()!=0:
                correct+=1
    test_loss /= len(test_target)
    print("Test --- Averager loss : {:.4f}, Accuracy : {:.3f}\n".format(
        test_loss,100.0*correct/total_act
    ))
    print(f"act : {total_act}")
    print(f"detected act : {detected_act}")

def save_model(model):
    que=input("save the model?[y/n]")
    if que=='y':
        name=input("model name:")
        model_path="C:/Users/k/Desktop/body_pre/models/"+name+".pth"
        torch.save(model,model_path)
        print("model saved")

if __name__=="__main__":
    print("start")
    get_data()
    model=train_model(train_set,train_target)    
    test_model(model,test_set,test_target)
    save_model(model)