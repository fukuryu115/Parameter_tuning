import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import string
import argparse  
import datetime


parser = argparse.ArgumentParser(description='ハイパーパラメータ')    
# 3. parser.add_argumentで受け取る引数を追加していく
parser.add_argument('--epoch', default=10,type=int, help='epoch')
parser.add_argument('--activation', default="ReLU", help='ReLU or ELU',)  
parser.add_argument('--optimizer', default="Adam", help='Adam or MomentumSGD')  
parser.add_argument('--weight_decay', default=1.3452825924268737e-07,type=float, help='1e-10, 1e-3')
parser.add_argument('--adam_lr', default=0.0003348252618961708,type=float, help='1e-5, 1e-1')
parser.add_argument('--momentum_sgd_lr', default=1e-3,type=float, help='1e-5, 1e-1')
parser.add_argument('--num_layer', default=5,type=int, help='3 to 7')
parser.add_argument('--mid_units', default=500,type=int, help='100 to 500')
tp = lambda x:list(map(int, x.split('.')))
parser.add_argument('--num_filter',default=[112,48,80,96,112], type=tp, help='16 to 128 list')
parser.add_argument('--filename', default='',type=str, help='save name of logs.')
args = parser.parse_args()


path=os.path.dirname(os.path.abspath(__file__))
if args.filename == '':
    save_name = 'no_name'
else:
    save_name = args.filename

if not os.path.exists(path+f"/../../logs/MNIST/{save_name}"):
    os.makedirs(path+f"/../../logs/MNIST/{save_name}")
writer = SummaryWriter(log_dir=path+f"/../../logs/MNIST/{save_name}")
BATCHSIZE = 128
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = MNIST(root=path+"/../../dataset/MNIST", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=2)
test_set = MNIST(root=path+"/../../dataset/MNIST", train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCHSIZE, shuffle=False, num_workers=2)
classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))


#入力画像の高さと幅，畳み込み層のカーネルサイズ
in_height = 28
in_width = 28
kernel = 3
#学習結果の保存
history = {
    "train_loss": [],
    "validation_loss": [],
    "validation_acc": []
}

iex=0
jex=0
#ネットワーク構造の定義
class Net(nn.Module):
    def __init__(self, num_layer, mid_uints, num_filters):
        super(Net,self).__init__()
        self.activation = get_activation()

        #第1層
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters[0], kernel_size=3)])
        self.out_height = in_height - kernel +1
        self.out_width = in_width - kernel +1
        #第2層以降
        for i in range(1, num_layer):
          self.convs.append(nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=3))
          self.out_height = self.out_height - kernel + 1
          self.out_width = self.out_width - kernel +1
        #pooling層
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_height = int(self.out_height / 2)
        self.out_width = int(self.out_width / 2)
        #線形層
        self.out_feature = self.out_height * self.out_width * num_filters[num_layer - 1]
        self.fc1 = nn.Linear(in_features=self.out_feature, out_features=mid_uints) 
        self.fc2 = nn.Linear(in_features=mid_uints, out_features=10)


    def forward(self, x):
        for i, l in enumerate(self.convs):
            x = l(x)
            x = self.activation(x)
        x = self.pool(x)
        x = x.view(-1, self.out_feature)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
def train(model, step,device, train_loader, optimizer):
    model.train()
    #print("\nTrain start")
    i=0
    train_loss=0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        i+=1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i % 100 == 99:
            pass
            #print("Training: {} epoch. {} iteration. Loss: {}".format(step+1,i+1,loss.item()))
    train_loss /= len(train_loader)
    #print("Training loss (ave.): {}".format(train_loss))
    history["train_loss"].append(train_loss)
    global jex
    writer.add_scalar("train_loss", train_loss, jex)
    jex+=1

def test(model, device, test_loader):
    #print("\nValidation start")
    model.eval()
    correct = 0.0
    val_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output,target).item()
            val_loss += F.nll_loss(output,target,reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(test_loader.dataset)
        #correct /= len(test_loader)
        correct /= len(test_loader.dataset)

        #print("Validation loss: {}, Accuracy: {}\n".format(val_loss,correct))
        history["validation_loss"].append(val_loss)
        history["validation_acc"].append(correct)
        global iex
        writer.add_scalar("validation_loss", val_loss, iex)
        writer.add_scalar("validation_acc", correct, iex)
        iex+=1
    return 1 - correct / len(test_loader.dataset)



def get_optimizer(model):
    optimizer_name = args.optimizer
    weight_decay = args.weight_decay
    if optimizer_name == "Adam": 
        adam_lr = args.adam_lr
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    elif optimizer_name == "MomentumSGD":
        momentum_sgd_lr = args.momentum_sgd_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.RMSprop(model.parameters())
    return optimizer

def get_activation():
    activation_name = args.activation
    if activation_name == "ReLU":
        activation = F.relu
    else:
        activation = F.elu
    return activation



def objective(epoch):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #畳み込み層の数
    num_layer = args.num_layer

    #FC層のユニット数
    mid_units = args.mid_units

    #各畳込み層のフィルタ数
    num_filters = args.num_filter

    model = Net(num_layer, mid_units, num_filters).to(device)
    optimizer = get_optimizer(model)

    for step in range(epoch):
        #print("step")
        i = step + 1
        bar = '*' * i + " " * (epoch - i)
        print(f"\r\033[K[{bar}] {i/epoch*100:.02f}% ({i}/{epoch})", end="")
        train(model, step,device, train_loader, optimizer)
        error_rate = test(model, device, test_loader)
        
        if not os.path.exists(path+"/../../saved_model/MNIST"):
            os.makedirs(path+"/../../saved_model/MNIST")
        torch.save(model.state_dict(), path+"/../../saved_model/MNIST/MNIST.pth")

    print('\nleaning finished')

    
    return error_rate

if __name__ == "__main__":
    epoch = args.epoch
    objective(epoch)
    #print(history)
    
    if not os.path.exists(path+"/img/"+save_name):
        os.makedirs(path+"/img/"+save_name)

    ntime = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
    print('\n')
    print(ntime)
    fig,ax = plt.subplots()
    
    ax.plot(range(1, epoch+1), history["train_loss"], label="train_loss")
    ax.plot(range(1, epoch+1), history["validation_loss"], label="validation_loss")
    
    ax.text(0.5,0.5,'train_loss = {}'.format(np.average(history["train_loss"][-5:-1])),transform=ax.transAxes)
    ax.text(0.5,0.4,'validation_loss = {}'.format(np.average(history["validation_loss"][-5:-1])),transform=ax.transAxes)
    ax.set_xlabel("epoch")
    ax.legend()
    plt.savefig(path+"/img/"+save_name+"/loss.png")

    fig, ax = plt.subplots()
    ax.plot(range(1, epoch+1), history["validation_acc"])
    ax.set_title("test accuracy")
    ax.set_xlabel("epoch")
    ax.text(0.5,0.5,'test_acc = {}'.format(np.average(history["validation_acc"][-5:-1])),transform=ax.transAxes)
    writer.close()

    print('train_loss = {}'.format(np.average(history["train_loss"][-5:-1])))
    print('validation_loss = {}'.format(np.average(history["validation_loss"][-5:-1])))
    print('test_acc = {}'.format(np.average(history["validation_acc"][-5:-1])))
    plt.savefig(path+"/img/"+save_name+"/test_acc.png")
    
    
    