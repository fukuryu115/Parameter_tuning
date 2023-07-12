import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import optuna
optuna.logging.disable_default_handler()
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np


BATCHSIZE = 128

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=2)

test_set = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCHSIZE, shuffle=False, num_workers=2)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))


#入力画像の高さと幅，畳み込み層のカーネルサイズ
in_height = 28
in_width = 28
kernel = 3

#ネットワーク構造の定義
class Net(nn.Module):
    def __init__(self, trial, num_layer, mid_uints, num_filters):
        super(Net,self).__init__()
        self.activation = get_activation(trial)

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
        
def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 1 - correct / len(test_loader.dataset)


import torch.optim as optim

def get_optimizer(trial, model):
    optimizer_names = ['Adam', 'MomentumSGD', 'rmsprop']
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    if optimizer_name == optimizer_names[0]: 
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    elif optimizer_name == optimizer_names[1]:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters())
    return optimizer

def get_activation(trial):
    activation_names = ['ReLU', 'ELU']
    activation_name = trial.suggest_categorical('activation', activation_names)
    if activation_name == activation_names[0]:
        activation = F.relu
    else:
        activation = F.elu
    return activation

EPOCH = 1
def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #畳み込み層の数
    num_layer = trial.suggest_int('num_layer', 3, 7)

    #FC層のユニット数
    mid_units = int(trial.suggest_discrete_uniform("mid_units", 100, 500, 100))

    #各畳込み層のフィルタ数
    num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 16, 128, 16)) for i in range(num_layer)]

    model = Net(trial, num_layer, mid_units, num_filters).to(device)
    optimizer = get_optimizer(trial, model)

    for step in range(EPOCH):
        print("step")
        train(model, device, train_loader, optimizer)
        error_rate = test(model, device, test_loader)

    return error_rate

if __name__ == "__main__":
    TRIAL_SIZE = 1
    print("tuning start")
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)
    print(study.best_params)
