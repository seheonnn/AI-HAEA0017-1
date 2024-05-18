# ===== 1. import library =====
# torch
import torch
import torch.nn as nn
import torchvision.datasets as dset # MNIST dataset 사용
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
# matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline # 코랩 or 주피터 전용

# ===== 2. hyperparameter 설정 =====
# batch size, learning rate, num_epoch
batch_size = 16
learning_rate = 0.002
num_epoch = 200

# ===== 3. model 정의 =====
class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        # 1 단계 : CNN layer
        self.cnn_layer = nn.Sequential(
            # conv + relu -> 1 특징을, 16 특징으로 뽑고, 5 (5x5 filter 사용), padding = 2 유지
            nn.Conv2d(1, 16, 5, padding=2), # 28x28x1 input -> 28x28x16 output
            nn.ReLU(),

            # conv + relu -> 16, 32, 5 (5x5 filter), padding = 2
            nn.Conv2d(16, 32, 5, padding=2), # 28x28x16 input -> 28x28x32 output
            nn.ReLU(),

            # pool : 28x28 -> 14x14
            nn.MaxPool2d(2, 2),

            # conv + relu -> 32, 64, 5, padding = 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),

            # pool : 14x14 -> 7x7
            nn.MaxPool2d(2, 2)

        ) # cnn.layer의 출력 :
        # 2 단계 : FC layer (fully-connected)
        self.fc_layer = nn.Sequential(
            nn.Linear(64*7*7, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x): # CNN 은  data 를 하나씩 처리하는 것이 아닌 batch_size 단위로 처리함
        out = self.cnn_layer(x) # out : batch_size x 7x7x64 -> 4d tensor
        out = out.view(batch_size, -1) # 1차원 형태로 펼침. out : batch_size x 7*7*64 -> 2d tensor
        out = self.fc_layer(out) # fc_layer의 input : 7*7*64x1 1d tensor
        return out

# ===== 4. data loading =====
mnist_train = dset.MNIST("", train=True, transform=transforms.ToTensor(),
                         target_transform=None, download=True)
mnist_test = dset.MNIST("", transform=False, transforms=transforms.ToTensor(),
                        target_transform=None, download=True)

# ===== 5. data loader 설정 =====
# batch_size*100 만큼 슬라이싱, batch_size 넘겨 받음, 섞음, preprocessor 2개 사용, 슬라이싱 후 남은 우수리 버림
train_loader = torch.utils.data.DataLoader(list(mnist_train)[:batch_size*100], batch_size=batch_size,
                                           shuffle=True, num_workers=2, drop_last=True)
test_loader = torch.utils.data.DataLoader(list(mnist_test), batch_size=batch_size,
                                          shuffle=False, num_workers=2, drop_last=True)

# ===== 6. optimizer 설정 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = myCNN().to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ===== 7. accuracy 측정 함수 =====
def EstimateAccuracy (dloader, imodel):
    correct = 0
    total = 0

    for image, label in dloader:
        x = Variable(image, volatitle=True).to(device)
        y = Variable(label).to(device)

        y_hat = imodel.forward(x)
        _, y_hat_index = torch.max(y_hat, 1)

        total += label.size(0)
        correct += (y_hat_index == y).sum().float()

    print("Accuracy: {}".format(100*correct/total()))
    return 100*correct/total

# ===== 8. 훈련 =====
loss_arr = []
accu_arr = []

for i in range(num_epoch):
    for image, label in train_loader:
        x = Variable(image).to(device)
        y = Variable(label).to(device)

        optimizer.zero_grad()
        y_hat = model.forward(x)
        loss = loss_func(y_hat, y)
        loss.backward()
        optimizer.step()

    if i%0 == 0:
        print(i, loss)
        accu = EstimateAccuracy(test_loader, model)
        loss_arr.append(loss)
        accu_arr.append(accu)

# ===== 9. 결과 출력 =====
