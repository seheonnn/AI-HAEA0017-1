# MLP
# ===== 1. import library =====
# 1. numpy
import numpy as np

# 2. pytorch
import torch
import torch.nn as nn
import torch.optim as optim # gradient descent method를 구현
import torchvision.datasets as dset # pytorch 에서 제공하는 dataset -> MNIST
import torchvision.transforms as transforms # data 를 훈련하기 위해서 관리하는 library
from torch.utils.data import DataLoader # data 를 읽어들이는 library
from torch.autograd import Variable # numpy 의 배열을 pytorch 의 변수로 변환 -> 자동 미분하려고

# 3. matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline # 코랩 or 주피터 전용

# ===== 2. 모델 설계 (MLP) =====
# n_input: 28x28 = 784
# 1st hidden: 256
# 2nd hidden: 128
# n_output: 10
model = nn.Sequential (
    nn.Linear(784, 256), # input -> 1st hidden
    nn.Sigmoid(),
    nn.Linear(256, 128), # 1st hidden -> 2nd hidden
    nn.Linear(128, 10), # 2nd hidden -> output
)

# ===== 3. 데이터 로딩 (MNIST) =====
# MINIST dataset = train set + test set
mnist_train = dset.MNIST("", train=True, transform=transforms.ToTensor(),
                         target_transform=None, download=True)
mnist_test = dset.MNIST("", transform=False, transforms=transforms.ToTensor(),
                        target_transform=None, download=True)
# 3.1 data 확인
# download 받은 data 의 크기는 ?
print("mnist_train의 길이: ", len(mnist_train))
print("mnist_test의 길이: ", len(mnist_test))

# 하나의 데이터는 어떤 모양 ?
image, label = mnist_train.__getitem__(0)
print("image 의 크기: ", image.size())
print("label: ", label)

# image 를 그려보기 -> 그리기는 matplotlib, 입력은 numpy ndarray
img = image.numpy() # pytorch 의 Tensor 를 numpy 의 ndarray 로 변환
# plt.title("label: %d", %label) # % 명령어 -> 코랩 or 주피터만 가능
plt.title("label: {}".format(label))
plt.imshow(img[0], camp="gray")
plt.show()

# ===== 4. 초기 설정 =====
# 4.1 hyperparameter 설정
# batch_size <- 한 번에 훈련할 데이터의 양
batch_size = 1024
# learning rate 클수록 학습이 빠르지만 예측이 틀림 / 작을수록 안정적이지만 느림
learning_rate = 0.001
# epoch
num_epoch = 500

# 4.2 dataloader 설정
train_loader = torch.utills.data.DataLoader (mnist_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=2, # data 를 넣는 동안 몇 개씩 자를지
                                             drop_last=True # 자르고 남은 데이터 제거
                                             )
test_loader = torch.utills.data.DataLoader (mnist_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=2,
                                            drop_last=True
                                            )

# 4.3 loss 함수와 optimizer 를 설정
criterion = nn.CrossEntropyLoss ()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# ===== 5. 훈련 =====
# 5.1 Accuracy 측정 함수 정의
# accuracy = 맞는 답의 수 (correct) / 전체 답의 수 (total)
def ComputeAccuracy (dloader, imode):
    # 전체 답의 수, 맞는 답의 수
    total = 0
    correct = 0

    # dloader 를 이용해서 data 를 불러오는 과정을 수행
    for j, [imgs, labels] in enumerate(dloader): #  batch_size 개 만큼의 [img, label]을 읽어 옴
        img = imgs
        print(len(img))
        label = Variable(labels)
        img = Variable(img, requires_drad=False)
        # img data 는 1024 x 1 x 28 x 28
        # model 의 입력은 1024 x 28 x 28 -> 1024 x 784 (28 x 28)
        # print(img.shape)
        img = img.reshape((img.shape[0], img.shape[2], img.shape[3]))
        # print(img.shape)
        img = img.reshape((img.shape[0], img.shape[1] * img.shape[2]))
        # print(img.shape)
        # 출력 생성
        output = imode(img) # output: 10개의 확률, i 번째 확률은 label 이 i 일 확률 -> argmax
        _, output_index = torch.max(output, 1)
        total += label.size(0)
        correct += (output_index == label).sum().float()

    print("Accuracy: {}".format(100*correct / total))

ComputeAccuracy(test_loader, model)

# ===== 6. 실행 (테스트) =====
loss_arr = []
for i in range(num_epoch):
    for j, [imgs, labels] in enumerate(train_loader):
        img = imgs
        label = Variable(labels)

        # 1024 x 1 x 28 -> 1024 x 28 x 28
        img = img.reshape((img.shape[0], img.shape[2], img.shape[3]))
        # 1024 x 28 x 28 -> 1024 x 784
        img = img.reshape((img.shape[0], img.shape[1] * img.shape[2]))
        img = Variable(img, requires_grad=True)

        optimizer.zero_grad() # 미분
        output = model(img) # 모델 실행
        loss = criterion(output, label) # loss 계산
        loss_arr.append(loss)

        loss.backword() # gradient 구하기
        optimizer.step() # gradient descent method

    if(i%5 == 0):
        # print("%d.." %i)
        print("{}".format(i))
        ComputeAccuracy(test_loader, model)
        print(loss)