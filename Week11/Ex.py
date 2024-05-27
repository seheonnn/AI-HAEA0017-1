# ===== 1. 라이브러리 =====
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# ===== 2. model 정의 =====
# 2.1 변수 설정
latent_size = 64 # GAN 의 입력으로 사용되는 latent vector 의 크기
hidden_size = 256 # GAN 의 FCN 의 hidden layer 의 node 의 수
image_size = 28 * 28 # MNIST 영상의 크기
batch_size = 64 # 한꺼번에 처리할 데이터의 크기

# 2.2 generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 3개의 fully connected layer 를 사용 -> ReLU, ReLU, Tanh
        self.fc = nn.Sequential(
            # 1st Fully Connected Layer: latent_size -> hidden_size
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),

            # 2nd FC: hidden_size -> hidden_size
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            # 3rd FC: hidden_size -> image_size
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

# 2.3 discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            # 1st FC: image_size -> hidden_size, ReLu
            nn.Linear(image_size, hidden_size),
            nn.ReLU(),

            # 2nd FC: hidden_size -> hidden_size, ReLu
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            # 3rd FC: hidden_size -> 1 (True or False), Sigmoid
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# ===== 3. data loading =====
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

mnist = MNIST(root='', train=True, transform=transform, download=True)

data_loader = DataLoader(dataset=mnist,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=2,
                         drop_last=True
                         )
# ===== 4. 초기 설정 =====
n_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = Generator().to(device)
D = Discriminator().to(device)

# loss 함수
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr = 0.0002, betas = (0.5, 0.999))
g_optimizer = optim.Adam(G.parameters(), lr = 0.0002, betas = (0.5, 0.999))

# 4.1 결과 출력 함수: image 를 (5x5)로 출력
def show_images(images, num_images = 25, size = (1, 28, 28)):
    print(images.shape)
    image_flat = images.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_flat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

# ===== 5. 훈련 =====
# n_epochs 만큼 수행
for epoch in range(n_epochs):
    # data_loader 에서 batch_size 만큼 data 읽어오기
    for i, (images, _) in enumerate(data_loader):
        # 1. flatten the images
        images = images.reshape(batch_size, -1).to(device)
        # 2. label 설저이 real -> 1, fake -> 0
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # 3. Discriminator 훈련
        # 3.1 real image 에 대한 loss 계산
        output = D(images) # discriminator 에 real image 를 넣어서 결과를 출력
        d_loss_real = criterion(output, real_labels) # output 에 대한 loss 를 계산: loss: ay_hat 와 y 의 차이 -> y_hat: output, y: real_labels

        # 3.2 fake image 에 대한 loss 계산
        z = torch.randn(batch_size, latent_size).to(device) # latent vector 부터 생성
        fake_images = G(z) # fake image 를 생성
        output = D(fake_images.detach()) # discriminator 에 fake image 를 넣어서 결과를 출력
        d_loss_fake = criterion(output, fake_labels) # loss 함수 구하기

        d_loss = d_loss_real + d_loss_fake

        # 3.3 gradient -> gradient descent
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 4. Generator 훈련
        # 4.1 fake image 를 생성해서 loss 계산
        z = torch.randn(batch_size, latent_size).to(device)  # latent vector 부터 생성
        fake_images = G(z)  # fake image 를 생성
        output = D(fake_images)  # discriminator 에 fake image 를 넣어서 결과를 출력
        g_loss = criterion(output, real_labels)

        # 4.2 gradient -> gradient descent
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    show_images(fake_images)
    show_images(images)

# ===== 6. 실행 =====
z = torch.randn(batch_size, latent_size).to(device)
fake_image = G(z)
show_images(fake_image)