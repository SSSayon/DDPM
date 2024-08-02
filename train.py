import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim

from net import DDPM
from util import Scheduler

seed = 0
torch.random.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
STEPS = 100000
BATCH_SIZE = 64

def main():
    model = DDPM(in_channels=1, n_feat=128).to(device)
    train_dataset = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.MSELoss()

    scheduler = Scheduler(T, seed)

    step = 0
    while step < STEPS:
        for data, label in train_iter:
            data: torch.Tensor = data.to(device)

            t: torch.Tensor = torch.randint(1, scheduler.T + 1, (data.shape[0],)).to(device)
            noise = torch.randn(data.shape).to(device)

            model_predict = model(
                scheduler.alpha_bar[t, None, None, None] * data + scheduler.beta_bar[t, None, None, None] * noise,
                t / scheduler.T
            )

            l: torch.Tensor = loss(model_predict, noise)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            step += 1
            print(f"Step {step}, loss {l.detach().item()}")

            if step % 1000 == 0:
                torch.save(model.state_dict(), f"./checkpoints/{step}.ckpt")

    torch.save(model.state_dict(), f"./checkpoints/{step}.ckpt")

if __name__ == "__main__":
    main()
