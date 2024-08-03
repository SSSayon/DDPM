import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim

from net_classifier_free import ClassifierFreeDDPM
from util import Scheduler

seed = 0
torch.random.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
STEPS = 100000
BATCH_SIZE = 64

def main():
    model = ClassifierFreeDDPM(
        in_channels=1, 
        model_channels=96, 
        out_channels=1, 
        channel_mult=(1, 2, 2), 
        attention_resolutions=[],
        label_num=11
    ).to(device)
    train_dataset = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    loss = nn.MSELoss()

    scheduler = Scheduler(T, seed)

    step = 0
    while step < STEPS:
        for data, labels in train_iter:
            data: torch.Tensor = data.to(device)
            labels: torch.Tensor = labels.to(device)

            # label 10 corresponds to ALL labels (used in inference, to balance between relativity & diversity)
            mask = torch.rand(labels.shape, dtype=torch.float).to(device) < (1 / 11)
            labels[mask] = 10

            t: torch.Tensor = torch.randint(1, scheduler.T + 1, (data.shape[0],)).to(device)
            noise = torch.randn(data.shape).to(device)

            model_predict = model(
                scheduler.alpha_bar[t, None, None, None] * data + scheduler.beta_bar[t, None, None, None] * noise,
                t / scheduler.T,
                labels
            )

            l: torch.Tensor = loss(model_predict, noise)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            step += 1

            if step % 100 == 0:
                print(f"Step {step}, loss {l.detach().item()}")

            if step % 5000 == 0:
                torch.save(model.state_dict(), f"./checkpoints/classifier_free_{step}.ckpt")

    torch.save(model.state_dict(), f"./checkpoints/classifier_free_{step}.ckpt")

if __name__ == "__main__":
    main()
