import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from net import Classifier
from util import Scheduler

seed = 0
torch.random.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
STEPS = 1000000
BATCH_SIZE = 64

def main():
    classifier = Classifier(in_channels=1, n_feat=128, n_class=10).to(device)
    train_dataset = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(classifier.parameters(), lr=5e-4)
    loss = nn.CrossEntropyLoss()

    scheduler = Scheduler(T, seed)

    step = 0
    while step < STEPS:
        for data, labels in train_iter:
            data: torch.Tensor = data.to(device)
            labels: torch.Tensor = labels.to(device)

            t: torch.Tensor = torch.randint(1, scheduler.T + 1, (data.shape[0],)).to(device)
            noise = torch.randn_like(data).to(device)

            model_predict = classifier(
                scheduler.alpha_bar[t, None, None, None] * data + scheduler.beta_bar[t, None, None, None] * noise,
                t / scheduler.T
            )

            l: torch.Tensor = loss(model_predict, labels)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            step += 1

            if step % 1000 == 0:
                print(f"Step {step}, loss {l.detach().item()}")

            if step % 100000 == 0:
                torch.save(classifier.state_dict(), f"./checkpoints/classifier_{step}.ckpt")

    torch.save(classifier.state_dict(), f"./checkpoints/classifier_{step}.ckpt")

if __name__ == "__main__":
    main()
