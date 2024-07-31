import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from net import DDPM
from util import Scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def main(ckpt):
    model = DDPM(in_channels=1, n_feat=128).to(device)
    model.load_state_dict(torch.load(f"./checkpoints/{ckpt}.ckpt"))
    model.eval()

    x: torch.Tensor = torch.randn((10, 1, 28, 28)).to(device)


    scheduler = Scheduler(1000)

    outputs = []
    for t in tqdm(range(scheduler.T, 0, -1)):    # 1000 ... 1
        z = torch.randn(x.shape).to(device)
        x = (x - scheduler.beta[t] ** 2 / scheduler.beta_bar[t] * model(x, torch.Tensor([t / scheduler.T]))) / scheduler.alpha[t] + scheduler.beta[t] * z

        if t % 100 == 0:
            outputs.append(x.clone().detach().to("cpu"))

    outputs.append(x.clone().detach().to("cpu"))

    figure, axes = plt.subplots(11, 10, figsize=(28, 28))
    for i, output in enumerate(outputs):
        for j in range(10):
            axes[i][j].imshow(output[j][0])
            axes[i][j].axis("off")

    # output = x.clone().detach().to("cpu")
    # figure, axes = plt.subplots(10, 10, figsize=(28, 28))
    # for k in range(100):
    #     i = k // 10
    #     j = k % 10
    #     axes[i][j].imshow(output[k][0])
    #     axes[i][j].axis("off")

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

    plt.show()
    plt.savefig(f"./output/10_gradually_{ckpt}.png")


if __name__ == "__main__":
    main(100366)
