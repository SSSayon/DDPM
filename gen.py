import torch
from tqdm import tqdm
import numpy as np

from net import DDPM
from util import Scheduler, draw

seed = 0
torch.random.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
NUM_OUTPUT = 10
IS_GRADUALLY = True
DESC = "DDPM"
STEPS = 1000

@torch.no_grad()
def main(ckpt):
    model = DDPM(in_channels=1, n_feat=128).to(device)
    model.load_state_dict(torch.load(f"./checkpoints/{ckpt}.ckpt", map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    scheduler = Scheduler(T, seed)

    # x: torch.Tensor = torch.randn((NUM_OUTPUT, 1, 28, 28)).to(device)

    x1, x2 = torch.randn((1, 28, 28)), torch.randn((1, 28, 28))
    x: torch.Tensor = scheduler.interpolate(x1, x2, NUM_OUTPUT).to(device)

    outputs = x.clone().detach().to("cpu")

    assert scheduler.T % STEPS == 0
    tau = np.linspace(scheduler.T, 0, STEPS + 1).astype(int)    # 1000 ... 0

    for idx, t in tqdm(enumerate(tau[:-1]), total=STEPS):

        t_prev = tau[idx + 1]

        # z = torch.randn(x.shape).to(device)
        # x = (x - scheduler.beta[t] ** 2 / scheduler.beta_bar[t] * model(x, torch.Tensor([t / scheduler.T]))) / scheduler.alpha[t] + scheduler.beta[t] * z

        x = scheduler.forward(x, model(x, torch.Tensor([t / scheduler.T])), t_prev, t).to(device)

        if (idx + 1) % (STEPS // 10) == 0:
            if IS_GRADUALLY:
                outputs = torch.concat((outputs, x.clone().detach().to("cpu")), dim=0)

    if not IS_GRADUALLY:  
        outputs = x.clone().detach().to("cpu")

    name = f"{DESC}_{STEPS}_gradually_{ckpt}.png" if IS_GRADUALLY else f"{DESC}_{STEPS}_{ckpt}.png"
    draw(outputs, name, width=NUM_OUTPUT if IS_GRADUALLY else None)


if __name__ == "__main__":
    main(100366)
