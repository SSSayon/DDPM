import torch
from tqdm import tqdm
import numpy as np

from net import DDPM, Classifier
from util import Scheduler, draw

seed = 0
torch.random.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
NUM_OUTPUT = 16
IS_GRADUALLY = False
STEPS = 20
GAMMA = 10
DESC = f"DDPM_with_classifier(gamma={GAMMA})"

@torch.no_grad()
def main(ckpt, ckpt2):
    model = DDPM(in_channels=1, n_feat=128).to(device)
    model.load_state_dict(torch.load(f"./checkpoints/{ckpt}.ckpt", map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    classifier = Classifier(in_channels=1, n_feat=128).to(device)
    classifier.load_state_dict(torch.load(f"./checkpoints/classifier_{ckpt2}.ckpt", map_location=torch.device('cpu'), weights_only=True))
    classifier.eval()

    scheduler = Scheduler(T, seed)

    x: torch.Tensor = torch.randn((NUM_OUTPUT, 1, 28, 28)).to(device)
    labels = torch.randint(0, 10, (NUM_OUTPUT,))

    outputs = x.clone().detach().to("cpu")

    assert scheduler.T % STEPS == 0
    tau = np.linspace(scheduler.T, 0, STEPS + 1).astype(int)    # 1000 ... 0

    for idx, t in tqdm(enumerate(tau[:-1]), total=STEPS):

        t_prev = tau[idx + 1]

        x = scheduler.forward(x, model(x, t / scheduler.T), t_prev, t).to(device)

        with torch.enable_grad():
            grad = classifier.gradient(x.clone().detach().requires_grad_(), t / scheduler.T, labels).to(device)

        x += grad * scheduler.sigma_t(t_prev, t) ** 2 * GAMMA

        if (idx + 1) % (STEPS // 10) == 0:
            if IS_GRADUALLY:
                outputs = torch.concat((outputs, x.clone().detach().to("cpu")), dim=0)

    if not IS_GRADUALLY:  
        outputs = x.clone().detach().to("cpu")

    name = f"{DESC}_{STEPS}_gradually_{ckpt}.png" if IS_GRADUALLY else f"{DESC}_{STEPS}_{ckpt}.png"
    draw(outputs, name, width=NUM_OUTPUT if IS_GRADUALLY else None, labels=labels.numpy())


if __name__ == "__main__":
    main(100366, 500000)
