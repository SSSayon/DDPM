import torch
from tqdm import tqdm

from net_classifier_free import ClassifierFreeDDPM
from util import Scheduler, draw

seed = 0
torch.random.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
NUM_OUTPUT = 16
IS_GRADUALLY = False
STEPS = 20
OMEGA = 4
DESC = f"DDPM_classifier_free(omega={OMEGA})"

@torch.no_grad()
def main(ckpt):
    model = ClassifierFreeDDPM(
        in_channels=1, 
        model_channels=96, 
        out_channels=1, 
        channel_mult=(1, 2, 2), 
        attention_resolutions=[],
        label_num=11
    ).to(device)
    model.load_state_dict(torch.load(f"./checkpoints/classifier_free_{ckpt}.ckpt", map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    scheduler = Scheduler(T, seed)

    x: torch.Tensor = torch.randn((NUM_OUTPUT, 1, 28, 28)).to(device)
    labels = torch.randint(0, 10, (NUM_OUTPUT,))

    outputs = x.clone().detach().to("cpu")

    assert scheduler.T % STEPS == 0
    tau = torch.linspace(scheduler.T, 0, STEPS + 1).to(int).to(device)    # 1000 ... 0

    for idx, t in tqdm(enumerate(tau[:-1]), total=STEPS):
        t_prev = tau[idx + 1]

        x = scheduler.forward(    # see https://spaces.ac.cn/archives/9257
            x, 
            model(x, t.unsqueeze(0) / scheduler.T, labels) * (1 + OMEGA) - model(x, t.unsqueeze(0) / scheduler.T, torch.ones_like(labels) * 10) * OMEGA, 
            t_prev, t
        ).to(device)

        if (idx + 1) % (STEPS // 10) == 0:
            if IS_GRADUALLY:
                outputs = torch.concat((outputs, x.clone().detach().to("cpu")), dim=0)

    if not IS_GRADUALLY:  
        outputs = x.clone().detach().to("cpu")

    name = f"{DESC}_{STEPS}_gradually_{ckpt}.png" if IS_GRADUALLY else f"{DESC}_{STEPS}_{ckpt}.png"
    draw(outputs, name, width=NUM_OUTPUT if IS_GRADUALLY else None, labels=labels.numpy())


if __name__ == "__main__":
    main(15000)
