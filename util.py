import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from math import sqrt, sin, cos, pi


def draw(outputs: torch.Tensor, name, width=None, labels=None):

    if width is not None:
        height = outputs.shape[0] // width
    else:
        height = int(sqrt(outputs.shape[0]))
        width  = outputs.shape[0] // height
    assert height * width == outputs.shape[0]

    figure, axes = plt.subplots(height, width, figsize=(28, 28))
    for i in range(height):
        for j in range(width):
            ax: Axes = axes[i, j]
            ax.imshow(outputs[i * width + j][0])
            ax.axis("off")
            if labels is not None: 
                ax.set_title(labels[i * width + j], fontdict={"fontsize": 15}, pad=-10)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
    plt.savefig(f"./output/{name}")


class Scheduler():
    def __init__(self, _T, _seed=0) -> None:
        torch.random.manual_seed(_seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._T = _T

        _beta_square = torch.arange(_T + 1) * 0.02 / _T    # index 0 is just a placeholder
        self._beta = torch.sqrt(_beta_square).to(device)
        self._alpha = torch.sqrt(1 - _beta_square).to(device)

        self._alpha_bar = torch.ones_like(self._alpha).to(device)
        self._alpha_bar[1:] = torch.cumprod(self._alpha[1:], dim=0)    # alpha_bar[t] = alpha[1] * ... * alpha[t]
        self._beta_bar = torch.sqrt(1 - self._alpha_bar).to(device)

    @property
    def T(self):
        return self._T

    @property
    def alpha(self):            
        return self._alpha
    
    def _alpha_t(self, t_prev, t):    # alpha[t] = alpha_bar[t] / alpha_bar[t_prev]
        return self._alpha_bar[t] / self._alpha_bar[t_prev]

    @property
    def beta(self):             
        return self._beta
    
    def _beta_t(self, t_prev, t):     # beta[t] = sqrt(1 - alpha_bar[t] ** 2 / alpha_bar[t_prev] ** 2)
        return torch.sqrt(1 - self._alpha_t(t_prev, t) ** 2)

    @property
    def alpha_bar(self):              # can still use in tau
        return self._alpha_bar

    @property
    def beta_bar(self):               # can still use in tau
        return self._beta_bar

    # ------------ used in diffusion & forward process ------------

    def _sigma_t(self, t_prev, t):

        # choice 1
        return self._beta_bar[t_prev] / self._beta_bar[t] * self._beta_t(t_prev, t)

        # choice 2
        # return self._beta_t(t_prev, t)

        # choice 3: sigma = 0, DDIM
        return 0

    def forward(self, x, eps, t_prev, t):    # see https://spaces.ac.cn/archives/9181
        z = torch.randn_like(x)
        return (x - (self._beta_bar[t] - self._alpha_t(t_prev, t) * 
                     torch.sqrt(torch.clamp(self._beta_bar[t_prev] ** 2 - self._sigma_t(t_prev, t) ** 2, torch.tensor(0.0)))) * eps    # floating point error?
               ) / self._alpha_t(t_prev, t) + self._sigma_t(t_prev, t) * z

    def interpolate(self, x1, x2, n):
        lambdas = torch.linspace(0, 1, n)
        
        interpolated = []
        for lam in lambdas:
            interpolated.append(x1 * cos(lam * pi / 2) + x2 * sin(lam * pi / 2))
        
        return torch.stack(interpolated, dim=0)
