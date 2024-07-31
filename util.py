import torch

class Scheduler():
    def __init__(self, _T) -> None:
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

    @property
    def beta(self):
        return self._beta

    @property
    def alpha_bar(self):
        return self._alpha_bar

    @property
    def beta_bar(self):
        return self._beta_bar
