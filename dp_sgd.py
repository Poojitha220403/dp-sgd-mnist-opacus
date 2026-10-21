"""Model and DP-SGD setup used by dp_sgd_mnist_opacus.ipynb."""
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

NOISE_MULTIPLIER = 1.1
MAX_GRAD_NORM = 1.0
DELTA = 1e-5


# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def build_model():
    """Create the CNN and make it Opacus-compatible."""
    model = CNN()
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)
    return model


def make_private(model, optimizer, train_loader,
                 noise_multiplier=NOISE_MULTIPLIER, max_grad_norm=MAX_GRAD_NORM):
    """Wrap model, optimizer and loader for DP-SGD; returns (engine, model, optimizer, loader)."""
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return privacy_engine, model, optimizer, train_loader
