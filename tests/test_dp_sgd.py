import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dp_sgd import DELTA, MAX_GRAD_NORM, NOISE_MULTIPLIER, build_model, make_private  # noqa: E402


def fake_mnist(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.rand(n, 1, 28, 28, generator=g), torch.randint(0, 10, (n,), generator=g))


def private_setup(n=256, batch_size=64):
    torch.manual_seed(0)
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(fake_mnist(n), batch_size=batch_size, shuffle=True)
    return make_private(model, optimizer, loader)


def train_steps(model, optimizer, loader, steps):
    criterion = nn.CrossEntropyLoss()
    done = 0
    while done < steps:
        for data, target in loader:
            if len(target) == 0:  # Poisson sampling can yield empty batches
                continue
            optimizer.zero_grad()
            criterion(model(data), target).backward()
            optimizer.step()
            done += 1
            if done == steps:
                return


def test_model_output_shape():
    model = build_model()
    assert model(torch.rand(8, 1, 28, 28)).shape == (8, 10)


def test_model_is_opacus_compatible():
    assert ModuleValidator.validate(build_model(), strict=True) == []


def test_dp_settings_match_the_reported_experiment():
    assert (NOISE_MULTIPLIER, MAX_GRAD_NORM, DELTA) == (1.1, 1.0, 1e-5)
    _engine, _model, optimizer, _loader = private_setup()
    assert optimizer.noise_multiplier == 1.1
    assert optimizer.max_grad_norm == 1.0


def test_privacy_budget_is_finite_and_grows_with_training():
    engine, model, optimizer, loader = private_setup()
    train_steps(model, optimizer, loader, steps=2)
    eps_early = engine.get_epsilon(delta=DELTA)
    train_steps(model, optimizer, loader, steps=6)
    eps_late = engine.get_epsilon(delta=DELTA)
    assert math.isfinite(eps_early) and eps_early > 0
    assert eps_late > eps_early


def test_per_sample_gradients_are_clipped():
    _engine, model, optimizer, loader = private_setup()
    data, target = next(x for x in loader if len(x[1]) > 0)
    nn.CrossEntropyLoss()(model(data), target).backward()
    optimizer.pre_step()  # clip per-sample grads, before noise is added
    per_sample_norms = torch.stack([
        p.grad_sample.reshape(len(target), -1).norm(2, dim=1) ** 2
        for p in model.parameters() if p.grad_sample is not None
    ]).sum(0).sqrt()
    summed = torch.cat([p.summed_grad.reshape(-1) for p in model.parameters() if p.requires_grad])
    assert per_sample_norms.max() > 0
    # Each example contributes at most MAX_GRAD_NORM to the summed (pre-noise) gradient
    assert summed.norm(2) <= MAX_GRAD_NORM * len(target) + 1e-4


def test_notebook_uses_the_module():
    nb = json.loads((ROOT / "dp_sgd_mnist_opacus.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "from dp_sgd import build_model, make_private" in code
    assert "class CNN" not in code
    outputs = "".join("".join(o.get("text", "")) for c in nb["cells"] for o in c.get("outputs", []))
    assert "Test Accuracy: 90.02%" in outputs and "Final ε = 0.30" in outputs
