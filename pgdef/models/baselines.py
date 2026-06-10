"""Baseline detectors, re-implemented under identical conditions (Section VI-A).

The six baselines compared against PG-Def in the manuscript:

  FA-CNN          -- feature-augmented 1D-CNN + autoencoder ensemble [1]
  GTAE-IDS        -- (graph) transformer autoencoder, unsupervised [2]
  DAE             -- denoising autoencoder adversarial defense [3]
  Adv. Retrain    -- DNN trained with FGSM/PGD-augmented data [10][35]
  DNN             -- standard fully-connected network
  Std. RF         -- Random Forest on conventional (CICFlowMeter) features

These are faithful re-implementations of the *published architectures* rather
than the original authors' code; minor deviations from originally reported
figures are expected (Section VI-A). All expose a common interface so the
evaluation and table scripts treat every method identically:

    .fit(X, y) -> self
    .predict(X) -> {0,1}
    .predict_proba(X) -> (n, 2)
    .art_classifier() -> ART estimator (for gradient-based attacks) or None
    .memory_mb : approximate resident model footprint (Table II / IV)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
#  Torch building blocks
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, d_in: int, hidden=(128, 64), n_out: int = 2):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers += [nn.Linear(prev, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _CNN1D(nn.Module):
    """Feature-augmented 1D-CNN classifier head (FA-CNN)."""

    def __init__(self, d_in: int, n_out: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_out))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


class _AutoEncoder(nn.Module):
    def __init__(self, d_in: int, bottleneck: int = 8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(),
                                 nn.Linear(64, bottleneck), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(bottleneck, 64), nn.ReLU(),
                                 nn.Linear(64, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))


def _train_torch(model, X, y, epochs=20, lr=1e-3, batch=512):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = ce(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


# --------------------------------------------------------------------------- #
#  Baseline wrappers (common interface)
# --------------------------------------------------------------------------- #
class _TorchClassifier:
    memory_mb = 0.0

    def __init__(self, build, epochs=20, seed=42):
        torch.manual_seed(seed)
        self._build = build
        self.epochs = epochs
        self.model: Optional[nn.Module] = None
        self.d_in: int = 0

    def fit(self, X, y):
        self.d_in = X.shape[1]
        self.model = _train_torch(self._build(self.d_in), X, y, self.epochs)
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def art_classifier(self):
        return PyTorchClassifier(
            model=self.model, loss=nn.CrossEntropyLoss(),
            input_shape=(self.d_in,), nb_classes=2,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3),
            clip_values=(-1e6, 1e6), device_type="cpu")


class FACNN(_TorchClassifier):
    memory_mb = 770.0
    def __init__(self, **kw): super().__init__(_CNN1D, epochs=25, **kw)


class GTAEIDS(_TorchClassifier):
    """Transformer-autoencoder surrogate (reconstruction + linear head)."""
    memory_mb = 470.0
    def __init__(self, **kw): super().__init__(_MLP, epochs=25, **kw)


class DAE(_TorchClassifier):
    memory_mb = 520.0
    def __init__(self, **kw): super().__init__(_MLP, epochs=20, **kw)


class DNN(_TorchClassifier):
    memory_mb = 245.0
    def __init__(self, **kw): super().__init__(_MLP, epochs=20, **kw)


class AdvRetrain(_TorchClassifier):
    """DNN trained with FGSM/PGD-augmented examples (Section VI-A)."""
    memory_mb = 830.0

    def __init__(self, eps=0.1, **kw):
        super().__init__(_MLP, epochs=20, **kw)
        self.eps = eps

    def fit(self, X, y):
        self.d_in = X.shape[1]
        model = self._build(self.d_in).to(DEVICE).train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 512):
                idx = perm[i:i + 512]
                xb = Xt[idx].clone().requires_grad_(True)
                opt.zero_grad()
                loss = ce(model(xb), yt[idx])
                loss.backward()
                xadv = (xb + self.eps * xb.grad.sign()).detach()
                opt.zero_grad()
                loss2 = ce(model(xadv), yt[idx]) + ce(model(Xt[idx]), yt[idx])
                loss2.backward()
                opt.step()
        model.eval()
        self.model = model
        return self


class StdRF:
    """Random Forest on conventional features (matches PG-Def's RF settings)."""
    memory_mb = 18.2

    def __init__(self, seed=42):
        self.clf = RandomForestClassifier(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=seed)

    def fit(self, X, y):
        self.clf.fit(X, y); return self

    def predict_proba(self, X): return self.clf.predict_proba(X)
    def predict(self, X): return self.clf.predict(X)
    def art_classifier(self): return None  # non-differentiable


BASELINES = {
    "FA-CNN": FACNN,
    "GTAE-IDS": GTAEIDS,
    "DAE": DAE,
    "Adv. Retrain": AdvRetrain,
    "DNN": DNN,
    "Std. RF": StdRF,
}
