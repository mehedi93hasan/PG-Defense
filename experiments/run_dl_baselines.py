"""Table X -- representative deep-learning architectures (MLP, 1D-CNN, AE-DNN)
trained on all 30 features and attacked by mimicry, vs PG-Def. Reports clean and
attacked ROC-AUC and TPR@1%FPR. Requires torch."""
import numpy as np
from _common import base_parser
from pgdef.features import COLUMN_NAMES, PROTECTED, MANIPULABLE, resolve
from pgdef.model import PGDef
from pgdef import attacks, metrics
from pgdef.data import load_csv, split, standardise, smote
from pgdef.io_utils import dataset_key, save_results
from sklearn.neural_network import MLPClassifier

try:
    import torch, torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

if _HAS_TORCH:
    class CNN(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.c = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
                                nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
                                nn.AdaptiveAvgPool1d(8))
            s.f = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8, 64), nn.ReLU(),
                                nn.Dropout(0.2), nn.Linear(64, 1))
        def forward(s, x): return s.f(s.c(x.unsqueeze(1))).squeeze(-1)

    class AEClf(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
            s.dec = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, d))
            s.head = nn.Linear(16, 1)
        def recon(s, x): return s.dec(s.enc(x))
        def forward(s, x): return s.head(s.enc(x)).squeeze(-1)

    def train_mb(model, X, y, epochs, bs=256, recon=False, seed=0):
        torch.manual_seed(seed); opt = torch.optim.Adam(model.parameters(), 1e-3)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = None if y is None else torch.tensor(y, dtype=torch.float32)
        mse, bce, n = nn.MSELoss(), nn.BCEWithLogitsLoss(), len(Xt)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; opt.zero_grad()
                loss = mse(model.recon(Xt[idx]), Xt[idx]) if recon else bce(model(Xt[idx]), yt[idx])
                loss.backward(); opt.step()
        return model.eval()

    def predict(m, X):
        with torch.no_grad():
            return torch.sigmoid(m(torch.tensor(X, dtype=torch.float32))).numpy()

def main():
    ap = base_parser("Deep-learning baselines"); ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(); out = {}
    for path in a.csv:
        key = dataset_key(path); df = load_csv(path)
        allc = resolve(df.columns, COLUMN_NAMES); protc = resolve(df.columns, PROTECTED)
        manip = resolve(df.columns, MANIPULABLE)
        Xtr, Xte, ytr, yte = split(df, allc, seed=a.seed)
        midx = [allc.index(c) for c in manip]; pidx = [allc.index(c) for c in protc]
        bmean = Xtr[ytr == 0].mean(0); Xmim = attacks.mimicry(Xte, yte, midx, bmean)
        sc, Xtr_s, Xte_s = standardise(Xtr, Xte); Xmim_s = sc.transform(Xmim)
        Xsm, ysm = smote(Xtr_s, ytr, a.seed)
        rec = {}
        mlp = MLPClassifier((128, 64, 32), max_iter=120, early_stopping=True,
                            random_state=a.seed).fit(Xsm, ysm)
        rec["mlp"] = _row(mlp.predict_proba(Xte_s)[:, 1], mlp.predict_proba(Xmim_s)[:, 1], yte, a.target_fpr)
        if _HAS_TORCH:
            cnn = train_mb(CNN(len(allc)), Xsm, ysm, a.epochs, seed=a.seed)
            rec["cnn"] = _row(predict(cnn, Xte_s), predict(cnn, Xmim_s), yte, a.target_fpr)
            ae = AEClf(len(allc)); train_mb(ae, Xtr_s, None, max(a.epochs // 2, 10), recon=True, seed=a.seed)
            train_mb(ae, Xsm, ysm, a.epochs, seed=a.seed)
            rec["ae_dnn"] = _row(predict(ae, Xte_s), predict(ae, Xmim_s), yte, a.target_fpr)
        pg = PGDef(seed=a.seed).fit(Xtr[:, pidx], ytr)
        rec["pgdef"] = _row(pg.proba(Xte[:, pidx]), pg.proba(Xmim[:, pidx]), yte, a.target_fpr)
        out[key] = rec
        for k, v in rec.items():
            print(f"[{key}] {k:8s} clean AUC={v['clean_auc']} TPR={v['clean_tpr']} | "
                  f"atk AUC={v['atk_auc']} TPR={v['atk_tpr']}")
    save_results("dl_baselines", out)

def _row(s_clean, s_atk, y, fpr):
    c = metrics.operating_point(s_clean, y, fpr); aatk = metrics.operating_point(s_atk, y, fpr)
    return {"clean_auc": round(c["auc"], 3), "clean_tpr": round(c["tpr"], 1),
            "atk_auc": round(aatk["auc"], 3), "atk_tpr": round(aatk["tpr"], 1)}

if __name__ == "__main__":
    main()
