"""
Train the CNN driver on recorded sessions (CPU).

    python scripts/train_cnn.py                                  # every valid session in data/raw
    python scripts/train_cnn.py --init models/cnn.pt --epochs 4 --lr 5e-4 --out models/cnn_new.pt  # fine-tune after a DAgger round

Targets: steering (MSE), throttle (MSE), brake on/off (binary cross-entropy).
Validation = the last 15 % of each session (time-wise, so neighbouring
near-identical frames do not leak into validation).
Preprocessed images are cached in data/processed/<session>.npz (built once).
Output: models/cnn.pt (best validation loss) + models/cnn_train.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                        # noqa: E402
import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402
from PIL import Image                                                     # noqa: E402

from donkey.cnn import SPEED_SCALE, DriveNet, preprocess, to_tensor       # noqa: E402
from donkey.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, PROJECT_ROOT  # noqa: E402
from donkey.session import load_session, validate_session                 # noqa: E402


def load_one(p: Path):
    """Preprocessed arrays of one session, cached in data/processed/."""
    cache = DATA_PROCESSED_DIR / f"{p.name}_v2.npz"     # v2: raw brake amount as 3rd target
    if cache.is_file():
        d = np.load(cache)
        return d["X"], d["V"], d["Y"]
    _, rows = load_session(p)
    X = np.stack([preprocess(np.asarray(Image.open(p / r["image"]).convert("RGB"))) for r in rows])
    V = np.asarray([r["speed"] / SPEED_SCALE for r in rows], np.float32)[:, None]
    # third target = brake amount (0..1); the "binary" mode thresholds it at > 0
    Y = np.asarray([(r["cmd_steering"], r["cmd_throttle"], r["cmd_brake"]) for r in rows], np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, V=V, Y=Y)
    return X, V, Y


def load(paths, val_frac):
    Xs, Vs, Ys, vals = [], [], [], []
    for p in paths:
        t = time.time()
        X, V, Y = load_one(p)
        n = len(X)
        Xs.append(X), Vs.append(V), Ys.append(Y)
        vals.append(np.arange(n) >= n * (1 - val_frac))
        print(f"  {p.name}: {n} frames ({time.time() - t:.0f} s)")
    return np.concatenate(Xs), np.concatenate(Vs), np.concatenate(Ys), np.concatenate(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--brake-mode", choices=["binary", "regression"], default="regression",
                    help="binary: on/off at 0.3 (v3 model); regression: brake amount 0..1")
    ap.add_argument("--init", default=None, help="start from this checkpoint (matching layers only)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "models" / "cnn.pt"))
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)

    paths = [Path(p) for p in args.sessions] or sorted(
        p for p in DATA_RAW_DIR.glob("*") if (p / "meta.json").is_file())
    paths = [p for p in paths if validate_session(p, decode_every=200)["ok"]]
    print(f"Loading {len(paths)} session(s)")
    X, V, Y, is_val = load(paths, args.val)
    if args.brake_mode == "binary":
        Y[:, 2] = (Y[:, 2] > 0).astype(np.float32)
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    print(f"{len(tr)} train / {len(va)} val frames, brake frames: {int((Y[:, 2] > 0).sum())}")

    model = DriveNet()
    if args.init:
        old = torch.load(args.init, map_location="cpu")["model"]
        own = model.state_dict()
        kept = {k: v for k, v in old.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(kept, strict=False)
        print(f"initialized {len(kept)}/{len(own)} tensors from {args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.default_rng(0)
    pos = max(1.0, float((Y[tr, 2] > 0).sum()))
    brake_pos_weight = torch.tensor((len(tr) - pos) / pos)   # brake frames are rare

    def batch_loss(idx, train):
        x = to_tensor(X[idx])
        if train:   # brightness / contrast jitter (sim lighting is randomized)
            x = (x * torch.empty(len(idx), 1, 1, 1).uniform_(0.7, 1.3)
                 + torch.empty(len(idx), 1, 1, 1).uniform_(-0.1, 0.1)).clamp_(0, 1)
        y = torch.from_numpy(Y[idx])
        steer, thr, brake_logit = model(x, torch.from_numpy(V[idx]))
        ls = F.mse_loss(steer[:, 0], y[:, 0])
        lt = F.mse_loss(thr[:, 0], y[:, 1])
        if args.brake_mode == "binary":
            lb = F.binary_cross_entropy_with_logits(brake_logit[:, 0], y[:, 2], pos_weight=brake_pos_weight)
        else:   # amount: MSE on sigmoid output, braking frames weighted up (they are rare)
            w = 1.0 + (y[:, 2] > 0).float() * (brake_pos_weight.clamp(max=10.0) - 1.0)
            lb = (w * (torch.sigmoid(brake_logit[:, 0]) - y[:, 2]) ** 2).mean() * 10.0
        acc = ((brake_logit[:, 0] > 0).float() == (y[:, 2] > 0).float()).float().mean()
        return ls + 0.5 * lt + 0.1 * lb, (ls.item(), lt.item(), lb.item(), acc.item())

    best, history = float("inf"), []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        t = time.time()
        model.train()
        rng.shuffle(tr)
        for i in range(0, len(tr), args.batch):
            loss, _ = batch_loss(tr[i:i + args.batch], True)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            parts = [(len(va[i:i + 256]), batch_loss(va[i:i + 256], False)[1]) for i in range(0, len(va), 256)]
        vs, vt, vb, vacc = (sum(n * m[k] for n, m in parts) / len(va) for k in range(4))
        val = vs + 0.5 * vt + 0.1 * vb
        history.append({"epoch": ep + 1, "val_steer_mse": round(vs, 5), "val_throttle_mse": round(vt, 5),
                        "val_brake_acc": round(vacc, 4), "seconds": round(time.time() - t, 1)})
        tag = ""
        if val < best:
            best, tag = val, "  *saved"
            torch.save({"model": model.state_dict(),
                        "meta": {"sessions": [p.name for p in paths], "epoch": ep + 1,
                                 "brake_mode": args.brake_mode,
                                 "val_steer_mse": vs, "val_throttle_mse": vt, "val_brake_acc": vacc}},
                       args.out)
        print(f"epoch {ep + 1}/{args.epochs}  steer RMSE {vs ** 0.5:.3f}  throttle MSE {vt:.4f}  "
              f"brake acc {100 * vacc:.1f}%  [{time.time() - t:.0f} s]{tag}", flush=True)
    Path(args.out).with_name(Path(args.out).stem + "_train.json").write_text(json.dumps(
        {"sessions": [p.name for p in paths], "train": len(tr), "val": len(va),
         "steer_std": float(Y[:, 0].std()), "history": history}, indent=2))
    print(f"steering std in val data: {Y[va, 0].std():.3f} (RMSE of a constant prediction)")
    print("model:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
