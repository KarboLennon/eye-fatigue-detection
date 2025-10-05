import argparse, os, glob
import pandas as pd
import numpy as np
from collections import deque

# ---------- utils ----------
def smooth_ear(series, win=5):
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.array([])
    k = max(1, int(win))
    if k <= 1 or len(arr) < k:
        return arr
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    core = (csum[k:] - csum[:-k]) / k
    pad_front = np.full(k//2, core[0])
    pad_back  = np.full(len(arr)-len(core)-len(pad_front), core[-1])
    return np.concatenate([pad_front, core, pad_back])

def normalize_true_label(col):
    """
    Bersihkan kolom label ke 'open' / 'closed' saja.
    Support: 1/0, 'closed'/'open', 'tutup'/'buka', spasi/NaN.
    """
    s = col.astype("string")
    s = s.str.strip().str.lower()

    # map variasi
    mapping = {
        "1": "closed", "closed": "closed", "tutup": "closed", "tertutup": "closed",
        "0": "open",   "open": "open",     "buka": "open",   "terbuka": "open",
    }
    s = s.map(lambda x: mapping.get(x, x))
    s = s.where(s.isin(["open", "closed"]), other=pd.NA)
    return s

def apply_hysteresis(ears, thr=0.23, h_frames=7, blink_frames=4):
    """
    Replikasi simplifikasi logika runtime:
    - closed_raw = (ear < thr)
    - stabil pakai mayoritas dari window hysteresis
    """
    preds = []
    if len(ears) == 0:
        return preds
    hist = deque(maxlen=max(1, h_frames))
    blink_consec = 0
    for e in ears:
        is_closed = (e < thr)
        hist.append(is_closed)
        stable = sum(hist) > (len(hist) // 2)
        if stable:
            blink_consec += 1
        else:
            if blink_consec >= blink_frames:
                pass
            blink_consec = 0
        preds.append("closed" if stable else "open")
    return preds

def f1_from_cm(tp, fp, fn):
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
    return f1, prec, rec

def eval_file(path, thr_grid, win=7, h_frames=9, blink_frames=5):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[SKIP] {os.path.basename(path)} gagal dibaca: {e}")
        return None

    if "ear" not in df.columns or "true_label" not in df.columns:
        print(f"[SKIP] {os.path.basename(path)} tidak punya kolom 'ear'/'true_label'.")
        return None

    df["ear"] = pd.to_numeric(df["ear"], errors="coerce")
    df = df.dropna(subset=["ear"])
    if df.empty:
        print(f"[SKIP] {os.path.basename(path)} semua 'ear' kosong/invalid.")
        return None

    df["true_label"] = normalize_true_label(df["true_label"])
    df = df.dropna(subset=["true_label"])
    if df.empty:
        print(f"[SKIP] {os.path.basename(path)} tidak ada baris dengan true_label valid (open/closed).")
        return None
    ears = smooth_ear(df["ear"], win=win)
    if len(ears) != len(df):
        ears = df["ear"].to_numpy(dtype=float)

    y_true = df["true_label"].tolist()

    best = None
    for thr in thr_grid:
        y_pred = apply_hysteresis(ears, thr=float(thr), h_frames=h_frames, blink_frames=blink_frames)
        n = min(len(y_true), len(y_pred))
        y_t = y_true[:n]
        y_p = y_pred[:n]

        tp = sum(1 for t,p in zip(y_t,y_p) if t=="closed" and p=="closed")
        tn = sum(1 for t,p in zip(y_t,y_p) if t=="open"  and p=="open")
        fp = sum(1 for t,p in zip(y_t,y_p) if t=="open"  and p=="closed")
        fn = sum(1 for t,p in zip(y_t,y_p) if t=="closed" and p=="open")
        acc = (tp+tn)/max(1, tp+tn+fp+fn)
        f1, prec, rec = f1_from_cm(tp, fp, fn)

        cand = {
            "file": os.path.basename(path),
            "n_eval": (tp+tn+fp+fn),
            "thr": float(thr),
            "win": int(win),
            "hyst": int(h_frames),
            "blink_frames": int(blink_frames),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        }
        if best is None or cand["f1"] > best["f1"]:
            best = cand
    return best

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/logs/*.csv")
    ap.add_argument("--win", type=int, default=7, help="moving average window")
    ap.add_argument("--hyst", type=int, default=9, help="hysteresis frames")
    ap.add_argument("--blink", type=int, default=5, help="blink consecutive frames")
    ap.add_argument("--min_thr", type=float, default=0.16)
    ap.add_argument("--max_thr", type=float, default=0.30)
    ap.add_argument("--step", type=float, default=0.002)
    ap.add_argument("--out", default="results/threshold_tuning.csv")
    args = ap.parse_args()

    files = glob.glob(args.glob)
    if not files:
        print("Tidak ada file yang cocok dengan pattern:", args.glob)
        return

    thr_grid = np.arange(args.min_thr, args.max_thr + 1e-9, args.step)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for f in files:
        res = eval_file(f, thr_grid, win=args.win, h_frames=args.hyst, blink_frames=args.blink)
        if res:
            rows.append(res)
            print(f"{res['file']} → best thr={res['thr']:.3f} | F1={res['f1']:.3f} | "
                  f"P={res['precision']:.3f} R={res['recall']:.3f} Acc={res['accuracy']:.3f}")

    if rows:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"Saved: {args.out}")
    else:
        print("Tidak ada hasil yang bisa disimpan.")

if __name__ == "__main__":
    main()
