import argparse
import os
import glob
import pandas as pd

def safe_metrics(y_true, y_pred, pos_label="closed"):
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[pos_label, "open"])
        return acc, prec, rec, f1, cm
    except Exception:
        # Manual
        labels = [pos_label, "open"]
        tp = sum(1 for t,p in zip(y_true,y_pred) if t==pos_label and p==pos_label)
        tn = sum(1 for t,p in zip(y_true,y_pred) if t=="open" and p=="open")
        fp = sum(1 for t,p in zip(y_true,y_pred) if t=="open" and p==pos_label)
        fn = sum(1 for t,p in zip(y_true,y_pred) if t==pos_label and p=="open")
        total = max(1, tp+tn+fp+fn)
        acc = (tp+tn)/total
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        cm = [[tp, fn],[fp, tn]]  
        return acc, prec, rec, f1, cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", help="Path CSV (boleh banyak) atau glob pattern", required=False)
    ap.add_argument("--glob", type=str, default="results/logs/*.csv", help="Glob pattern jika --csv tidak dipakai")
    ap.add_argument("--save", type=str, default="results/metrics_report.csv", help="Path simpan ringkasan metrik")
    args = ap.parse_args()

    # Kumpulkan file
    files = []
    if args.csv:
        for f in args.csv:
            if any(ch in f for ch in ["*","?","["]):
                files += glob.glob(f)
            else:
                files.append(f)
    else:
        files = glob.glob(args.glob)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("Tidak ada file CSV ditemukan.")
        return

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    rows_summary = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Gagal baca {path}: {e}")
            continue

        df_eval = df[df["true_label"].astype(str).str.len() > 0].copy()
        if df_eval.empty:
            print(f"[SKIP] {os.path.basename(path)} tidak ada true_label.")
            continue

        y_true = df_eval["true_label"].astype(str).str.lower().tolist()
        y_pred = df_eval["pred_label"].astype(str).str.lower().tolist()

        acc, prec, rec, f1, cm = safe_metrics(y_true, y_pred, pos_label="closed")

        print(f"=== {os.path.basename(path)} ===")
        print("Confusion Matrix (labels=['closed','open']):")
        print(cm)
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1-score : {f1:.3f}")
        print()

        rows_summary.append({
            "file": os.path.basename(path),
            "n_eval": len(df_eval),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

    if rows_summary:
        out = pd.DataFrame(rows_summary).sort_values("f1", ascending=False)
        out.to_csv(args.save, index=False)
        print(f"Ringkasan tersimpan → {args.save}")
    else:
        print("Tidak ada hasil untuk disimpan (tidak ada data berlabel).")

if __name__ == "__main__":
    main()
