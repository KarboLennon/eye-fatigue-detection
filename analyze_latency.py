import pandas as pd

def latency_closed_detection(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].astype(float)
    df = df[df['true_label'].astype(str).str.len() > 0].copy()
    df['true_label'] = df['true_label'].str.lower()
    times = []
    for i in range(len(df)-1):
        if df.iloc[i]['true_label']=='open' and df.iloc[i+1]['true_label']=='closed':
            t0 = df.iloc[i+1]['timestamp']
            later = df.iloc[i+1:].reset_index(drop=True)
            hit = later[later['pred_label']=='closed']
            if not hit.empty:
                t1 = float(hit.iloc[0]['timestamp'])
                times.append(t1 - t0)
    if times:
        return pd.Series(times).describe()
    return None

def time_to_perclos_alert(csv_path, thr=0.25):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['timestamp','perclos'])
    df['timestamp'] = df['timestamp'].astype(float)
    df['perclos'] = df['perclos'].astype(float)
    hit = df[df['perclos'] >= thr]
    if hit.empty:
        return None
    t_first = float(hit.iloc[0]['timestamp'])
    t_start = float(df.iloc[0]['timestamp'])
    return t_first - t_start

if __name__ == "__main__":
    path = "results/logs/L1_normal_light.csv"
    print("Latency closed-detection (s):")
    print(latency_closed_detection(path))
    print("Time to PERCLOS alert (s):", time_to_perclos_alert(path, thr=0.25))
