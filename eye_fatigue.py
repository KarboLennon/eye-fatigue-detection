import cv2
import time
import math
import os
import csv
import argparse
import numpy as np
from collections import deque
import threading

# =========================
# KONFIG DEFAULT
# =========================
VIDEO_SOURCE = 0
FPS_TARGET = 30
EAR_THRESHOLD_DEFAULT = 0.23        
BLINK_CONSEC_FRAMES = 3
PERCLOS_WINDOW_SEC = 60
PERCLOS_THRESHOLD = 0.25
HYSTERESIS_FRAMES = 7
MIN_ALERT_INTERVAL = 20
DRAW_HAAR_BOX = True
USE_DOWNSCALE = True
DOWNSCALE = 2

VOICE_TEXT = "Istirahatkan mata sejenak."
VOICE_RATE = 170
VOICE_ENABLED_DEFAULT = True

CALIB_OPEN_SEC = 5
CALIB_CLOSED_SEC = 5
CALIB_ALPHA = 0.40 

# =========================
# UTIL
# =========================
def euclid_dist(a, b):
    return math.dist(a, b)

def moving_avg(value, buf, size=5):
    buf.append(value)
    if len(buf) > size:
        buf.pop(0)
    return sum(buf) / len(buf)

def put_text(img, text, org, color=(0,255,0), scale=0.6, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def now_str(fmt="%Y%m%d_%H%M%S"):
    return time.strftime(fmt, time.localtime())

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# =========================
# TTS (non-blocking)
# =========================
class VoiceAlert:
    def __init__(self, text=VOICE_TEXT, rate=VOICE_RATE, enabled=VOICE_ENABLED_DEFAULT):
        self.text = text
        self.enabled = enabled
        self._lock = threading.Lock()
        self._last_ts = 0.0
        self._engine = None
        self._rate = rate

    def speak(self, min_interval=MIN_ALERT_INTERVAL):
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            if now - self._last_ts < min_interval:
                return
            self._last_ts = now
        threading.Thread(target=self._speak_worker, daemon=True).start()

    def say(self, text):
        if not self.enabled:
            return
        threading.Thread(target=self._say_worker, args=(text,), daemon=True).start()

    def _init_engine(self):
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self._rate)

    def _speak_worker(self):
        try:
            self._init_engine()
            self._engine.say(self.text)
            self._engine.runAndWait()
        except Exception:
            pass

    def _say_worker(self, text):
        try:
            self._init_engine()
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            pass

# =========================
# EAR dengan MediaPipe Face Mesh
# =========================
LEFT_EYE_IDXS  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(pts):
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    p1, p2, p3, p4, p5, p6 = pts
    num = euclid_dist(p2, p6) + euclid_dist(p3, p5)
    den = 2.0 * euclid_dist(p1, p4)
    if den == 0:
        return 0.0
    return num / den

def extract_eye_points(landmarks, width, height, idxs):
    return [(landmarks[i].x * width, landmarks[i].y * height) for i in idxs]

# =========================
# PERCLOS & Blink Counter
# =========================
class FatigueMeter:
    def __init__(self, ear_thr=EAR_THRESHOLD_DEFAULT, perclos_sec=PERCLOS_WINDOW_SEC):
        self.ear_thr = ear_thr
        self.window_sec = perclos_sec
        self.states = deque()  
        self.blink_consec = 0
        self.blinks = 0
        self.hysteresis = deque(maxlen=HYSTERESIS_FRAMES)

    def update_threshold(self, new_thr):
        self.ear_thr = float(new_thr)

    def update(self, ear, ts):
        is_closed = ear < self.ear_thr
        self.hysteresis.append(is_closed)
        stable = sum(1 for v in self.hysteresis if v) > len(self.hysteresis) // 2
        self.states.append((ts, stable))

        if stable:
            self.blink_consec += 1
        else:
            if self.blink_consec >= BLINK_CONSEC_FRAMES:
                self.blinks += 1
            self.blink_consec = 0

        while self.states and (ts - self.states[0][0]) > self.window_sec:
            self.states.popleft()

        perclos = (sum(1 for _, c in self.states if c) / len(self.states)) if self.states else 0.0
        return stable, perclos, self.blinks

# =========================
# LOGGING CSV
# =========================
CSV_HEADER = ["timestamp","frame_idx","ear","pred_label","true_label","is_closed","perclos","blinks","fps"]

class CsvLogger:
    def __init__(self, path=None):
        if path is None:
            folder = os.path.join("results","logs")
            ensure_dir(folder)
            path = os.path.join(folder, f"session_{now_str()}.csv")
        self.path = path
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._wr = csv.writer(self._fh)
        self._wr.writerow(CSV_HEADER)
        self._fh.flush()

    def write_row(self, row_dict):
        row = [row_dict.get(k, "") for k in CSV_HEADER]
        self._wr.writerow(row)
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

# =========================
# Kalibrasi
# =========================
def draw_center_banner(frame, lines, color=(0,255,255)):
    h, w = frame.shape[:2]
    y = h//2 - 40
    for i, text in enumerate(lines):
        put_text(frame, text, (30, y + i*25), color, scale=0.8, thick=2)

def collect_ear_samples(cap, face_mesh, duration_sec, mode_text, voice=None):
    """
    Kumpulkan EAR selama duration_sec detik.
    mode_text: "BUKA" / "TUTUP"
    """
    start = time.time()
    buf = []
    ear_smooth_buf = []
    fps = 0.0; fps_t0 = time.time(); fps_counter = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # media pipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        ear = None
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            left_pts = extract_eye_points(lm, w, h, LEFT_EYE_IDXS)
            right_pts = extract_eye_points(lm, w, h, RIGHT_EYE_IDXS)
            ear_left = eye_aspect_ratio(left_pts)
            ear_right = eye_aspect_ratio(right_pts)
            ear_raw = (ear_left + ear_right) / 2.0
            ear = moving_avg(ear_raw, ear_smooth_buf, size=5)
            buf.append(ear)

            # draw eyes
            def draw_eye(pts, color=(255,255,0)):
                pts_i = [(int(x), int(y)) for (x, y) in pts]
                for p in pts_i:
                    cv2.circle(frame, p, 1, color, -1, cv2.LINE_AA)
                cv2.line(frame, pts_i[0], pts_i[3], color, 1, cv2.LINE_AA)
                cv2.line(frame, pts_i[1], pts_i[5], color, 1, cv2.LINE_AA)
                cv2.line(frame, pts_i[2], pts_i[4], color, 1, cv2.LINE_AA)
            draw_eye(left_pts); draw_eye(right_pts)

        left = max(0, duration_sec - int(time.time()-start))
        draw_center_banner(frame, [
            f"MODE KALIBRASI: {mode_text}",
            f"Tahan selama {left} dtk",
            "Ikuti instruksi suara / teks di layar",
        ], (0,255,255))

        # FPS overlay
        fps_counter += 1
        ts = time.time()
        if ts - fps_t0 >= 1.0:
            fps = fps_counter / (ts - fps_t0)
            fps_counter = 0; fps_t0 = ts
        put_text(frame, f"FPS: {fps:.1f}", (w-140, 30), (200,200,200))

        cv2.imshow("Eye Fatigue - Kalibrasi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start >= duration_sec:
            break
    return np.array(buf, dtype=float)

def run_calibration(cap, face_mesh, voice=None):
    if voice: voice.say("Kalibrasi dimulai. Buka mata lebar-lebar selama lima detik.")
    open_samples = collect_ear_samples(cap, face_mesh, CALIB_OPEN_SEC, "BUKA MATA 5 detik", voice=voice)
    if voice: voice.say("Sekarang, tutup mata rapat selama lima detik.")
    closed_samples = collect_ear_samples(cap, face_mesh, CALIB_CLOSED_SEC, "TUTUP MATA 5 detik", voice=voice)

    if open_samples.size < 10 or closed_samples.size < 10:
        if voice: voice.say("Kalibrasi gagal. Wajah kurang stabil terdeteksi.")
        return None, None, None
    ear_open = float(np.median(open_samples))
    ear_closed = float(np.median(closed_samples))

    # guard
    if ear_open <= ear_closed:
        ear_open = float(np.percentile(open_samples, 75))
        ear_closed = float(np.percentile(closed_samples, 25))
        if ear_open <= ear_closed:
            return ear_open, ear_closed, EAR_THRESHOLD_DEFAULT

    thr = ear_closed + CALIB_ALPHA * (ear_open - ear_closed)
    try:
        import yaml
        ensure_dir("configs")
        data = {
            "time": now_str(),
            "ear_open_median": round(ear_open, 5),
            "ear_closed_median": round(ear_closed, 5),
            "alpha": CALIB_ALPHA,
            "threshold": round(thr, 5),
            "note": "EAR threshold hasil kalibrasi per-user"
        }
        with open("configs/calibration.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    except Exception:
        pass

    if voice: voice.say(f"Kalibrasi selesai. Ambang baru diterapkan.")
    return ear_open, ear_closed, float(thr)

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path file CSV log output")
    parser.add_argument("--no-voice", action="store_true", help="Matikan voice alert dari awal")
    parser.add_argument("--source", type=int, default=VIDEO_SOURCE, help="Index kamera (default 0)")
    parser.add_argument("--auto-calib", action="store_true", help="Mulai dengan kalibrasi dahulu")
    args = parser.parse_args()

    voice = VoiceAlert(enabled=(not args.no_voice))
    logger = CsvLogger(args.csv)

    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Gagal membuka kamera. Coba --source 1/2 dst.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ==== Threshold runtime ====
    ear_threshold_runtime = float(EAR_THRESHOLD_DEFAULT)
    fatigue = FatigueMeter(ear_thr=ear_threshold_runtime, perclos_sec=PERCLOS_WINDOW_SEC)
    ear_smooth_buf = []

    # ==== auto-calibration ====
    if args.auto_calib:
        put_text_img = np.zeros((200, 900, 3), dtype=np.uint8)
        put_text(put_text_img, "Auto-calibration aktif: ikuti instruksi di jendela kamera.", (20, 100), (0,255,255), 0.8, 2)
        cv2.imshow("Info", put_text_img)
        cv2.waitKey(1)
        eo, ec, thr = run_calibration(cap, face_mesh, voice=voice)
        cv2.destroyWindow("Info")
        if thr is not None:
            ear_threshold_runtime = float(thr)
            fatigue.update_threshold(ear_threshold_runtime)
        else:
            if not args.no_voice:
                voice.say("Kalibrasi gagal. Menggunakan ambang bawaan.")

    fps_t0 = time.time()
    fps_counter = 0
    fps = 0.0
    frame_idx = 0

    voice_on = (not args.no_voice)
    current_true_label = ""  

    print("[INFO] q=keluar | s=toggle suara | c=kalibrasi | 1=label 'closed' | 0=label 'open' | x=clear label")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Haar wajah 
        if DRAW_HAAR_BOX:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            src = gray; scale = 1
            if USE_DOWNSCALE:
                small = cv2.resize(gray, (w // DOWNSCALE, h // DOWNSCALE))
                src = small; scale = DOWNSCALE
            faces = face_cascade.detectMultiScale(src, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda b: b[2]*b[3])
                x, y, fw, fh = x*scale, y*scale, fw*scale, fh*scale
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 200, 255), 2)

        # Face Mesh
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        ear = None
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            left_pts = extract_eye_points(lm, w, h, LEFT_EYE_IDXS)
            right_pts = extract_eye_points(lm, w, h, RIGHT_EYE_IDXS)
            ear_left = eye_aspect_ratio(left_pts)
            ear_right = eye_aspect_ratio(right_pts)
            ear_raw = (ear_left + ear_right) / 2.0
            ear = moving_avg(ear_raw, ear_smooth_buf, size=7)  

            # gambar mata
            def draw_eye(pts, color=(255, 255, 0)):
                pts_i = [(int(x), int(y)) for (x, y) in pts]
                for p in pts_i:
                    cv2.circle(frame, p, 1, color, -1, cv2.LINE_AA)
                cv2.line(frame, pts_i[0], pts_i[3], color, 1, cv2.LINE_AA)
                cv2.line(frame, pts_i[1], pts_i[5], color, 1, cv2.LINE_AA)
                cv2.line(frame, pts_i[2], pts_i[4], color, 1, cv2.LINE_AA)

            draw_eye(left_pts); draw_eye(right_pts)

        ts = time.time()

        pred_label = ""
        is_closed = False
        perclos = 0.0
        blinks = 0

        if ear is not None:
            is_closed, perclos, blinks = fatigue.update(ear, ts)
            pred_label = "closed" if is_closed else "open"

            col = (0, 0, 255) if is_closed else (0, 255, 0)
            put_text(frame, f"EAR: {ear:.3f} (thr {ear_threshold_runtime:.3f})", (10, 30), (255,255,255))
            put_text(frame, f"Status: {'TERTUTUP' if is_closed else 'TERBUKA'}", (10, 55), col)
            put_text(frame, f"Blinks: {blinks}", (10, 80), (255,255,255))
            put_text(frame, f"PERCLOS({PERCLOS_WINDOW_SEC}s): {perclos:.2f}", (10, 105), (0,255,255))

            if perclos >= PERCLOS_THRESHOLD:
                put_text(frame, "LELAH - ISTIRAHATKAN MATA!", (10, 140), (0,0,255), 0.8, 2)
                if voice_on:
                    VoiceAlert.speak(voice, min_interval=MIN_ALERT_INTERVAL)
        else:
            put_text(frame, "Wajah tidak terdeteksi dengan stabil", (10, 30), (0,0,255))

        # FPS
        fps_counter += 1
        if (ts - fps_t0) >= 1.0:
            fps = fps_counter / (ts - fps_t0)
            fps_counter = 0
            fps_t0 = ts
        put_text(frame, f"FPS: {fps:.1f}", (w-140, 30), (200,200,200))

        # Info kanan atas
        put_text(frame, f"[S]uara: {'ON' if voice_on else 'OFF'}", (w-250, 55), (200,200,200))
        put_text(frame, f"[C]alibrate", (w-250, 80), (200,200,200))
        if current_true_label:
            put_text(frame, f"TrueLabel: {current_true_label}", (w-250, 105), (0,255,255))

        # ====== LOG CSV ======
        logger.write_row({
            "timestamp": f"{ts:.3f}",
            "frame_idx": frame_idx,
            "ear": f"{ear:.5f}" if ear is not None else "",
            "pred_label": pred_label,
            "true_label": current_true_label,
            "is_closed": int(is_closed),
            "perclos": f"{perclos:.5f}",
            "blinks": blinks,
            "fps": f"{fps:.2f}",
        })
        frame_idx += 1

        cv2.imshow("Eye Fatigue - Haar + EAR + Voice + LOG + Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            voice_on = not voice_on
        elif key == ord('1'):
            current_true_label = "closed"
        elif key == ord('0'):
            current_true_label = "open"
        elif key == ord('x'):
            current_true_label = ""
        elif key == ord('c'):
            eo, ec, thr = run_calibration(cap, face_mesh, voice=voice if voice_on else None)
            cv2.destroyWindow("Eye Fatigue - Kalibrasi")
            if thr is not None:
                ear_threshold_runtime = float(thr)
                fatigue.update_threshold(ear_threshold_runtime)
                if voice_on:
                    voice.say(f"Ambang baru diterapkan.")
            else:
                if voice_on:
                    voice.say("Kalibrasi gagal. Gunakan pencahayaan lebih baik dan ulangi.")

    cap.release()
    cv2.destroyAllWindows()
    logger.close()

if __name__ == "__main__":
    main()
