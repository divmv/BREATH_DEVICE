import serial
import time
import os
import pandas as pd
from datetime import datetime

# ---------- CONFIG ----------
PORT         = "COM9"
BAUD         = 1000000
SAVE_DIR     = r"C:\Users\Alex\Dice_lab\Breath_device\DICE_lab_breath_device\captures"
TARGET_ROWS  = 30_000                      # EXACT number required
DEFAULT_HDR  = ["Samples","Xpos","Ypos","Pow","NebMode"]
PROGRESS_EVERY = 5_000                     # progress print interval
# ----------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def open_serial(port: str, baud: int) -> serial.Serial:
    ser = serial.Serial(port, baud, timeout=0.2, write_timeout=0)
    try:
        if hasattr(ser, "set_buffer_size"):
            ser.set_buffer_size(rx_size=262_144, tx_size=131_072)
        ser.setDTR(False); ser.setRTS(False)
    except Exception:
        pass
    time.sleep(0.2)
    ser.reset_input_buffer(); ser.reset_output_buffer()
    return ser

def main():
    ser = open_serial(PORT, BAUD)

    save_dir = ensure_dir(SAVE_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"analog_{ts}.csv")
    print(f"[i] Capturing EXACTLY {TARGET_ROWS} rows from {PORT} @ {BAUD}")
    print(f"[i] Saving to {out_path}")

    # Phase 1: capture-only (avoid splitting during capture for speed)
    raw_lines = []
    header_line = None
    data_count = 0
    t0 = time.time()

    try:
        while data_count < TARGET_ROWS:
            b = ser.read_until(b'\n')
            if not b:
                continue

            # quick strip without split for now
            s = b.decode("utf-8", errors="ignore").strip()
            if not s:
                continue

            # If we haven't seen a header yet and the line has letters, treat as header
            if header_line is None and any(ch.isalpha() for ch in s):
                header_line = s
                continue

            # assume it's data
            raw_lines.append(s)
            data_count += 1

            if data_count % PROGRESS_EVERY == 0:
                dt = time.time() - t0
                rate = data_count / max(dt, 1e-6)
                print(f"[i] {data_count}/{TARGET_ROWS} captured  (~{rate:.0f} rows/s)")
    except KeyboardInterrupt:
        print("\n[!] Interrupted. Still writing what we captured…")
    finally:
        try: ser.close()
        except Exception: pass

    if not raw_lines:
        print("[!] No data captured. Check baud/COM and that firmware sends '\\n'-terminated lines.")
        return

    # If somehow we overshot (e.g., multiple lines buffered right at the end), trim.
    if len(raw_lines) > TARGET_ROWS:
        raw_lines = raw_lines[:TARGET_ROWS]

    elapsed = time.time() - t0
    rate = len(raw_lines) / max(elapsed, 1e-6)
    print(f"[i] Capture complete: {len(raw_lines)} rows in {elapsed:.3f}s (~{rate:.0f} rows/s)")

    # Phase 2: parse/split once
    # Determine header
    if header_line is not None:
        header = [p.strip() for p in header_line.split(",")]
    else:
        # infer width from first data line
        first_parts = [p.strip() for p in raw_lines[0].split(",")]
        header = DEFAULT_HDR[:len(first_parts)]

    cols = len(header)
    # build rows with normalized length
    rows = []
    for s in raw_lines:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) < cols:
            parts = parts + [""]*(cols - len(parts))
        else:
            parts = parts[:cols]
        rows.append(parts)

    df = pd.DataFrame(rows, columns=header)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[i] Saved EXACTLY {len(df)} rows to: {out_path}")

if __name__ == "__main__":
    main()
