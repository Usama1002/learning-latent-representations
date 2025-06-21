import os
import numpy as np
import pandas as pd
import random
from eye_label import eye_diagram_label

DATA_DIR = "data/original_data"
OUT_DIR = "data/prepared_data"
FILES = [
    "1_ISI.csv",
    "2_ISI.csv",
    "3_ISI.csv",
    "4_ISI.csv",
    "5_XTALK.csv"
]
SEGMENT_LENGTH = 10000
CHANNELS = 16
VALID_RATIO = 0.88  # Not used for labeling anymore

def ensure_dirs(base, subdirs):
    for sub in subdirs:
        os.makedirs(os.path.join(base, sub), exist_ok=True)

def split_and_save_vectors(file_path, out_base, is_xtalk=False):
    df = pd.read_csv(file_path)
    if is_xtalk:
        input_cols = [f'v(din{str(i).zfill(2)}) ' for i in range(CHANNELS)]
        output_cols = [f'v(dq{str(i).zfill(2)}_ctl) ' for i in range(CHANNELS)]
        time_col = 'TIME '
        time = df[time_col].values
        for ch in range(CHANNELS):
            in_sig = df[input_cols[ch]].values
            out_sig = df[output_cols[ch]].values
            n_segments = len(in_sig) // SEGMENT_LENGTH
            for seg in range(n_segments):
                x = in_sig[seg*SEGMENT_LENGTH:(seg+1)*SEGMENT_LENGTH]
                y = out_sig[seg*SEGMENT_LENGTH:(seg+1)*SEGMENT_LENGTH]
                t = time[seg*SEGMENT_LENGTH:(seg+1)*SEGMENT_LENGTH]
                # Use eye diagram based labeling on output signal
                label = eye_diagram_label(y, t)
                fname = f'ch{ch:02d}_seg{seg:04d}.npz'
                np.savez_compressed(os.path.join(out_base, label, fname), x=x, y=y)
    else:
        for ch in range(CHANNELS):
            in_col = f'v(din{str(ch).zfill(2)}) '
            out_col = f'v(dq{str(ch).zfill(2)}_ctl) '
            time_col = 'TIME ' if ch == 0 else f'TIME .{ch}'
            in_sig = df[in_col].values
            out_sig = df[out_col].values
            t = df[time_col].values
            n_segments = len(in_sig) // SEGMENT_LENGTH
            for seg in range(n_segments):
                x = in_sig[seg*SEGMENT_LENGTH:(seg+1)*SEGMENT_LENGTH]
                y = out_sig[seg*SEGMENT_LENGTH:(seg+1)*SEGMENT_LENGTH]
                t_seg = t[seg*SEGMENT_LENGTH:(seg+1)*SEGMENT_LENGTH]
                # Use eye diagram based labeling on output signal
                label = eye_diagram_label(y, t_seg)
                fname = f'ch{ch:02d}_seg{seg:04d}.npz'
                np.savez_compressed(os.path.join(out_base, label, fname), x=x, y=y)

def main():
    for fname in FILES:
        file_path = os.path.join(DATA_DIR, fname)
        out_base = os.path.join(OUT_DIR, fname.replace('.csv', ''))
        ensure_dirs(out_base, ['valid', 'invalid'])
        is_xtalk = fname == "5_XTALK.csv"
        split_and_save_vectors(file_path, out_base, is_xtalk)
        print(f"Processed {fname}")

if __name__ == "__main__":
    main()
