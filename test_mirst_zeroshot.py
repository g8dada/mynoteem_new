"""Zero-shot evaluation of EffNetb0/AMTAdapter on MIR-ST500.

Runs the pre-trained AST backbone (no EM fine-tuning) on all #0 (unshifted)
segments in the NoteEM audio directory and writes:
  - mirst_zeroshot_results.json  — per-song pred + gt note lists in absolute time
  - mirst_zeroshot_viz/          — piano-roll PNGs for the first N_VIZ segments
  - mirst_zeroshot_viz/arrays/    — raw onset/frame arrays for all segments, onset_np = data['onset'], frame_np = data['frame']

Usage:
    python test_mirst_zeroshot.py [--limit N]
"""

import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import soundfile
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onsets_and_frames.constants import HOP_LENGTH, MIN_MIDI, N_KEYS, SAMPLE_RATE
from onsets_and_frames.ast_model import EffNetb0
from onsets_and_frames.dataset import _compute_cqt
from onsets_and_frames.transcriber import AMTAdapter
from onsets_and_frames.utils import get_peaks

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AUDIO_DIR  = '/data/hakka/mynoteem_new/data/mirst500_15sec_data_full_quantized_NoteEM_audio'
MODEL_PATH = '/data/hakka/singing_transcription_ICASSP2021/AST/models/1005_e_4'
GT_JSON    = '/data/hakka/singing_transcription_ICASSP2021/MIR-ST500_20210206/MIR-ST500_corrected.json'
OUT_JSON   = '/data/hakka/mynoteem_new/test_zeroshot/mirst_zeroshot_results.json'
VIZ_DIR    = '/data/hakka/mynoteem_new/test_zeroshot'
ARRAYS_DIR = '/data/hakka/mynoteem_new/test_zeroshot/arrays'

SEGMENT_HOP      = 15.0  # seconds between segment starts (matches preprocess_mirst500.py)
ONSET_THRESHOLD  = 0.05  # minimum post-peak onset probability to count as a note
FRAME_THRESHOLD  = 0.05  # minimum frame probability to sustain a note
MIN_NOTE_FRAMES  = 3     # discard notes shorter than this many frames (~96 ms at 31.25 fps)
N_VIZ            = 20    # number of segments to save as piano-roll PNGs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    backbone = EffNetb0()
    backbone.load_state_dict(
        torch.load(MODEL_PATH, map_location='cpu', weights_only=False),
        strict=False,
    )
    model = AMTAdapter(backbone).cuda().eval()
    return model


# ---------------------------------------------------------------------------
# Segment discovery
# ---------------------------------------------------------------------------

_SEG_RE = re.compile(r'^mirst_(\d+)_seg(\d+)#0$')


def collect_segments(audio_dir):
    """Return sorted list of (sid_str, seg_idx, flac_path) for all #0 clips."""
    segments = []
    for entry in os.scandir(audio_dir):
        m = _SEG_RE.match(entry.name)
        if m is None:
            continue
        sid     = m.group(1)
        seg_idx = int(m.group(2))
        flac    = os.path.join(entry.path, f'{entry.name}.flac')
        if os.path.isfile(flac):
            segments.append((sid, seg_idx, flac))
    segments.sort(key=lambda x: (int(x[0]), x[1]))
    return segments


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_segment(model, flac_path):
    """Load one FLAC, run AMTAdapter, return (onset_np, frame_np) as (T_mel, 88) arrays."""
    audio_raw, sr = soundfile.read(flac_path, dtype='int16')
    if audio_raw.ndim == 2:
        audio_raw = audio_raw.mean(axis=1).astype(np.int16)
    audio_short = torch.ShortTensor(audio_raw)

    cqt     = _compute_cqt(audio_short)          # (168, T_cqt)
    audio_f = audio_short.float() / 32768.0       # (L,) float in [-1, 1]

    with torch.no_grad():
        onset_pred, _, _, frame_pred, _ = model(
            audio_f.unsqueeze(0).cuda(),
            cqt=cqt.unsqueeze(0).cuda(),
        )

    onset_np = onset_pred.squeeze(0).cpu().numpy()   # (T_mel, 88)
    frame_np = frame_pred.squeeze(0).cpu().numpy()   # (T_mel, 88)
    return onset_np, frame_np


def extract_notes(onset_np, frame_np, is_last_seg):
    """Convert raw adapter output to segment-relative note list [onset_s, offset_s, midi].

    For non-last segments the audio extends past 15 s (preprocess_mirst500.py captures
    note offsets), but the notes assigned to this segment have onset < SEGMENT_HOP.
    We clip to [0, SEGMENT_HOP) so overlap with the next segment is avoided.

    Steps:
      1. Peak-find on onset_np (local maxima in ±3-frame window).
      2. Per-pitch scan: when an onset peak is found, track frame activity to find
         the offset, then jump past it — so secondary onset peaks that fire while
         a note is already sustained do NOT spawn additional overlapping notes.
      3. Filter notes shorter than MIN_NOTE_FRAMES.
    Returns notes sorted by onset (segment-relative seconds).
    """
    onset_t   = torch.from_numpy(onset_np)
    peaks     = get_peaks(onset_t, win_size=3)   # (T_mel, 88) bool
    onset_pkd = onset_np.copy()
    onset_pkd[~peaks.numpy()] = 0.0

    onsets_bin = (onset_pkd > ONSET_THRESHOLD).astype(np.uint8)   # (T, 88)
    frames_bin = (frame_np  > FRAME_THRESHOLD).astype(np.uint8)   # (T, 88)

    T, P        = onsets_bin.shape
    fps         = SAMPLE_RATE / HOP_LENGTH                         # 31.25 frames / s
    primary_end = int(SEGMENT_HOP * fps) if not is_last_seg else T

    notes = []
    for pitch in range(P):
        t = 0
        while t < primary_end:
            if not onsets_bin[t, pitch]:
                t += 1
                continue
            # Onset found at frame t — track forward until frame activity ends.
            f_on = t
            f_off = t
            while f_off < T and (onsets_bin[f_off, pitch] or frames_bin[f_off, pitch]):
                f_off += 1
            if f_off - f_on >= MIN_NOTE_FRAMES:
                midi = pitch + MIN_MIDI
                notes.append([round(f_on / fps, 6), round(f_off / fps, 6), midi])
            # Jump past the end of this note so mid-note onset peaks are skipped.
            t = f_off if f_off > t else t + 1

    notes.sort()
    return notes


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_segment(viz_dir, sid, seg_idx, pred_rel, gt_rel, seg_dur):
    """Save a piano-roll PNG.

    pred_rel, gt_rel : [[onset_s, offset_s, midi], ...] already in segment-relative seconds.
    seg_dur          : actual audio duration of this segment (x-axis limit).
    """
    os.makedirs(viz_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(f'Zero-shot  mirst_{sid}_seg{seg_idx}  '
                 f'({len(pred_rel)} pred / {len(gt_rel)} GT notes)',
                 fontweight='bold')

    for (onset_s, offset_s, midi) in gt_rel:
        ax.barh(midi, max(offset_s - onset_s, 0.05), left=onset_s,
                height=0.7, fill=False, edgecolor='#1f77b4', linewidth=1.2)
    for (onset_s, offset_s, midi) in pred_rel:
        ax.barh(midi, max(offset_s - onset_s, 0.05), left=onset_s,
                height=0.7, color='#ff7f0e', alpha=0.75)

    ax.set_xlim(0, seg_dur)
    ax.set_ylim(MIN_MIDI - 1, MIN_MIDI + N_KEYS)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('MIDI pitch')
    ax.legend(handles=[
        Patch(facecolor='none', edgecolor='#1f77b4', linewidth=1.2, label='GT (MIR-ST500)'),
        Patch(facecolor='#ff7f0e', alpha=0.75,                       label='pred (zero-shot)'),
    ], loc='upper right', fontsize=9)

    plt.tight_layout()
    out = os.path.join(viz_dir, f'mirst_{sid}_seg{seg_idx}.png')
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    # print(f'Saved viz: {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N segments (for quick testing)')
    args = parser.parse_args()

    print('Loading model...')
    model = load_model()

    print('Loading GT JSON...')
    with open(GT_JSON) as f:
        gt_data = json.load(f)   # {sid_str: [[onset_abs, offset_abs, midi], ...]}

    print('Scanning audio directory...')
    segments = collect_segments(AUDIO_DIR)
    if args.limit:
        segments = segments[:args.limit]
    print(f'Found {len(segments)} #0 segments to process.')

    # Pre-compute the last seg_idx per song so we know not to clip that segment.
    last_seg_of = {}
    for sid, seg_idx, _ in segments:
        last_seg_of[sid] = max(last_seg_of.get(sid, 0), seg_idx)

    results = {}   # 'mirst_{sid}_seg{i}' -> {'pred': [...], 'gt': [...]}

    for i, (sid, seg_idx, flac_path) in enumerate(tqdm(segments, desc='Segments')):
        seg_start   = seg_idx * SEGMENT_HOP
        is_last_seg = (seg_idx == last_seg_of[sid])
        seg_key     = f'mirst_{sid}_seg{seg_idx}'

        # GT notes for this segment (segment-relative, onset in primary window)
        seg_end_gt = float('inf') if is_last_seg else seg_start + SEGMENT_HOP
        gt_rel = [
            [round(float(o) - seg_start, 6), round(float(f) - seg_start, 6), int(m)]
            for o, f, m in gt_data.get(sid, [])
            if seg_start <= float(o) < seg_end_gt
        ]

        # Run model
        try:
            onset_np, frame_np = predict_segment(model, flac_path)
        except Exception as e:
            print(f'  Warning: failed on {flac_path}: {e}')
            continue

        pred_rel = extract_notes(onset_np, frame_np, is_last_seg)

        # Save raw frame probabilities before any thresholding/filtering.
        # onset_np : (T_mel, 88) float32 — sigmoid(onset_logit) * P(oct) * P(cls)
        # frame_np : (T_mel, 88) float32 — P(oct) * P(cls)  (no onset factor)
        os.makedirs(ARRAYS_DIR, exist_ok=True)
        np.savez_compressed(
            os.path.join(ARRAYS_DIR, f'{seg_key}.npz'),
            onset=onset_np.astype(np.float32),
            frame=frame_np.astype(np.float32),
        )

        results[seg_key] = {'pred': pred_rel, 'gt': gt_rel}

        # Piano-roll PNG for first N_VIZ segments
        if i < N_VIZ:
            try:
                seg_dur = soundfile.info(flac_path).duration
            except Exception:
                seg_dur = SEGMENT_HOP
            visualize_segment(VIZ_DIR, sid, seg_idx, pred_rel, gt_rel, seg_dur)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    print(f'\nWriting {OUT_JSON} ...')
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    n_pred = sum(len(v['pred']) for v in results.values())
    n_gt   = sum(len(v['gt'])   for v in results.values())
    print(f'Done.  {len(results)} segments  |  {n_pred} pred notes  |  {n_gt} GT notes')


if __name__ == '__main__':
    main()
