"""Zero-shot evaluation of EffNetb0/AMTAdapter on MIR-ST500.

Runs the pre-trained AST backbone (no EM fine-tuning) on all #0 (unshifted)
segments in the NoteEM audio directory and writes:
  - mirst_zeroshot_results.json  — per-song pred + gt note lists in absolute time
  - mirst_zeroshot_viz/          — piano-roll PNGs for the first N_VIZ segments
  - mirst_zeroshot_viz/arrays/    — raw onset/offset/frame arrays for all segments, onset_np = data['onset'], offset_np = data['offset'], frame_np = data['frame']

Usage:
    python test_mirst_zeroshot.py [--limit N]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import librosa
import mir_eval
import numpy as np
import soundfile
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onsets_and_frames.constants import HOP_LENGTH, MIN_MIDI, N_KEYS, SAMPLE_RATE
from onsets_and_frames.ast_model import EffNetb0
from onsets_and_frames.transcriber import AMTAdapter
from onsets_and_frames.utils import get_peaks

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AUDIO_DIR  = '/data/hakka/mynoteem_new/data/mirst500_15sec_data_full_quantized_NoteEM_audio'
MODEL_PATH = '/data/hakka/singing_transcription_ICASSP2021/AST/models/1005_e_4'
GT_JSON    = '/data/hakka/singing_transcription_ICASSP2021/MIR-ST500_20210206/MIR-ST500_corrected.json'
VIZ_DIR    = '/data/hakka/mynoteem_new/test_zeroshot'
OUT_JSON   = os.path.join(VIZ_DIR, 'mirst_zeroshot_results.json')
ARRAYS_DIR = os.path.join(VIZ_DIR, 'arrays')

SEGMENT_HOP      = 15.0  # seconds between segment starts (matches preprocess_mirst500.py)
ONSET_THRESHOLD  = 0.05  # minimum post-peak onset probability to count as a note
OFFSET_THRESHOLD = 0.5   # offset_sig > this ends a note (matches original _parse_frame_info offset_thres)
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
    """Load one FLAC, run AMTAdapter, return (onset_np, offset_np, frame_np) as (T_mel, 88) arrays."""
    audio_raw, sr = soundfile.read(flac_path, dtype='int16')
    if audio_raw.ndim == 2:
        audio_raw = audio_raw.mean(axis=1).astype(np.int16)

    # Peak-normalize to match training pipeline (audio_dataset.py: librosa.util.normalize)
    audio_np = audio_raw.astype(np.float32) / 32768.0
    peak = np.max(np.abs(audio_np))
    if peak > 0:
        audio_np = audio_np / peak
    audio_44k = librosa.resample(audio_np, orig_sr=SAMPLE_RATE, target_sr=44100)
    cqt_mag = np.abs(librosa.cqt(
        audio_44k, sr=44100, hop_length=1024,
        fmin=librosa.midi_to_hz(36), n_bins=168, bins_per_octave=24,
    ))
    cqt = torch.from_numpy(cqt_mag).float()

    # audio_f is only used for its length (T_mel) when cqt is provided
    audio_f = torch.ShortTensor(audio_raw).float() / 32768.0

    with torch.no_grad():
        onset_pred, offset_pred, _, frame_pred, _ = model(
            audio_f.unsqueeze(0).cuda(),
            cqt=cqt.unsqueeze(0).cuda(),
        )

    onset_np  = onset_pred.squeeze(0).cpu().numpy()   # (T_mel, 88)
    offset_np = offset_pred.squeeze(0).cpu().numpy()  # (T_mel, 88) — same scalar broadcast per frame
    frame_np  = frame_pred.squeeze(0).cpu().numpy()   # (T_mel, 88)
    return onset_np, offset_np, frame_np


def extract_notes(onset_np, offset_np, frame_np, is_last_seg):
    """Convert raw adapter output to a segment-relative note list [onset_s, offset_s, midi].

    Matches the original _parse_frame_info decoding logic:
      1. Global onset signal = max over 88 pitches (≈ sigmoid(onset_logit) weighted by
         the dominant pitch probability) — one onset event per frame (monophonic).
      2. Local-max peak-finding in a ±3-frame window on the global signal.
      3. At each onset peak: start a note and reset the pitch-vote accumulator.
         A new onset also terminates the previous note (onset-as-terminator).
      4. At each offset frame (offset_np[:,0] > OFFSET_THRESHOLD): terminate current note.
      5. Otherwise (sustain): accumulate argmax(frame_np[t]) votes for pitch.
      6. Pitch of each note = majority vote over its sustained frames (argmax at onset frame
         seed + sustain frames), matching the original majority-vote pitch logic.
      7. Notes shorter than MIN_NOTE_FRAMES are discarded.

    For non-last segments clips to onset < SEGMENT_HOP to avoid overlap with the next segment.
    """
    # Global onset signal: max over 88 pitches ≈ original scalar onset probability
    onset_global = onset_np.max(axis=1)                       # (T,)
    peaks_1d = get_peaks(
        torch.from_numpy(onset_global[:, None]), win_size=3   # (T, 1) → (T, 1) bool
    )[:, 0].numpy()                                           # (T,) bool

    onset_active = (onset_global * peaks_1d) > ONSET_THRESHOLD  # (T,) — peak-filtered
    offset_sig   = offset_np[:, 0] > OFFSET_THRESHOLD           # (T,) bool

    T   = onset_np.shape[0]
    fps = SAMPLE_RATE / HOP_LENGTH                               # 31.25 fps
    primary_end = int(SEGMENT_HOP * fps) if not is_last_seg else T

    notes       = []
    cur_onset   = None
    pitch_votes = []   # argmax(frame_np[t]) for each frame in current note

    def _flush(end_frame):
        if cur_onset is None or end_frame - cur_onset < MIN_NOTE_FRAMES or not pitch_votes:
            return
        best = Counter(pitch_votes).most_common(1)[0][0]
        notes.append([round(cur_onset / fps, 6), round(end_frame / fps, 6), best + MIN_MIDI])

    for t in range(primary_end):
        if onset_active[t]:
            _flush(t)                                  # onset terminates previous note
            cur_onset   = t
            pitch_votes = [int(np.argmax(frame_np[t]))]
        elif offset_sig[t]:
            _flush(t)                                  # offset terminates current note
            cur_onset   = None
            pitch_votes = []
        else:
            if cur_onset is not None:                  # sustain: accumulate pitch vote
                pitch_votes.append(int(np.argmax(frame_np[t])))

    _flush(primary_end)   # flush any note open at the segment boundary

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
# Evaluation (mir_eval, matching test_adapter.py conventions)
# ---------------------------------------------------------------------------

ONSET_TOL  = 0.05   # 50 ms
OFFSET_TOL = 0.05   # 50 ms


def _midi_to_hz(midi_list):
    return 440.0 * (2.0 ** ((np.array(midi_list, dtype=float) - 69) / 12.0))


def evaluate_song(pred_notes, gt_notes):
    """Return COn / COnP / COnPOff P/R/F1 for one song (absolute-time notes)."""
    empty = {m: {'p': 0.0, 'r': 0.0, 'f1': 0.0} for m in ('COn', 'COnP', 'COnPOff')}
    if not pred_notes or not gt_notes:
        return empty

    ref_iv = np.array([[o, f] for o, f, _ in gt_notes],   dtype=float)
    ref_hz = _midi_to_hz([m for _, _, m in gt_notes])
    est_iv = np.array([[o, f] for o, f, _ in pred_notes], dtype=float)
    est_hz = _midi_to_hz([m for _, _, m in pred_notes])

    dummy_r = np.full(len(ref_hz), 440.0)
    dummy_e = np.full(len(est_hz), 440.0)

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, dummy_r, est_iv, dummy_e,
        onset_tolerance=ONSET_TOL, pitch_tolerance=50.0,
        offset_ratio=None, offset_min_tolerance=1e6,
    )
    con = {'p': float(p), 'r': float(r), 'f1': float(f)}

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, est_iv, est_hz,
        onset_tolerance=ONSET_TOL, pitch_tolerance=50.0,
        offset_ratio=None, offset_min_tolerance=1e6,
    )
    conp = {'p': float(p), 'r': float(r), 'f1': float(f)}

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, est_iv, est_hz,
        onset_tolerance=ONSET_TOL, pitch_tolerance=50.0,
        offset_ratio=1e-10, offset_min_tolerance=OFFSET_TOL,
    )
    conpoff = {'p': float(p), 'r': float(r), 'f1': float(f)}

    return {'COn': con, 'COnP': conp, 'COnPOff': conpoff}


def aggregate(per_song):
    summary = {}
    for metric in ('COn', 'COnP', 'COnPOff'):
        for sk in ('p', 'r', 'f1'):
            vals = [per_song[sid][metric][sk] for sid in per_song]
            summary[f'{metric}_{sk}'] = {'mean': float(np.mean(vals)),
                                         'std':  float(np.std(vals))}
    return summary


# ---------------------------------------------------------------------------
# Metrics visualisation
# ---------------------------------------------------------------------------

def plot_zeroshot(summary, n_songs, out_path):
    metrics      = ['COn', 'COnP', 'COnPOff']
    score_keys   = ['p',   'r',    'f1']
    score_labels = ['Precision', 'Recall', 'F1']
    colors       = ['#4878CF', '#6ACC65', '#D65F5F']

    n_metrics = len(metrics)
    n_scores  = len(score_keys)
    group_w   = 0.70
    bar_w     = group_w / n_scores
    x         = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f'Zero-Shot EffNetb0 — MIR-ST500 ({n_songs} songs)',
        fontweight='bold', fontsize=13,
    )

    for si, (sk, slabel, color) in enumerate(zip(score_keys, score_labels, colors)):
        offset = (si - n_scores / 2 + 0.5) * bar_w
        vals = [summary[f'{m}_{sk}']['mean'] for m in metrics]
        errs = [summary[f'{m}_{sk}']['std']  for m in metrics]

        bars = ax.bar(
            x + offset, vals, bar_w * 0.88,
            yerr=errs, capsize=3,
            color=color, edgecolor='#444444', linewidth=0.6,
            alpha=0.90, label=slabel,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.012,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_xlabel(f'Metric  (per-song average ± std, n={n_songs} songs)', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend(
        handles=[mpatches.Patch(facecolor=c, edgecolor='#444', label=l)
                 for c, l in zip(colors, score_labels)],
        fontsize=9, loc='upper right', framealpha=0.85,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


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

    results   = {}                  # 'mirst_{sid}_seg{i}' -> {'pred': [...], 'gt': [...]}
    song_pred = defaultdict(list)   # sid -> [(onset_abs, offset_abs, midi), ...]

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
            onset_np, offset_np, frame_np = predict_segment(model, flac_path)
        except Exception as e:
            print(f'  Warning: failed on {flac_path}: {e}')
            continue

        pred_rel = extract_notes(onset_np, offset_np, frame_np, is_last_seg)

        # Save raw model probabilities before any thresholding/filtering.
        # onset_np  : (T_mel, 88) float32 — sigmoid(onset_logit) * P(oct) * P(cls)
        # offset_np : (T_mel, 88) float32 — sigmoid(offset_logit) broadcast to all pitches
        # frame_np  : (T_mel, 88) float32 — P(oct) * P(cls)  (no onset factor)
        os.makedirs(ARRAYS_DIR, exist_ok=True)
        np.savez_compressed(
            os.path.join(ARRAYS_DIR, f'{seg_key}.npz'),
            onset=onset_np.astype(np.float32),
            offset=offset_np.astype(np.float32),
            frame=frame_np.astype(np.float32),
        )

        results[seg_key] = {'pred': pred_rel, 'gt': gt_rel}

        for on_rel, off_rel, midi in pred_rel:
            song_pred[sid].append((round(seg_start + on_rel, 6),
                                   round(seg_start + off_rel, 6),
                                   midi))

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

    # ── Evaluate and plot COn / COnP / COnPOff ────────────────────────────────
    print('\nEvaluating...')
    for sid in song_pred:
        song_pred[sid].sort()

    per_song = {}
    for sid in sorted(song_pred.keys(), key=int):
        if sid not in gt_data:
            continue
        gt_abs = [(float(o), float(f), int(m)) for o, f, m in gt_data[sid]]
        per_song[sid] = evaluate_song(song_pred[sid], gt_abs)

    summary = aggregate(per_song)

    print(f'\n=== Zero-Shot Results ({len(per_song)} songs) ===')
    print(f'{"Metric":<12} {"P":>7} {"R":>7} {"F1":>7}')
    print('-' * 36)
    for metric in ('COn', 'COnP', 'COnPOff'):
        p  = summary[f'{metric}_p']['mean']
        r  = summary[f'{metric}_r']['mean']
        f1 = summary[f'{metric}_f1']['mean']
        print(f'{metric:<12} {p:>7.4f} {r:>7.4f} {f1:>7.4f}')

    out_png = os.path.join(VIZ_DIR, 'metrics.png')
    plot_zeroshot(summary, len(per_song), out_png)


if __name__ == '__main__':
    main()
