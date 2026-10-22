"""Evaluate and compare two AMT models on MIR-ST500 (quantized dataset).

Models compared:
  - NoteEM  : OnsetsAndFrames checkpoint trained via the EM loop (epoch 15)
  - ZeroShot: EffNetb0 (AST) wrapped in AMTAdapter, no fine-tuning (loaded from ZS_CACHE_JSON)

Both models run on the same #0 (unshifted) segments from
mirst500_15sec_data_full_quantized_NoteEM_audio and are evaluated against the
MIR-ST500 corrected ground-truth using COn / COnP / COnPOff F1 at 50 ms.

Outputs:
    <OUT_DIR>/metrics.png   — side-by-side grouped bar chart
    <OUT_DIR>/results.json  — per-song scores + summary for both models
"""

import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import soundfile
import torch
from collections import Counter, defaultdict
from tqdm import tqdm

import mir_eval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onsets_and_frames.constants import HOP_LENGTH, MIN_MIDI, N_KEYS, SAMPLE_RATE
from onsets_and_frames.mel import melspectrogram
from onsets_and_frames.utils import get_peaks

# ─── Paths ────────────────────────────────────────────────────────────────────
AUDIO_DIR        = '/data/hakka/mynoteem_new/data/mirst500_15sec_data_full_quantized_NoteEM_audio'
ZS_CACHE_JSON    = '/data/hakka/mynoteem_new/test_zeroshot/mirst_zeroshot_results.json'
GT_JSON          = '/data/hakka/singing_transcription_ICASSP2021/MIR-ST500_20210206/MIR-ST500_corrected.json'
EPOCH = 1

run_dir = os.environ.get('RUN_DIR', 'transcriber-261013-161009-adapter_False')
NOTEM_MODEL_PATH = os.environ.get(
    'NOTEM_MODEL_PATH',
    f'/data/hakka/mynoteem_new/runs/{run_dir}/transcriber_{EPOCH}.pt')   # epoch 15 checkpoint
OUT_DIR          = f'/data/hakka/mynoteem_new/runs/{run_dir}/test_results_epoch_{EPOCH}'

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
SEGMENT_HOP     = 15.0   # seconds (must match preprocess_mirst500.py)
MIN_NOTE_FRAMES = 3      # ~96 ms at 31.25 fps

# NoteEM onset/offset are true per-pitch sigmoids reaching ~1.0, so they need a much
# higher threshold than the AMTAdapter product formula (which is capped ~0.2).
NOTEM_ONSET_THR  = 0.5
NOTEM_OFFSET_THR = 0.5

ONSET_TOL  = 0.05   # 50 ms
OFFSET_TOL = 0.05   # 50 ms


# ─── Model loading ────────────────────────────────────────────────────────────

def load_notem_model():
    model = torch.load(NOTEM_MODEL_PATH, map_location='cpu', weights_only=False)
    return model.cuda().eval()


def load_zs_from_cache(path):
    """Load pre-extracted zero-shot notes (monophonic decode) from ZS_CACHE_JSON.

    The cache is keyed 'mirst_{sid}_seg{i}' with segment-relative notes in
    entry['pred'] = [[on_rel, off_rel, midi], ...]; we offset them back to
    absolute song time and group by song id.
    """
    with open(path) as f:
        data = json.load(f)
    zs_pred = defaultdict(list)
    for seg_key, entry in data.items():           # 'mirst_{sid}_seg{i}'
        m = re.match(r'^mirst_(\d+)_seg(\d+)$', seg_key)
        if m is None:
            continue
        sid    = m.group(1)
        offset = int(m.group(2)) * SEGMENT_HOP
        for on_rel, off_rel, midi in entry.get('pred', []):
            zs_pred[sid].append((round(on_rel + offset, 6),
                                 round(off_rel + offset, 6),
                                 midi))
    for sid in zs_pred:
        zs_pred[sid].sort()
    return zs_pred


# ─── Segment discovery ────────────────────────────────────────────────────────

_SEG_RE = re.compile(r'^mirst_(\d+)_seg(\d+)#0$')


def collect_segments(audio_dir):
    segs = []
    for entry in os.scandir(audio_dir):
        m = _SEG_RE.match(entry.name)
        if m is None:
            continue
        sid     = m.group(1)
        seg_idx = int(m.group(2))
        flac    = os.path.join(entry.path, f'{entry.name}.flac')
        if os.path.isfile(flac):
            segs.append((sid, seg_idx, flac))
    segs.sort(key=lambda x: (int(x[0]), x[1]))
    return segs


# ─── Inference ────────────────────────────────────────────────────────────────

def _load_audio(flac_path):
    audio_raw, _ = soundfile.read(flac_path, dtype='int16')
    if audio_raw.ndim == 2:
        audio_raw = audio_raw.mean(axis=1).astype(np.int16)
    return audio_raw


def predict_notem(model, flac_path):
    """OnsetsAndFrames: mel → forward → (onset_np, offset_np, frame_np) as (T, 88)."""
    audio_raw = _load_audio(flac_path)
    audio_t = torch.ShortTensor(audio_raw).float() / 32768.0
    with torch.no_grad():
        mel = melspectrogram(audio_t.unsqueeze(0).cuda()[:, :-1]).transpose(-1, -2)
        onset_pred, offset_pred, _, frame_pred, _ = model(mel)
    onset_np  = onset_pred.squeeze(0).cpu().numpy()
    offset_np = offset_pred.squeeze(0).cpu().numpy()
    frame_np  = frame_pred.squeeze(0).cpu().numpy()
    # n_instruments > 1 → onset has output_features * n_instruments cols; take the
    # pitch-only head (last N_KEYS, matching train.py's `-N_KEYS:` metric slice).
    if onset_np.shape[-1] > N_KEYS:
        onset_np = onset_np[:, -N_KEYS:]
    if offset_np.shape[-1] > N_KEYS:
        offset_np = offset_np[:, -N_KEYS:]
    if frame_np.shape[-1] > N_KEYS:
        frame_np = frame_np[:, -N_KEYS:]
    return onset_np, offset_np, frame_np


def extract_notes(onset_np, offset_np, frame_np, is_last_seg, onset_thr, offset_thr):
    """Monophonic decoding matching test_adapter.py / test_mirst_zeroshot.py:
    global onset peak-detection, per-frame argmax pitch with majority vote, offset termination.

    Unlike the AMTAdapter (which broadcasts one scalar offset across all 88 keys), the
    OnsetsAndFrames offset head is genuinely per-pitch, so the global offset signal is the
    max over pitches rather than offset_np[:, 0].
    """
    onset_global = onset_np.max(axis=1)                              # (T,)
    peaks_1d = get_peaks(
        torch.from_numpy(onset_global[:, None]), win_size=3
    )[:, 0].numpy()                                                  # (T,) bool
    onset_active = (onset_global * peaks_1d) > onset_thr            # (T,)
    offset_sig   = offset_np.max(axis=1) > offset_thr              # (T,)

    T   = onset_np.shape[0]
    fps = SAMPLE_RATE / HOP_LENGTH
    primary_end = int(SEGMENT_HOP * fps) if not is_last_seg else T

    notes       = []
    cur_onset   = None
    pitch_votes = []

    def _flush(end_frame):
        if cur_onset is None or end_frame - cur_onset < MIN_NOTE_FRAMES or not pitch_votes:
            return
        best = Counter(pitch_votes).most_common(1)[0][0]
        notes.append((round(cur_onset / fps, 6), round(end_frame / fps, 6), best + MIN_MIDI))

    for t in range(primary_end):
        if onset_active[t]:
            _flush(t)
            cur_onset   = t
            pitch_votes = [int(np.argmax(frame_np[t]))]
        elif offset_sig[t]:
            _flush(t)
            cur_onset   = None
            pitch_votes = []
        else:
            if cur_onset is not None:
                pitch_votes.append(int(np.argmax(frame_np[t])))

    _flush(primary_end)
    notes.sort()
    return notes


# ─── Inference loop ───────────────────────────────────────────────────────────

def run_inference(model, onset_thr, offset_thr, segments, last_seg_of, label):
    song_pred = defaultdict(list)
    for sid, seg_idx, flac_path in tqdm(segments, desc=f'Inference [{label}]'):
        seg_start = seg_idx * SEGMENT_HOP
        is_last   = (seg_idx == last_seg_of[sid])
        try:
            onset_np, offset_np, frame_np = predict_notem(model, flac_path)
        except Exception as e:
            print(f'  Warning – skipping {flac_path}: {e}')
            continue
        for on_rel, off_rel, midi in extract_notes(onset_np, offset_np, frame_np, is_last, onset_thr, offset_thr):
            song_pred[sid].append((
                round(seg_start + on_rel,  6),
                round(seg_start + off_rel, 6),
                midi,
            ))
    for sid in song_pred:
        song_pred[sid].sort()
    return song_pred


# ─── Evaluation ───────────────────────────────────────────────────────────────

def midi_to_hz(midi_list):
    return 440.0 * (2.0 ** ((np.array(midi_list, dtype=float) - 69) / 12.0))


def evaluate_song(pred_notes, gt_notes):
    """Return COn / COnP / COnPOff P/R/F1 for one song."""
    empty = {m: {'p': 0.0, 'r': 0.0, 'f1': 0.0} for m in ('COn', 'COnP', 'COnPOff')}
    if not pred_notes or not gt_notes:
        return empty

    ref_iv  = np.array([[o, f] for o, f, _ in gt_notes],   dtype=float)
    ref_hz  = midi_to_hz([m for _, _, m in gt_notes])
    est_iv  = np.array([[o, f] for o, f, _ in pred_notes], dtype=float)
    est_hz  = midi_to_hz([m for _, _, m in pred_notes])

    # COn: onset only — dummy equal pitches, enormous offset tolerance
    dummy = np.full(len(ref_hz), 440.0)
    dummy_e = np.full(len(est_hz), 440.0)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, dummy, est_iv, dummy_e,
        onset_tolerance=ONSET_TOL, pitch_tolerance=50.0,
        offset_ratio=None, offset_min_tolerance=1e6,
    )
    con = {'p': float(p), 'r': float(r), 'f1': float(f)}

    # COnP: onset + pitch, offset ignored
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, est_iv, est_hz,
        onset_tolerance=ONSET_TOL, pitch_tolerance=50.0,
        offset_ratio=None, offset_min_tolerance=1e6,
    )
    conp = {'p': float(p), 'r': float(r), 'f1': float(f)}

    # COnPOff: onset + pitch + offset (strict 50 ms).
    # offset_ratio=None silently ignores offsets; offset_ratio=1e-10 collapses
    # to max(1e-10 * duration, 0.05) = 0.05 s for any real note duration.
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, est_iv, est_hz,
        onset_tolerance=ONSET_TOL, pitch_tolerance=50.0,
        offset_ratio=1e-10, offset_min_tolerance=OFFSET_TOL,
    )
    conpoff = {'p': float(p), 'r': float(r), 'f1': float(f)}

    return {'COn': con, 'COnP': conp, 'COnPOff': conpoff}


def evaluate_all(song_pred, gt_data):
    per_song = {}
    for sid in sorted(song_pred.keys(), key=int):
        if sid not in gt_data:
            continue
        gt = [(float(o), float(f), int(m)) for o, f, m in gt_data[sid]]
        per_song[sid] = evaluate_song(song_pred[sid], gt)
    return per_song


def aggregate(per_song):
    summary = {}
    for metric in ('COn', 'COnP', 'COnPOff'):
        for sk in ('p', 'r', 'f1'):
            vals = [per_song[sid][metric][sk] for sid in per_song]
            summary[f'{metric}_{sk}'] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return summary


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_comparison(summary_notem, summary_zs, n_songs, out_path):
    """Grouped bar chart: 3 metrics × 3 scores × 2 models."""
    metrics    = ['COn', 'COnP', 'COnPOff']
    score_keys = ['p',   'r',    'f1']
    score_labels = ['Precision', 'Recall', 'F1']

    # Two model colours per score type (solid = NoteEM, hatched = ZeroShot)
    base_colors = ['#4878CF', '#6ACC65', '#D65F5F']
    light_colors = ['#9BBEE8', '#ABDFAA', '#EFA3A3']

    n_metrics = len(metrics)
    n_scores  = len(score_keys)
    n_models  = 2
    group_w   = 0.80
    bar_w     = group_w / (n_scores * n_models)

    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        f'NoteEM (epoch {EPOCH}) vs Zero-Shot EffNetb0 — MIR-ST500 ({n_songs} songs)',
        fontweight='bold', fontsize=13,
    )

    for si, (sk, slabel, base_c, light_c) in enumerate(
            zip(score_keys, score_labels, base_colors, light_colors)):

        for mi, (model_summary, hatch, color, model_tag) in enumerate(
                zip([summary_notem, summary_zs],
                    ['', '//'],
                    [base_c, light_c],
                    ['NoteEM', 'ZeroShot'])):

            # Bar centres: pair score types together, then space between models
            col_idx  = si * n_models + mi
            n_cols   = n_scores * n_models
            offset   = (col_idx - n_cols / 2 + 0.5) * bar_w

            vals = [model_summary[f'{m}_{sk}']['mean'] for m in metrics]
            errs = [model_summary[f'{m}_{sk}']['std']  for m in metrics]

            bars = ax.bar(
                x + offset, vals, bar_w * 0.88,
                yerr=errs, capsize=3,
                color=color, hatch=hatch,
                edgecolor='#444444', linewidth=0.6,
                alpha=0.90,
                label=f'{model_tag} {slabel}',
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.012,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_xlabel(f'Metric  (per-song average ± std, n={n_songs} songs)', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    # Compact legend: one row per score type, two patches each
    legend_handles = []
    for slabel, base_c, light_c in zip(score_labels, base_colors, light_colors):
        legend_handles.append(mpatches.Patch(facecolor=base_c,  edgecolor='#444', label=f'NoteEM {slabel}'))
        legend_handles.append(mpatches.Patch(facecolor=light_c, edgecolor='#444', hatch='//', label=f'ZeroShot {slabel}'))

    ax.legend(handles=legend_handles, ncol=3, fontsize=8.5,
              loc='upper right', framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N segments (quick smoke-test)')
    parser.add_argument('--zs-json', default=ZS_CACHE_JSON,
                        help='Path to mirst_zeroshot_results.json (default: ZS_CACHE_JSON)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading GT JSON...')
    with open(GT_JSON) as f:
        gt_data = json.load(f)

    print('Scanning audio directory...')
    segments = collect_segments(AUDIO_DIR)
    if args.limit:
        segments = segments[:args.limit]
    print(f'Found {len(segments)} #0 segments.')

    last_seg_of = {}
    for sid, seg_idx, _ in segments:
        last_seg_of[sid] = max(last_seg_of.get(sid, 0), seg_idx)

    # ── NoteEM inference ───────────────────────────────────────────────────────
    print('\nLoading NoteEM model...')
    notem_model = load_notem_model()
    notem_pred  = run_inference(
        notem_model,
        NOTEM_ONSET_THR, NOTEM_OFFSET_THR,
        segments, last_seg_of, 'NoteEM',
    )
    del notem_model
    torch.cuda.empty_cache()

    # ── Zero-shot predictions (pre-extracted, loaded from cache) ─────────────────
    print(f'\nLoading ZS predictions from {args.zs_json} ...')
    zs_pred = load_zs_from_cache(args.zs_json)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print('\nEvaluating NoteEM...')
    notem_per_song = evaluate_all(notem_pred, gt_data)
    print(f'  Evaluated {len(notem_per_song)} songs.')

    print('Evaluating ZeroShot...')
    zs_per_song = evaluate_all(zs_pred, gt_data)
    print(f'  Evaluated {len(zs_per_song)} songs.')

    # Use the intersection of songs evaluated by both models for a fair common set
    common_sids = sorted(set(notem_per_song) & set(zs_per_song), key=int)
    notem_common = {s: notem_per_song[s] for s in common_sids}
    zs_common    = {s: zs_per_song[s]    for s in common_sids}
    notem_summary_common = aggregate(notem_common)
    zs_summary_common    = aggregate(zs_common)

    # ── Print ──────────────────────────────────────────────────────────────────
    print(f'\n=== Results ({len(common_sids)} songs in common) ===')
    print(f'{"Metric":<12} {"Model":<12} {"P":>7} {"R":>7} {"F1":>7}')
    print('-' * 48)
    for metric in ('COn', 'COnP', 'COnPOff'):
        for tag, summ in (('NoteEM', notem_summary_common), ('ZeroShot', zs_summary_common)):
            p  = summ[f'{metric}_p']['mean']
            r  = summ[f'{metric}_r']['mean']
            f1 = summ[f'{metric}_f1']['mean']
            print(f'{metric:<12} {tag:<12} {p:>7.4f} {r:>7.4f} {f1:>7.4f}')
        print()

    # ── Save JSON ──────────────────────────────────────────────────────────────
    # Store per-song zero-shot pred notes alongside their scores
    zs_pred_serialisable = {
        sid: {'pred_notes': list(map(list, notes))}
        for sid, notes in zs_pred.items()
    }

    out_json = os.path.join(OUT_DIR, 'results.json')
    with open(out_json, 'w') as f:
        json.dump({
            'n_songs_common': len(common_sids),
            'notem_summary':    notem_summary_common,
            'zeroshot_summary': zs_summary_common,
            'notem_per_song':   notem_per_song,
            'zeroshot_per_song': {
                sid: {**zs_pred_serialisable.get(sid, {}), **zs_per_song.get(sid, {})}
                for sid in zs_per_song
            },
        }, f, indent=2)
    print(f'Saved: {out_json}')

    # ── Save PNG ───────────────────────────────────────────────────────────────
    out_png = os.path.join(OUT_DIR, 'metrics.png')
    plot_comparison(notem_summary_common, zs_summary_common, len(common_sids), out_png)


if __name__ == '__main__':
    main()
