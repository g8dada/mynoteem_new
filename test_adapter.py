"""Evaluate and compare two AMT models on MIR-ST500 (quantized dataset).

Models compared:
  - Adapter  : AMTAdapter (EffNetb0 fine-tuned via NoteEM EM loop), epoch 15
  - ZeroShot : EffNetb0 (AST) wrapped in AMTAdapter, no fine-tuning

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
from collections import defaultdict
from tqdm import tqdm

import mir_eval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onsets_and_frames.constants import HOP_LENGTH, MIN_MIDI, N_KEYS, SAMPLE_RATE
from onsets_and_frames.dataset import _compute_cqt
from onsets_and_frames.utils import get_peaks

# ─── Paths ────────────────────────────────────────────────────────────────────
AUDIO_DIR          = '/data/hakka/mynoteem_new/data/mirst500_15sec_data_full_quantized_NoteEM_audio'
ZS_MODEL_PATH      = '/data/hakka/singing_transcription_ICASSP2021/AST/models/1005_e_4'
ZS_CACHE_JSON      = '/data/hakka/mynoteem_new/mirst_zeroshot_viz/mirst_zeroshot_results.json'
GT_JSON            = '/data/hakka/singing_transcription_ICASSP2021/MIR-ST500_20210206/MIR-ST500_corrected.json'

run_dir = 'transcriber-260908-192658'
ADAPTER_MODEL_PATH = f'/data/hakka/mynoteem_new/runs/{run_dir}/transcriber_15.pt'
OUT_DIR            = f'/data/hakka/mynoteem_new/runs/{run_dir}/test_results'

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
SEGMENT_HOP     = 15.0   # seconds (must match preprocess_mirst500.py)
MIN_NOTE_FRAMES = 3      # ~96 ms at 31.25 fps

# Both adapter and zero-shot use AMTAdapter with the same low thresholds
ONSET_THR  = 0.05
FRAME_THR  = 0.05

ONSET_TOL  = 0.05   # 50 ms
OFFSET_TOL = 0.05   # 50 ms


# ─── Model loading ────────────────────────────────────────────────────────────

def load_adapter_model():
    """Load the EM-fine-tuned AMTAdapter checkpoint directly."""
    model = torch.load(ADAPTER_MODEL_PATH, map_location='cpu', weights_only=False)
    return model.cuda().eval()


def load_zeroshot_model():
    from onsets_and_frames.ast_model import EffNetb0
    from onsets_and_frames.transcriber import AMTAdapter
    backbone = EffNetb0()
    backbone.load_state_dict(
        torch.load(ZS_MODEL_PATH, map_location='cpu', weights_only=False),
        strict=False,
    )
    return AMTAdapter(backbone).cuda().eval()


def load_zs_from_cache(path):
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


def predict_amtadapter(model, flac_path):
    """Shared inference for any AMTAdapter model: CQT + audio → (onset_np, frame_np)."""
    audio_raw   = _load_audio(flac_path)
    audio_short = torch.ShortTensor(audio_raw)
    cqt         = _compute_cqt(audio_short)
    audio_f     = audio_short.float() / 32768.0
    with torch.no_grad():
        onset_pred, _, _, frame_pred, _ = model(
            audio_f.unsqueeze(0).cuda(),
            cqt=cqt.unsqueeze(0).cuda(),
        )
    onset_np = onset_pred.squeeze(0).cpu().numpy()   # (T, 88)
    frame_np = frame_pred.squeeze(0).cpu().numpy()
    return onset_np, frame_np


def extract_notes(onset_np, frame_np, is_last_seg):
    """Per-pitch scan; jumps past each note to avoid mid-note re-triggers."""
    onset_t   = torch.from_numpy(onset_np)
    peaks     = get_peaks(onset_t, win_size=3)
    onset_pkd = onset_np.copy()
    onset_pkd[~peaks.numpy()] = 0.0

    onsets_bin = (onset_pkd > ONSET_THR).astype(np.uint8)
    frames_bin = (frame_np  > FRAME_THR).astype(np.uint8)

    T, P        = onsets_bin.shape
    fps         = SAMPLE_RATE / HOP_LENGTH
    primary_end = int(SEGMENT_HOP * fps) if not is_last_seg else T

    notes = []
    for pitch in range(P):
        t = 0
        while t < primary_end:
            if not onsets_bin[t, pitch]:
                t += 1
                continue
            f_on = t; f_off = t
            while f_off < T and (onsets_bin[f_off, pitch] or frames_bin[f_off, pitch]):
                f_off += 1
            if f_off - f_on >= MIN_NOTE_FRAMES:
                notes.append((round(f_on / fps, 6), round(f_off / fps, 6), pitch + MIN_MIDI))
            t = f_off if f_off > t else t + 1

    notes.sort()
    return notes


# ─── Inference loop ───────────────────────────────────────────────────────────

def run_inference(model, segments, last_seg_of, label):
    song_pred = defaultdict(list)
    for sid, seg_idx, flac_path in tqdm(segments, desc=f'Inference [{label}]'):
        seg_start = seg_idx * SEGMENT_HOP
        is_last   = (seg_idx == last_seg_of[sid])
        try:
            onset_np, frame_np = predict_amtadapter(model, flac_path)
        except Exception as e:
            print(f'  Warning – skipping {flac_path}: {e}')
            continue
        for on_rel, off_rel, midi in extract_notes(onset_np, frame_np, is_last):
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

    # COn: onset only — dummy equal pitches, offset ignored
    dummy_r = np.full(len(ref_hz), 440.0)
    dummy_e = np.full(len(est_hz), 440.0)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, dummy_r, est_iv, dummy_e,
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

    # COnPOff: onset + pitch + strict 50 ms offset.
    # offset_ratio=None ignores offsets entirely; offset_ratio=1e-10 collapses
    # to max(1e-10 * duration, 0.05) = 0.05 s for any real note.
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

def plot_comparison(summary_adapter, summary_zs, n_songs, out_path):
    metrics      = ['COn', 'COnP', 'COnPOff']
    score_keys   = ['p',   'r',    'f1']
    score_labels = ['Precision', 'Recall', 'F1']

    base_colors  = ['#4878CF', '#6ACC65', '#D65F5F']
    light_colors = ['#9BBEE8', '#ABDFAA', '#EFA3A3']

    n_metrics = len(metrics)
    n_scores  = len(score_keys)
    n_models  = 2
    group_w   = 0.80
    bar_w     = group_w / (n_scores * n_models)

    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        f'Adapter EM (epoch 15) vs Zero-Shot EffNetb0 — MIR-ST500 ({n_songs} songs)',
        fontweight='bold', fontsize=13,
    )

    for si, (sk, slabel, base_c, light_c) in enumerate(
            zip(score_keys, score_labels, base_colors, light_colors)):
        for mi, (model_summary, hatch, color, model_tag) in enumerate(
                zip([summary_adapter, summary_zs],
                    ['', '//'],
                    [base_c, light_c],
                    ['Adapter', 'ZeroShot'])):

            col_idx = si * n_models + mi
            n_cols  = n_scores * n_models
            offset  = (col_idx - n_cols / 2 + 0.5) * bar_w

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

    legend_handles = []
    for slabel, base_c, light_c in zip(score_labels, base_colors, light_colors):
        legend_handles.append(mpatches.Patch(facecolor=base_c,  edgecolor='#444', label=f'Adapter {slabel}'))
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
    parser.add_argument('--skip-zeroshot', action='store_true',
                        help='Re-use cached zeroshot results from JSON if present')
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

    # ── Adapter inference ──────────────────────────────────────────────────────
    print('\nLoading Adapter model...')
    adapter_model = load_adapter_model()
    adapter_pred  = run_inference(adapter_model, segments, last_seg_of, 'Adapter')
    del adapter_model
    torch.cuda.empty_cache()

    # ── Zero-shot inference ────────────────────────────────────────────────────
    cached_json = os.path.join(OUT_DIR, 'results.json')
    zs_pred = None

    if os.path.isfile(args.zs_json):
        print(f'\nLoading ZS predictions from {args.zs_json} ...')
        zs_pred = load_zs_from_cache(args.zs_json)
    elif args.skip_zeroshot and os.path.isfile(cached_json):
        print('\nLoading cached zero-shot predictions from results.json ...')
        with open(cached_json) as f:
            prev = json.load(f)
        if 'zeroshot_per_song' in prev:
            zs_pred = defaultdict(list)
            for sid, entry in prev['zeroshot_per_song'].items():
                for n in entry.get('pred_notes', []):
                    zs_pred[sid].append(tuple(n))

    if zs_pred is None:
        print('\nLoading ZeroShot model...')
        zs_model = load_zeroshot_model()
        zs_pred  = run_inference(zs_model, segments, last_seg_of, 'ZeroShot')
        del zs_model
        torch.cuda.empty_cache()

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print('\nEvaluating Adapter...')
    adapter_per_song = evaluate_all(adapter_pred, gt_data)
    print(f'  Evaluated {len(adapter_per_song)} songs.')

    print('Evaluating ZeroShot...')
    zs_per_song = evaluate_all(zs_pred, gt_data)
    print(f'  Evaluated {len(zs_per_song)} songs.')

    common_sids      = sorted(set(adapter_per_song) & set(zs_per_song), key=int)
    adapter_common   = {s: adapter_per_song[s] for s in common_sids}
    zs_common        = {s: zs_per_song[s]      for s in common_sids}
    adapter_summary  = aggregate(adapter_common)
    zs_summary       = aggregate(zs_common)

    # ── Print ──────────────────────────────────────────────────────────────────
    print(f'\n=== Results ({len(common_sids)} songs in common) ===')
    print(f'{"Metric":<12} {"Model":<12} {"P":>7} {"R":>7} {"F1":>7}')
    print('-' * 48)
    for metric in ('COn', 'COnP', 'COnPOff'):
        for tag, summ in (('Adapter', adapter_summary), ('ZeroShot', zs_summary)):
            p  = summ[f'{metric}_p']['mean']
            r  = summ[f'{metric}_r']['mean']
            f1 = summ[f'{metric}_f1']['mean']
            print(f'{metric:<12} {tag:<12} {p:>7.4f} {r:>7.4f} {f1:>7.4f}')
        print()

    # ── Save JSON ──────────────────────────────────────────────────────────────
    zs_pred_serialisable = {
        sid: {'pred_notes': list(map(list, notes))}
        for sid, notes in zs_pred.items()
    }

    out_json = os.path.join(OUT_DIR, 'results.json')
    with open(out_json, 'w') as f:
        json.dump({
            'n_songs_common': len(common_sids),
            'adapter_summary':  adapter_summary,
            'zeroshot_summary': zs_summary,
            'adapter_per_song': adapter_per_song,
            'zeroshot_per_song': {
                sid: {**zs_pred_serialisable.get(sid, {}), **zs_per_song.get(sid, {})}
                for sid in zs_per_song
            },
        }, f, indent=2)
    print(f'Saved: {out_json}')

    # ── Save PNG ───────────────────────────────────────────────────────────────
    out_png = os.path.join(OUT_DIR, 'metrics.png')
    plot_comparison(adapter_summary, zs_summary, len(common_sids), out_png)


if __name__ == '__main__':
    main()
