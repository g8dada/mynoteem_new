"""
Preprocess MIR-ST500 for NoteEM training, seeded with ZERO-SHOT predicted labels.

This is a variant of preprocess_mirst500.py. The AUDIO pipeline is identical; only
the LABEL (TSV) pipeline differs:

  * Audio  — same as the original: 15-second hop, dynamic per-segment duration
             (extends just past the last note's real offset, capped at 20 s), 11 pitch
             shifts (-5..+5 semitones), 16 kHz mono FLAC.
             Output: {AUDIO_OUT}/mirst_{id}_seg{i}#{shift}/mirst_{id}_seg{i}#{shift}.flac
             The dynamic duration is reproduced from the GT JSON exactly as the original,
             so each FLAC is byte-for-byte what the zero-shot pass was decoded from — the
             predicted onset/offset times below stay valid against this audio.

  * TSV    — instead of the GT notes with ceiling-quantized, sequentially-placed
             durations (the original "unaligned" 0.5 s-grid labels), each TSV is built
             from the ZERO-SHOT model's PREDICTED notes, which carry REAL
             segment-relative onset/offset times. This gives NoteEM EM a temporally
             aligned starting state instead of the coarse quantized one.
             Predictions are read from test_zeroshot/mirst_zeroshot_results.json
             (the decoded `pred` lists; segment-relative seconds). That file was produced
             by test_mirst_zeroshot.py:extract_notes() over test_zeroshot/arrays/*.npz.
             Output: {TSV_OUT}/mirst_{id}_seg{i}/mirst_{id}_seg{i}.tsv

Scope: the zero-shot pass only covered the songs present in the results JSON
(MIR-ST500 test split, ids 401-500). Only those segments are produced here. Segments
with zero predicted notes are skipped entirely (no audio, no TSV).

Usage:
    python preprocess_mirst500_after_zeroshot.py [--limit N] [--workers W]

    --limit N   Process only first N songs (default: all with zero-shot labels).
    --workers W Parallel sox workers (default: 4).

After preprocessing, update train.py:
    train_groups = [...]
Copy the list printed at the end of this script.
"""


# python preprocess_mirst500_after_zeroshot.py [--limit N]

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile
from tqdm import tqdm

JSON_PATH = '../singing_transcription_ICASSP2021/MIR-ST500_20210206/MIR-ST500_corrected.json'
TRAIN_DIR = '../singing_transcription_ICASSP2021/test'
# Decoded zero-shot predictions: {'mirst_{sid}_seg{i}': {'pred': [[on_s, off_s, midi], ...], 'gt': [...]}}
ZEROSHOT_RESULTS = 'test_zeroshot/mirst_zeroshot_results.json'
data_name = 'mirst500_15sec_data_full_zeroshot'
AUDIO_OUT = f'data/{data_name}_NoteEM_audio'
TSV_OUT = f'data/{data_name}_NoteEM_tsv'

INSTRUMENT = 52   # General MIDI "Choir Aahs" — used for all singing
VELOCITY = 64
SEGMENT_HOP    = 15.0   # hop between segment starts (seconds); also the primary note window
SEGMENT_WINDOW = 20.0   # hard cap on audio duration per segment (seconds)
NOTE_BUFFER    =  0.5   # seconds of audio to keep after the last note's real offset


def get_audio_duration(wav_path):
    return soundfile.info(wav_path).duration


def get_segments(total_duration):
    """Returns list of (seg_idx, start_sec) spaced SEGMENT_HOP apart."""
    n_segs = max(1, int(np.ceil(total_duration / SEGMENT_HOP)))
    return [(i, i * SEGMENT_HOP) for i in range(n_segs)]


def sox_convert_segment(src_wav, dst_flac, start_sec, seg_duration, pitch_cents):
    cmd = ['sox', src_wav, '-r', '16000', '-c', '1', dst_flac,
           'trim', str(start_sec), str(seg_duration)]
    if pitch_cents != 0:
        cmd += ['pitch', str(pitch_cents)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stderr


def process_one_shift(args):
    sid, seg_idx, group, wav_path, start_sec, seg_duration, shift = args
    audio_dir = os.path.join(AUDIO_OUT, f'{group}#{shift}')
    os.makedirs(audio_dir, exist_ok=True)
    flac_path = os.path.join(audio_dir, f'{group}#{shift}.flac')

    if os.path.exists(flac_path) and os.path.getsize(flac_path) > 0:
        try:
            soundfile.read(flac_path, dtype='int16')
            return flac_path, 'skip', None
        except Exception:
            os.remove(flac_path)

    rc, err = sox_convert_segment(wav_path, flac_path, start_sec, seg_duration, shift * 100)
    if rc != 0:
        return flac_path, 'error', err
    return flac_path, 'ok', None


def make_tsv_from_pred(group, pred_notes):
    """
    Create a TSV from the zero-shot predicted notes for this segment.

    pred_notes : [[onset_s, offset_s, midi], ...] in segment-relative seconds
                 (as stored in mirst_zeroshot_results.json). Timing is kept as-is —
                 these are the model's real predicted onset/offset times, NOT quantized.
    Returns (tsv_path, status, note_count).
    """
    tsv_dir = os.path.join(TSV_OUT, group)
    tsv_path = os.path.join(tsv_dir, f'{group}.tsv')

    if os.path.exists(tsv_path):
        return tsv_path, 'skip', -1

    if not pred_notes:
        return tsv_path, 'empty', 0

    os.makedirs(tsv_dir, exist_ok=True)

    rows = []
    for onset, offset, midi in sorted(pred_notes, key=lambda r: r[0]):
        rows.append((float(onset), float(offset), float(midi), VELOCITY, INSTRUMENT))

    arr = np.array(rows, dtype=float)
    np.savetxt(tsv_path, arr, fmt='%1.6f', delimiter='\t',
               header='onset,offset,note,velocity,instrument')
    return tsv_path, 'ok', len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N songs (default: all with zero-shot labels)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel sox workers (default: 4)')
    args = parser.parse_args()

    if subprocess.run(['sox', '--version'], capture_output=True).returncode != 0:
        print('ERROR: sox not found. Install with: sudo apt install sox')
        sys.exit(1)

    with open(JSON_PATH) as f:
        data = json.load(f)

    with open(ZEROSHOT_RESULTS) as f:
        zeroshot = json.load(f)   # 'mirst_{sid}_seg{i}' -> {'pred': [...], 'gt': [...]}

    # Songs that actually have zero-shot predictions AND a Vocal.wav to re-segment.
    zeroshot_sids = {k.split('_')[1] for k in zeroshot.keys()}
    available_ids = sorted(
        [sid for sid in data.keys()
         if sid in zeroshot_sids
         and os.path.isfile(os.path.join(TRAIN_DIR, sid, 'Vocal.wav'))],
        key=lambda x: int(x)
    )

    if args.limit:
        available_ids = available_ids[:args.limit]
    print(f'Processing {len(available_ids)} songs with zero-shot labels.')

    # Compute segments for every song up front (needs audio duration)
    print('Reading audio durations...')
    song_segments = {}
    song_durations = {}
    for sid in tqdm(available_ids, desc='Durations'):
        wav_path = os.path.join(TRAIN_DIR, sid, 'Vocal.wav')
        total_dur = get_audio_duration(wav_path)
        song_durations[sid] = total_dur
        song_segments[sid] = get_segments(total_dur)

    total_segs = sum(len(v) for v in song_segments.values())
    print(f'Total segments: {total_segs} ({total_segs * 11} FLAC files to create). May take a while.\n')

    # --- TSV from zero-shot predictions (fast, run serially) ---
    ok_tsv = skip_tsv = empty_tsv = 0
    valid_groups = []   # ordered list of groups that have a valid (non-empty) TSV
    group_audio_dur = {}  # group -> dynamic audio duration (seconds)

    for sid in tqdm(available_ids, desc='TSVs'):
        segs = song_segments[sid]
        total_dur = song_durations[sid]
        for seg_idx, start in segs:
            tsv_end = start + SEGMENT_HOP   # primary note window boundary
            is_last = (seg_idx == len(segs) - 1)
            group = f'mirst_{sid}_seg{seg_idx}'

            # Dynamic audio duration: reproduce the original GT-based computation exactly,
            # so this audio matches what the zero-shot pass was decoded from.
            if is_last:
                notes_here = [r for r in data[sid] if r[0] >= start]
            else:
                notes_here = [r for r in data[sid] if start <= r[0] < tsv_end]
            if notes_here:
                latest_offset = max(r[1] for r in notes_here)
                audio_end = min(start + SEGMENT_WINDOW,
                                max(tsv_end, latest_offset + NOTE_BUFFER))
            else:
                audio_end = tsv_end  # segment will be skipped; value unused
            group_audio_dur[group] = min(audio_end, total_dur) - start

            pred_notes = zeroshot.get(group, {}).get('pred', [])
            _, status, n_notes = make_tsv_from_pred(group, pred_notes)
            if status == 'empty':
                empty_tsv += 1
                print(f'  SKIP (no pred notes) {group}  [{start:.1f}s – {tsv_end:.1f}s]')
            elif status == 'skip':
                skip_tsv += 1
                valid_groups.append(group)
            else:
                ok_tsv += 1
                valid_groups.append(group)
                midi_range = [r[2] for r in pred_notes]
                audio_dur_here = group_audio_dur[group]
                print(f'  TSV {group}  {n_notes} pred notes  [{start:.1f}s – {tsv_end:.1f}s]'
                      f'  audio={audio_dur_here:.1f}s'
                      f'  MIDI {int(min(midi_range))}-{int(max(midi_range))}')

    print(f'TSVs: {ok_tsv} created, {skip_tsv} already existed, {empty_tsv} empty/skipped.\n')

    # --- Audio (parallelised across shifts, only for valid groups) ---
    valid_group_set = set(valid_groups)
    tasks = []
    for sid in tqdm(available_ids, desc='Audios'):
        wav_path = os.path.join(TRAIN_DIR, sid, 'Vocal.wav')
        for seg_idx, start in song_segments[sid]:
            group = f'mirst_{sid}_seg{seg_idx}'
            if group not in valid_group_set:
                continue
            seg_dur = group_audio_dur[group]
            for shift in range(-5, 6):
                tasks.append((sid, seg_idx, group, wav_path, start, seg_dur, shift))

    ok_audio = skip_audio = err_audio = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one_shift, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Audio'):
            flac_path, status, err = fut.result()
            sid_shift = '/'.join(flac_path.split('/')[-2:])
            if status == 'skip':
                skip_audio += 1
            elif status == 'error':
                print(f'  ERROR {sid_shift}: {err}')
                err_audio += 1
            else:
                ok_audio += 1
                print(f'  OK {sid_shift}')

    print(f'\nAudio: {ok_audio} created, {skip_audio} already existed, {err_audio} errors.')

    print()
    print('TIP: Start with a small subset (--limit 5) to verify training works before processing all songs.')


if __name__ == '__main__':
    main()
