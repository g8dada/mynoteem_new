"""
Preprocess MIR-ST500 dataset for NoteEM unaligned supervision training.

Each song is split with a 15-second hop. For each segment:
1. Extracts audio from Vocal.wav → 16kHz mono FLAC with 11 pitch shifts (-5..+5 semitones).
   Audio duration is dynamic: extends just past the last note's real offset (capped at 20 s),
   so no note assigned to this segment is cut off mid-duration.
   Output: NoteEM_audio/mirst_{id}_seg{i}#{shift}/mirst_{id}_seg{i}#{shift}.flac
2. Creates a TSV note list from notes whose JSON onset falls in the primary 15-second window
   [i*15, (i+1)*15).
   Output: NoteEM_tsv/mirst_{id}_seg{i}/mirst_{id}_seg{i}.tsv

Notes are assigned by onset only; audio extends dynamically to capture their real offsets.
No note appears in two segments' TSVs. Segments with no notes are skipped entirely.
Within each segment TSV, notes are assigned uniform 0.5s synthetic slots (unaligned paradigm).

Usage:
    python preprocess_mirst500.py [--limit N] [--workers W]

    --limit N   Process only first N songs (default: all 339). Start small (e.g. 5)
                to test training before committing to the full dataset.
    --workers W Parallel sox workers (default: 4).

After preprocessing, update train.py:
    train_groups = [...]
Copy the list printed at the end of this script.
"""


# python preprocess_mirst500.py [--limit N]

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
data_name = 'mirst500_15sec_5_data'
AUDIO_OUT = f'data/{data_name}_NoteEM_audio'
TSV_OUT = f'data/{data_name}_NoteEM_tsv'

INSTRUMENT = 52   # General MIDI "Choir Aahs" — used for all singing
NOTE_DURATION = 0.5   # synthetic seconds per note within each segment
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


def make_tsv(sid, seg_idx, group, notes_json, start_sec, end_sec, is_last):
    """
    Create TSV for notes whose onset falls in [start_sec, end_sec).
    The last segment uses onset >= start_sec to capture all remaining notes.
    Returns (tsv_path, status, note_count).
    """
    tsv_dir = os.path.join(TSV_OUT, group)
    tsv_path = os.path.join(tsv_dir, f'{group}.tsv')

    if os.path.exists(tsv_path):
        return tsv_path, 'skip', -1

    if is_last:
        notes_in_seg = [r for r in notes_json if r[0] >= start_sec]
    else:
        notes_in_seg = [r for r in notes_json if start_sec <= r[0] < end_sec]

    notes_sorted = sorted(notes_in_seg, key=lambda r: r[0])

    if not notes_sorted:
        return tsv_path, 'empty', 0

    os.makedirs(tsv_dir, exist_ok=True)

    rows = []
    for i, (_, _, midi_note) in enumerate(notes_sorted):
        onset = i * NOTE_DURATION
        offset = onset + NOTE_DURATION - 0.01
        rows.append((onset, offset, float(midi_note), VELOCITY, INSTRUMENT))

    arr = np.array(rows, dtype=float)
    np.savetxt(tsv_path, arr, fmt='%1.6f', delimiter='\t',
               header='onset,offset,note,velocity,instrument')
    return tsv_path, 'ok', len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N songs (default: all)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel sox workers (default: 4)')
    args = parser.parse_args()

    if subprocess.run(['sox', '--version'], capture_output=True).returncode != 0:
        print('ERROR: sox not found. Install with: sudo apt install sox')
        sys.exit(1)

    with open(JSON_PATH) as f:
        data = json.load(f)

    available_ids = sorted(
        [sid for sid in data.keys()
         if os.path.isfile(os.path.join(TRAIN_DIR, sid, 'Vocal.wav'))],
        key=lambda x: int(x)
    )

    if args.limit:
        available_ids = available_ids[:args.limit]
    print(f'Processing {len(available_ids)} songs (of 339 available).')

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

    # --- TSV (fast, run serially) ---
    ok_tsv = skip_tsv = empty_tsv = 0
    valid_groups = []   # ordered list of groups that have a valid TSV
    group_audio_dur = {}  # group -> dynamic audio duration (seconds)

    for sid in tqdm(available_ids, desc='TSVs'):
        segs = song_segments[sid]
        total_dur = song_durations[sid]
        for seg_idx, start in segs:
            tsv_end = start + SEGMENT_HOP   # primary note window boundary
            is_last = (seg_idx == len(segs) - 1)
            group = f'mirst_{sid}_seg{seg_idx}'

            # Dynamic audio duration: extend just past the last note's real offset,
            # but never beyond the hard cap (SEGMENT_WINDOW).
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

            _, status, n_notes = make_tsv(sid, seg_idx, group, data[sid], start, tsv_end, is_last)
            if status == 'empty':
                empty_tsv += 1
                print(f'  SKIP (no notes) {group}  [{start:.1f}s – {tsv_end:.1f}s]')
            elif status == 'skip':
                skip_tsv += 1
                valid_groups.append(group)
            else:
                ok_tsv += 1
                valid_groups.append(group)
                midi_range = [r[2] for r in data[sid]]
                audio_dur_here = group_audio_dur[group]
                print(f'  TSV {group}  {n_notes} notes  [{start:.1f}s – {tsv_end:.1f}s]'
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
