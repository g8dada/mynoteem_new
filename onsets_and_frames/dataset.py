# This software is for non-commercial use only.
# Commercial use requires a separate license.
# This software is covered by US Patent Application No. 63/312,219.

from .constants import *
import numpy as np
from dtw import *
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import os
import json
from onsets_and_frames.mel import melspectrogram
from datetime import datetime
from onsets_and_frames.midi_utils import *
from onsets_and_frames.utils import *
import time
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _parse_seg_key(flac):
    """Extract (sid_str, seg_idx, seg_key) from a flac path like .../mirst_401_seg3#0/mirst_401_seg3#0.flac."""
    basename = os.path.splitext(os.path.basename(flac))[0]  # mirst_401_seg3#0
    seg_key = basename.rsplit('#', 1)[0]                     # mirst_401_seg3
    parts = seg_key.split('_')                               # ['mirst', '401', 'seg3']
    sid_str = parts[1]
    seg_idx = int(parts[2][3:])                              # strip 'seg'
    return sid_str, seg_idx, seg_key


def _get_gt_notes_for_segment(gt_data, sid_str, seg_idx):
    """Return GT notes as (onset_s, offset_s, midi_int) relative to segment start."""
    seg_start = seg_idx * 15.0
    seg_end = (seg_idx + 1) * 15.0
    notes = []
    for note in gt_data.get(sid_str, []):
        onset, offset, midi = float(note[0]), float(note[1]), int(note[2])
        if seg_start <= onset < seg_end:
            notes.append((onset - seg_start, offset - seg_start, midi))
    return notes


def _visualize_notes(viz_dir, seg_key, history):
    """
    Piano-roll PNG, one row per epoch, stacked vertically. Overwritten each epoch.
    Blue bars: GT notes (MIR-ST500). Red ticks: DTW-aligned pseudo-label onsets.
    history: {epoch_tag: {'pred': [(t_sec, midi)], 'gt': [(onset_s, offset_s, midi)]}}
    """
    os.makedirs(viz_dir, exist_ok=True)
    epochs = sorted(history.keys())
    n = len(epochs)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(2.5 * n, 2.5)), squeeze=False)
    fig.suptitle(f'Pseudo-labels vs GT: {seg_key}', fontweight='bold', y=1.01)

    for row, tag in enumerate(epochs):
        ax = axes[row, 0]
        entry = history[tag]

        for (onset_s, offset_s, midi) in entry['gt']:
            ax.barh(midi, max(offset_s - onset_s, 0.05), left=onset_s,
                    height=0.7, fill=False, edgecolor='#1f77b4', linewidth=1.0)

        for (onset_s, offset_s, midi) in entry['pred']:
            ax.barh(midi, max(offset_s - onset_s, 0.05), left=onset_s,
                    height=0.7, color='#ff7f0e')

        ax.set_xlim(0, 15)
        ax.set_ylim(MIN_MIDI - 1, MAX_MIDI + 1)
        ax.set_ylabel(tag, fontsize=7)
        ax.tick_params(labelsize=6)
        if row < n - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('time (s)', fontsize=8)

        if row == 0:
            ax.legend(handles=[
                Patch(facecolor='none', edgecolor='#1f77b4', linewidth=1.0, label='GT (MIR-ST500)'),
                Patch(facecolor='#ff7f0e', label='pseudo-label'),
            ], loc='upper right', fontsize=7)

    plt.tight_layout()
    out = os.path.join(viz_dir, f'{seg_key}.png')
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved viz: {out}')


def _compute_cqt(audio_short):
    """Resample ShortTensor audio (16 kHz) and compute CQT for AMTAdapter caching."""
    audio_np = audio_short.float().numpy() / 32768.0
    audio_44k = librosa.resample(audio_np, orig_sr=SAMPLE_RATE, target_sr=44100)
    fmin = librosa.midi_to_hz(36)  # MIDI_LOW = 36
    cqt_mag = np.abs(librosa.cqt(
        audio_44k, sr=44100, hop_length=1024,
        fmin=fmin, n_bins=168, bins_per_octave=24,
    ))
    return torch.from_numpy(cqt_mag).float()  # (168, T_cqt)


# CQT frame-rate conversion factor: (44100 Hz / 1024 hop) / 16000 Hz
_CQT_RATIO = 44100 / (SAMPLE_RATE * 1024)  # ≈ 0.002693 CQT frames per 16 kHz sample


class EMDATASET(Dataset):
    def __init__(self,
                 audio_path='NoteEM_audio',
                 labels_path='NoteEm_labels',
                 groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE,
                 instrument_map=None, update_instruments=False, transcriber=None,
                 conversion_map=None):
        self.audio_path = audio_path
        self.labels_path = labels_path
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.groups = groups
        self.conversion_map = conversion_map
        self.file_list = self.files(self.groups)
        if instrument_map is None:
            self.get_instruments()
        else:
            self.instruments = instrument_map
            if update_instruments:
                self.add_instruments()
        self.transcriber = transcriber
        self._notes_history = {}
        self.load_pts(self.file_list)
        self.data = []
        print('Reading files...')
        for input_files in tqdm(self.file_list):
            data = self.pts[input_files[0]]
            audio_len = len(data['audio'])
            minutes = audio_len // (SAMPLE_RATE * 60)
            copies = max(1, minutes)
            for _ in range(copies):
                self.data.append(input_files)
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def files(self, groups):
        self.path = self.audio_path
        tsvs_path = self.labels_path
        res = []
        good_ids = list(range(2075, 2084))
        # good_ids += list(range(1817, 1820))
        # good_ids += list(range(2202, 2205))
        # good_ids += list(range(2415, 2418))
        # good_ids += list(range(2504, 2508))
        for group in groups:
            tsvs = os.listdir(tsvs_path + '/' + group)
            tsvs = sorted(tsvs)
            for shft in range(-5, 6):
                curr_fls_pth = self.path + '/' + group + '#{}'.format(shft)
                fls = os.listdir(curr_fls_pth)
                fls = sorted(fls)
                for f, t in zip(fls, tsvs):
                    # #### MusicNet
                    if 'MusicNet' in group:
                        if all([str(elem) not in f for elem in good_ids]):
                            continue
                    res.append((curr_fls_pth + '/' + f, tsvs_path + '/' + group + '/' + t))
        return res

    def get_instruments(self):
        instruments = set()
        for _, f in self.file_list:
            print('loading midi from', f)
            events = np.loadtxt(f, delimiter='\t', skiprows=1, ndmin=2)
            curr_instruments = set(events[:, -1])
            instruments = instruments.union(curr_instruments)
        instruments = [int(elem) for elem in instruments if elem < 115]
        instruments = list(set(instruments))
        if 0 in instruments:
            piano_ind = instruments.index(0)
            instruments.pop(piano_ind)
            instruments.insert(0, 0)
        self.instruments = instruments
        self.instruments = list(set(self.instruments) - set(range(88, 104)) - set(range(112, 150)))
        print('Dataset instruments:', self.instruments)
        print('Total:', len(self.instruments), 'instruments')

    def add_instruments(self):
        for _, f in self.file_list:
            events = np.loadtxt(f, delimiter='\t', skiprows=1, ndmin=2)
            curr_instruments = set(events[:, -1])
            new_instruments = curr_instruments - set(self.instruments)
            self.instruments += list(new_instruments)
        instruments = [int(elem) for elem in self.instruments if (elem < 115)]
        self.instruments = instruments

    def __getitem__(self, index):
        data = self.load(*self.data[index])
        result = dict(path=data['path'])
        midi_length = len(data['label'])
        n_steps = self.sequence_length // HOP_LENGTH
        step_begin = self.random.randint(midi_length - n_steps) if midi_length > n_steps else 0
        step_end = step_begin + n_steps
        begin = step_begin * HOP_LENGTH
        end = begin + self.sequence_length
        result['audio'] = data['audio'][begin:end]
        diff = self.sequence_length - len(result['audio'])
        result['audio'] = torch.cat((result['audio'], torch.zeros(diff, dtype=result['audio'].dtype)))
        result['audio'] = result['audio'].to(self.device)

        if 'cqt' in data:
            _cqt_full = data['cqt']                          # (168, T_cqt_full)
            _cqt_start = int(begin * _CQT_RATIO)
            _cqt_len = int(self.sequence_length * _CQT_RATIO) + 2  # fixed size, +2 safety
            _cqt_end = _cqt_start + _cqt_len
            _slice = _cqt_full[:, _cqt_start:min(_cqt_end, _cqt_full.shape[1])]
            _right_pad = _cqt_len - _slice.shape[1]
            if _right_pad > 0:
                _slice = torch.nn.functional.pad(_slice, (0, _right_pad))
            result['cqt'] = _slice.to(self.device)           # (168, _cqt_len)
        result['label'] = data['label'][step_begin:step_end, ...]
        label_len = result['label'].shape[0]
        if label_len < n_steps:
            pad = torch.zeros(n_steps - label_len, *result['label'].shape[1:], dtype=result['label'].dtype)
            result['label'] = torch.cat([result['label'], pad], dim=0)
        result['label'] = result['label'].to(self.device)
        if 'velocity' in data:
            vel = data['velocity'][step_begin:step_end, ...]
            if vel.shape[0] < n_steps:
                pad = torch.zeros(n_steps - vel.shape[0], *vel.shape[1:], dtype=vel.dtype)
                vel = torch.cat([vel, pad], dim=0)
            result['velocity'] = vel.to(self.device).float() / 128.

        result['audio'] = result['audio'].float()
        result['audio'] = result['audio'].div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()

        if 'onset_mask' in data:
            om = data['onset_mask'][step_begin:step_end, ...].to(self.device).float()
            if om.shape[0] < n_steps:
                pad = torch.ones(n_steps - om.shape[0], *om.shape[1:], device=self.device)
                om = torch.cat([om, pad], dim=0)
            result['onset_mask'] = om
        if 'frame_mask' in data:
            fm = data['frame_mask'][step_begin:step_end, ...].to(self.device).float()
            if fm.shape[0] < n_steps:
                pad = torch.ones(n_steps - fm.shape[0], *fm.shape[1:], device=self.device)
                fm = torch.cat([fm, pad], dim=0)
            result['frame_mask'] = fm


        shape = result['frame'].shape
        keys = N_KEYS
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        # frame and offset currently do not differentiate between instruments,
        # so we compress them across instrument and save a copy of the original,
        # as 'big_frame' and 'big_offset'
        result['big_frame'] = result['frame']
        result['frame'], _ = result['frame'].reshape(new_shape).max(axis=-2)
        result['big_offset'] = result['offset']
        result['offset'], _ = result['offset'].reshape(new_shape).max(axis=-2)
        return result

    def load(self, audio_path, tsv_path):
        data = self.pts[audio_path]
        if len(data['audio'].shape) > 1:
            data['audio'] = (data['audio'].float().mean(dim=-1)).short()
        if 'label' in data:
            return data
        else:
            piece, part = audio_path.split('/')[-2:]
            piece_split = piece.split('#')
            if len(piece_split) == 2:
                piece, shift1 = piece_split
            else:
                piece, shift1 = '#'.join(piece_split[:2]), piece_split[-1]
            part_split = part.split('#')
            if len(part_split) == 2:
                part, shift2 = part_split
            else:
                part, shift2 = '#'.join(part_split[:2]), part_split[-1]
            shift2, _ = shift2.split('.')
            assert shift1 == shift2
            shift = shift1
            assert shift != 0
            orig = audio_path.replace('#{}'.format(shift), '#0')
            res = {}
            res['label'] = shift_label(self.pts[orig]['label'], int(shift))
            res['path'] = audio_path
            res['audio'] = data['audio']
            if 'cqt' in data:
                res['cqt'] = data['cqt']
            if 'velocity' in self.pts[orig]:
                res['velocity'] = shift_label(self.pts[orig]['velocity'], int(shift))
            if 'onset_mask' in self.pts[orig]:
                res['onset_mask'] = shift_label(self.pts[orig]['onset_mask'], int(shift))
            if 'frame_mask' in self.pts[orig]:
                res['frame_mask'] = shift_label(self.pts[orig]['frame_mask'], int(shift))
            return res

    def load_pts(self, files):
        self.pts = {}
        print('loading pts...')
        for flac, tsv in tqdm(files):
            print('flac, tsv', flac, tsv)
            _cache_path = self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt')
            _cache_valid = False
            _adapter_mode = len(self.instruments) == 0
            if os.path.isfile(_cache_path):
                _cached = torch.load(_cache_path, weights_only=False)
                _needs_cqt = _adapter_mode and 'cqt' not in _cached
                if 'unaligned_label' in _cached:
                    _expected_cols = (len(self.instruments) + 1) * N_KEYS
                    if _cached['unaligned_label'].shape[-1] != _expected_cols:
                        print(f'Cache column mismatch for {flac}: '
                              f'{_cached["unaligned_label"].shape[-1]} vs {_expected_cols}, regenerating.')
                        os.remove(_cache_path)
                    elif _needs_cqt:
                        print(f'Cache missing CQT for {flac}, regenerating.')
                        os.remove(_cache_path)
                    else:
                        self.pts[flac] = _cached
                        _cache_valid = True
                else:
                    if _needs_cqt:
                        print(f'Cache missing CQT for {flac}, regenerating.')
                        os.remove(_cache_path)
                    else:
                        self.pts[flac] = _cached
                        _cache_valid = True
            if not _cache_valid:
                if flac.count('#') != 2:
                    print('two #', flac)
                audio, sr = soundfile.read(flac, dtype='int16')
                if len(audio.shape) == 2:
                    audio = audio.astype(float).mean(axis=1)
                else:
                    audio = audio.astype(float)
                audio = audio.astype(np.int16)
                print('audio len', len(audio))
                assert sr == SAMPLE_RATE
                audio = torch.ShortTensor(audio)
                if '#0' not in flac:
                    assert '#' in flac
                    data = {'audio': audio}
                    if _adapter_mode:
                        data['cqt'] = _compute_cqt(audio)
                    self.pts[flac] = data
                    torch.save(data,
                               self.labels_path + '/' + flac.split('/')[-1]
                               .replace('.flac', '.pt').replace('.mp3', '.pt'))
                    continue
                midi = np.loadtxt(tsv, delimiter='\t', skiprows=1, ndmin=2)
                unaligned_label = midi_to_frames(midi, self.instruments, conversion_map=self.conversion_map)
                data = dict(path=self.labels_path + '/' + flac.split('/')[-1],
                            audio=audio, unaligned_label=unaligned_label)
                if _adapter_mode:
                    data['cqt'] = _compute_cqt(audio)
                torch.save(data, self.labels_path + '/' + flac.split('/')[-1]
                               .replace('.flac', '.pt').replace('.mp3', '.pt'))
                self.pts[flac] = data

    '''
    Update labels. 
    POS, NEG - pseudo labels positive and negative thresholds.
    PITCH_POS - pseudo labels positive thresholds for the pitch-only classes.
    first - is this the first labelling iteration.
    update - should the labels indeed be updated - if not, just saves the output.
    BEST_BON - if true, will update labels only if the bag of notes distance between the unaligned midi and the prediction improved.
    Bag of notes distance is computed based on pitch only.
    '''
    def update_pts(self, transcriber, POS=1.1, NEG=-0.001, FRAME_POS=0.5,
                   to_save=None, first=False, update=True, BEST_BON=False,
                   viz_dir=None, viz_keys=None, viz_tag='',
                   gt_json_path=None, notes_json_path=None):
        print('Updating pts...')
        print('POS, NEG', POS, NEG)
        if to_save is not None:
            os.makedirs(to_save, exist_ok=True)
        _viz_keys = set(viz_keys) if viz_keys is not None else set()
        _gt_data = {}
        if gt_json_path is not None and _viz_keys:
            try:
                with open(gt_json_path) as f:
                    _gt_data = json.load(f)
            except Exception as e:
                print(f'Warning: could not load GT JSON: {e}')

        # Detect whether the transcriber is an AMTAdapter so we can pass raw audio
        # instead of a pre-computed mel spectrogram.
        from onsets_and_frames.transcriber import AMTAdapter
        from torch.nn import DataParallel
        _inner = transcriber.module if isinstance(transcriber, DataParallel) else transcriber
        _is_adapter = isinstance(_inner, AMTAdapter)

        print('there are', len(self.pts), 'pts')
        for flac, data in self.pts.items():
            if 'unaligned_label' not in data:
                continue
            audio_inp = data['audio'].float() / 32768.
            MAX_TIME = 5 * 60 * SAMPLE_RATE
            audio_inp_len = len(audio_inp)
            if audio_inp_len > MAX_TIME:
                n_segments = 3 if audio_inp_len > 2 * MAX_TIME else 2
                print('long audio, splitting to {} segments'.format(n_segments))
                seg_len = audio_inp_len // n_segments
                onsets_preds = []
                offset_preds = []
                frame_preds = []
                vel_preds = []
                for i_s in range(n_segments):
                    curr = audio_inp[i_s * seg_len: (i_s + 1) * seg_len].unsqueeze(0).cuda()
                    if _is_adapter:
                        curr_onset_pred, curr_offset_pred, _, curr_frame_pred, curr_velocity_pred = transcriber(curr)
                    else:
                        curr_mel = melspectrogram(curr.reshape(-1, curr.shape[-1])[:, :-1]).transpose(-1, -2)
                        curr_onset_pred, curr_offset_pred, _, curr_frame_pred, curr_velocity_pred = transcriber(curr_mel)
                    onsets_preds.append(curr_onset_pred)
                    offset_preds.append(curr_offset_pred)
                    frame_preds.append(curr_frame_pred)
                    vel_preds.append(curr_velocity_pred)
                onset_pred = torch.cat(onsets_preds, dim=1)
                offset_pred = torch.cat(offset_preds, dim=1)
                frame_pred = torch.cat(frame_preds, dim=1)
                velocity_pred = torch.cat(vel_preds, dim=1)
            else:
                audio_inp = audio_inp.unsqueeze(0).cuda()
                if _is_adapter:
                    _cqt = data.get('cqt')
                    _cqt_inp = _cqt.unsqueeze(0).cuda() if _cqt is not None else None
                    onset_pred, offset_pred, _, frame_pred, velocity_pred = transcriber(audio_inp, _cqt_inp)
                else:
                    mel = melspectrogram(audio_inp.reshape(-1, audio_inp.shape[-1])[:, :-1]).transpose(-1, -2)
                    onset_pred, offset_pred, _, frame_pred, velocity_pred = transcriber(mel)
            print('done predicting.')
            # We assume onset predictions are of length N_KEYS * (len(instruments) + 1),
            # first N_KEYS classes are the first instrument, next N_KEYS classes are the next instrument, etc.,
            # and last N_KEYS classes are for pitch regardless of instrument
            # Currently, frame and offset predictions are only N_KEYS classes.
            onset_pred = onset_pred.detach().squeeze().cpu()
            frame_pred = frame_pred.detach().squeeze().cpu()

            peaks = get_peaks(onset_pred, 3) # we only want local peaks, in a 7-frame neighborhood, 3 to each side.
            onset_pred[~peaks] = 0

            unaligned_onsets = (data['unaligned_label'] == 3).float().numpy()
            unaligned_frames = (data['unaligned_label'] >= 2).float().numpy()

            onset_pred_np = onset_pred.numpy()
            frame_pred_np = frame_pred.numpy()

            ####
            pred_bag_of_notes = (onset_pred_np[:, -N_KEYS:] >= 0.5).sum(axis=0)
            gt_bag_of_notes = unaligned_onsets[:, -N_KEYS:].astype(bool).sum(axis=0)
            bon_dist = (((pred_bag_of_notes - gt_bag_of_notes) ** 2).sum()) ** 0.5
            # print('pred bag of notes', pred_bag_of_notes)
            # print('gt bag of notes', gt_bag_of_notes)
            bon_dist /= gt_bag_of_notes.sum()
            print('bag of notes dist', bon_dist)
            ####

            # We align based on likelihoods regardless of the octave (chroma features)
            onset_pred_comp = compress_across_octave(onset_pred_np[:, -N_KEYS:])
            onset_label_comp = compress_across_octave(unaligned_onsets[:, -N_KEYS:])
            # We can do DTW on super-frames since anyway we search for local max afterwards
            onset_pred_comp = compress_time(onset_pred_comp, DTW_FACTOR)
            onset_label_comp = compress_time(onset_label_comp, DTW_FACTOR)
            print('dtw lengths', len(onset_pred_comp), len(onset_label_comp))
            init_time = time.time()
            alignment = dtw(onset_pred_comp, onset_label_comp, dist_method='euclidean',
                            )
            finish_time = time.time()
            print('DTW took {} seconds.'.format(finish_time - init_time))
            index1, index2 = alignment.index1, alignment.index2
            matches1, matches2 = get_matches(index1, index2), get_matches(index2, index1)

            aligned_onsets = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_frames = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_offsets = np.zeros(onset_pred_np.shape, dtype=bool)

            # We go over onsets (t, f) in the unaligned midi. For each onset, we find its approximate time based on DTW,
            # then find its precise time with likelihood local max
            for t, f in zip(*unaligned_onsets.nonzero()):
                t_comp = t // DTW_FACTOR
                t_src = matches2[t_comp]
                t_sources = list(range(DTW_FACTOR * min(t_src), DTW_FACTOR * max(t_src) + 1))
                # we extend the search area of local max to be ~0.5 second:
                t_sources_extended = get_margin(t_sources, len(aligned_onsets))
                # eliminate occupied positions. Allow only a single onset per 5 frames:
                existing_eliminated = [t_source for t_source in t_sources_extended if (aligned_onsets[t_source - 2: t_source + 3, f] == 0).all()]
                if len(existing_eliminated) > 0:
                    t_sources_extended = existing_eliminated

                t_src = max(t_sources_extended, key=lambda x: onset_pred_np[x, f]) # t_src is the most likely time in the local neighborhood for this note onset
                f_pitch = (len(self.instruments) * N_KEYS) + (f % N_KEYS)
                if onset_pred_np[t_src, f_pitch] < NEG: # filter negative according to pitch-only likelihood (can use f instead)
                    continue
                aligned_onsets[t_src, f] = 1 # set the label

                # Now we need to decide note duration and offset time. Find note length in unaligned midi:
                t_off = t
                while t_off < len(unaligned_frames) and unaligned_frames[t_off, f]:
                    t_off += 1
                note_len = t_off - t # this is the note length in the unaligned midi. We need note length in the audio.

                # option 1: use mapping, traverse note length in the unaligned midi, and then use the reverse mapping:
                try:
                    t_off_src1 = max(matches2[(DTW_FACTOR * max(matches1[t_src // DTW_FACTOR]) + note_len) // DTW_FACTOR]) * DTW_FACTOR
                    t_off_src1 = max(t_src + 1, t_off_src1)
                except Exception as e:
                    t_off_src1 = len(aligned_offsets)
                # option 2: use relative note length
                t_off_src2 = t_src + int(note_len * (len(aligned_onsets) / len(unaligned_onsets)))
                t_off_src2 = min(len(aligned_onsets), t_off_src2)

                t_off_src = t_off_src2 # we choose option 2
                aligned_frames[t_src: t_off_src, f] = 1

                if t_off_src < len(aligned_offsets):
                    aligned_offsets[t_off_src, f] = 1

            # eliminate instruments that do not exist in the unaligned midi
            inactive_instruments, active_instruments_list = get_inactive_instruments(unaligned_onsets, len(aligned_onsets))
            onset_pred_np[inactive_instruments] = 0

            pseudo_onsets = (onset_pred_np >= POS) & (~aligned_onsets)
            inst_only = len(self.instruments) * N_KEYS
            if first: # do not use pseudo labels for instruments in first labelling iteration since the model doesn't distinguish yet
                pseudo_onsets[:, : inst_only] = 0
            onset_label = np.maximum(pseudo_onsets, aligned_onsets)

            onsets_unknown = (onset_pred_np >= 0.5) & (~onset_label) # for mask
            if first: # do not use mask for instruments in first labelling iteration since the model doesn't distinguish yet between instruments
                onsets_unknown[:, : inst_only] = 0
            onset_mask = torch.from_numpy(~onsets_unknown).byte()
            # onset_mask = torch.ones(onset_label.shape).byte()

            pseudo_frames = np.zeros(pseudo_onsets.shape, dtype=pseudo_onsets.dtype)
            pseudo_offsets = np.zeros(pseudo_onsets.shape, dtype=pseudo_onsets.dtype)
            for t, f in zip(*onset_label.nonzero()):
                t_off = t
                while t_off < len(pseudo_frames) and frame_pred[t_off, f % N_KEYS] >= FRAME_POS:
                    t_off += 1
                pseudo_frames[t: t_off, f] = 1
                if t_off < len(pseudo_offsets):
                    pseudo_offsets[t_off, f] = 1
            frame_label = np.maximum(pseudo_frames, aligned_frames)
            offset_label = get_diff(frame_label, offset=True)

            frames_pitch_only = frame_label[:, -N_KEYS:]
            frames_unknown = (frame_pred_np >= 0.5) & (~frames_pitch_only)
            frame_mask = torch.from_numpy(~frames_unknown).byte()
            # frame_mask = torch.ones(frame_pred.shape).byte()

            label = np.maximum(2 * frame_label, offset_label)
            label = np.maximum(3 * onset_label, label).astype(np.uint8)

            if viz_dir is not None and flac in _viz_keys:
                try:
                    sid_str, seg_idx, seg_key = _parse_seg_key(flac)
                    gt_notes = _get_gt_notes_for_segment(_gt_data, sid_str, seg_idx)
                    fps = SAMPLE_RATE / HOP_LENGTH
                    pred_notes = []
                    pred_pitch = aligned_frames[:, -N_KEYS:]
                    for col in range(N_KEYS):
                        in_note, t_on = False, 0
                        for t, val in enumerate(pred_pitch[:, col]):
                            if val and not in_note:
                                t_on, in_note = t, True
                            elif not val and in_note:
                                pred_notes.append((t_on / fps, t / fps, col + MIN_MIDI))
                                in_note = False
                        if in_note:
                            pred_notes.append((t_on / fps, len(pred_pitch) / fps, col + MIN_MIDI))
                    if seg_key not in self._notes_history:
                        self._notes_history[seg_key] = {}
                    self._notes_history[seg_key][viz_tag] = {
                        'pred': pred_notes,
                        'gt': [(o, f, m) for o, f, m in gt_notes],
                    }
                    _visualize_notes(viz_dir, seg_key, self._notes_history[seg_key])
                    if notes_json_path is not None:
                        try:
                            with open(notes_json_path, 'w') as jf:
                                json.dump(self._notes_history, jf)
                        except Exception as e:
                            print(f'Warning: could not write notes JSON: {e}')
                except Exception as e:
                    print(f'Warning: viz failed for {flac}: {e}')

            if to_save is not None:
                save_midi_alignments_and_predictions(to_save, data['path'], self.instruments,
                                         aligned_onsets, aligned_frames,
                                         onset_pred_np, frame_pred_np, prefix='')
                # time_now = datetime.now().strftime('%y%m%d-%H%M%S')
                # frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_alignment_' + time_now + '.mid',
                #             aligned_onsets[:, : inst_only], aligned_frames[:, : inst_only],
                #             64. * aligned_onsets[:, : inst_only],
                #             inst_mapping=self.instruments)
                # frames2midi_pitch(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_alignment_pitch_' + time_now + '.mid',
                #                 aligned_onsets[:, -N_KEYS:], aligned_frames[:, -N_KEYS:],
                #                 64. * aligned_onsets[:, -N_KEYS:])
                # predicted_onsets = onset_pred_np >= 0.5
                # predicted_frames = frame_pred_np >= 0.5
                # frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_' + time_now + '.mid',
                #             predicted_onsets[:, : inst_only], predicted_frames[:, : inst_only],
                #             64. * predicted_onsets[:, : inst_only],
                #             inst_mapping=self.instruments)
                # frames2midi_pitch(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_pitch_' + time_now + '.mid',
                #             predicted_onsets[:, -N_KEYS:], predicted_frames[:, -N_KEYS:],
                #             64. * predicted_onsets[:, -N_KEYS:])
                # if len(self.instruments) > 1:
                #     max_pred_onsets = max_inst(onset_pred_np)
                #     frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_max_' + time_now + '.mid',
                #                 max_pred_onsets[:, : inst_only], predicted_frames[:, : inst_only],
                #                 64. * max_pred_onsets[:, : inst_only],
                #                 inst_mapping=self.instruments)
            if update:
                if not BEST_BON or bon_dist < data.get('BON', float('inf')):
                    data['label'] = torch.from_numpy(label).byte()
                    data['onset_mask'] = onset_mask
                    data['frame_mask'] = frame_mask
                if bon_dist < data.get('BON', float('inf')):
                    print('Bag of notes distance improved from {} to {}'.format(data.get('BON', float('inf')), bon_dist))
                    data['BON'] = bon_dist

                    if to_save is not None:
                        os.makedirs(to_save + '/BEST_BON', exist_ok=True)
                        save_midi_alignments_and_predictions(to_save + '/BEST_BON', data['path'], self.instruments,
                                                             aligned_onsets, aligned_frames,
                                                             onset_pred_np, frame_pred_np, prefix='BEST_BON')

            velocity_pred = velocity_pred.detach().squeeze().cpu()
            # velocity_pred = torch.from_numpy(new_vels)
            velocity_pred = (128. * velocity_pred)
            velocity_pred[velocity_pred < 0.] = 0.
            velocity_pred[velocity_pred > 127.] = 127.
            velocity_pred = velocity_pred.byte()
            if update:
                data['velocity'] = velocity_pred

            del audio_inp
            try:
                del mel
            except:
                pass
            del onset_pred
            del offset_pred
            del frame_pred
            del velocity_pred
            torch.cuda.empty_cache()

    '''
        Update labels. Use only alignment without pseudo-labels.
    '''

    def update_pts_vanilla(self, transcriber,
                   to_save=None, first=False, update=True):
        print('Updating pts...')
        if to_save is not None:
            os.makedirs(to_save, exist_ok=True)
        print('there are', len(self.pts), 'pts')
        for flac, data in self.pts.items():
            if 'unaligned_label' not in data:
                continue
            audio_inp = data['audio'].float() / 32768.
            MAX_TIME = 5 * 60 * SAMPLE_RATE
            audio_inp_len = len(audio_inp)
            if audio_inp_len > MAX_TIME:
                n_segments = 3 if audio_inp_len > 2 * MAX_TIME else 2
                print('long audio, splitting to {} segments'.format(n_segments))
                seg_len = audio_inp_len // n_segments
                onsets_preds = []
                offset_preds = []
                frame_preds = []
                vel_preds = []
                for i_s in range(n_segments):
                    curr = audio_inp[i_s * seg_len: (i_s + 1) * seg_len].unsqueeze(0).cuda()
                    curr_mel = melspectrogram(curr.reshape(-1, curr.shape[-1])[:, :-1]).transpose(-1, -2)
                    curr_onset_pred, curr_offset_pred, _, curr_frame_pred, curr_velocity_pred = transcriber(curr_mel)
                    onsets_preds.append(curr_onset_pred)
                    offset_preds.append(curr_offset_pred)
                    frame_preds.append(curr_frame_pred)
                    vel_preds.append(curr_velocity_pred)
                onset_pred = torch.cat(onsets_preds, dim=1)
                offset_pred = torch.cat(offset_preds, dim=1)
                frame_pred = torch.cat(frame_preds, dim=1)
                velocity_pred = torch.cat(vel_preds, dim=1)
            else:
                audio_inp = audio_inp.unsqueeze(0).cuda()
                mel = melspectrogram(audio_inp.reshape(-1, audio_inp.shape[-1])[:, :-1]).transpose(-1, -2)
                onset_pred, offset_pred, _, frame_pred, velocity_pred = transcriber(mel)
            print('done predicting.')
            # We assume onset predictions are of length N_KEYS * (len(instruments) + 1),
            # first N_KEYS classes are the first instrument, next N_KEYS classes are the next instrument, etc.,
            # and last N_KEYS classes are for pitch regardless of instrument
            # Currently, frame and offset predictions are only N_KEYS classes.
            onset_pred = onset_pred.detach().squeeze().cpu()
            frame_pred = frame_pred.detach().squeeze().cpu()

            peaks = get_peaks(onset_pred, 3)  # we only want local peaks, in a 7-frame neighborhood, 3 to each side.
            onset_pred[~peaks] = 0

            unaligned_onsets = (data['unaligned_label'] == 3).float().numpy()
            unaligned_frames = (data['unaligned_label'] >= 2).float().numpy()

            onset_pred_np = onset_pred.numpy()
            frame_pred_np = frame_pred.numpy()

            # We align based on likelihoods regardless of the octave (chroma features)
            onset_pred_comp = compress_across_octave(onset_pred_np[:, -N_KEYS:])
            onset_label_comp = compress_across_octave(unaligned_onsets[:, -N_KEYS:])
            # We can do DTW on super-frames since anyway we search for local max afterwards
            onset_pred_comp = compress_time(onset_pred_comp, DTW_FACTOR)
            onset_label_comp = compress_time(onset_label_comp, DTW_FACTOR)
            print('dtw lengths', len(onset_pred_comp), len(onset_label_comp))
            init_time = time.time()
            alignment = dtw(onset_pred_comp, onset_label_comp, dist_method='euclidean',
                            )
            finish_time = time.time()
            print('DTW took {} seconds.'.format(finish_time - init_time))
            index1, index2 = alignment.index1, alignment.index2
            matches1, matches2 = get_matches(index1, index2), get_matches(index2, index1)

            aligned_onsets = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_frames = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_offsets = np.zeros(onset_pred_np.shape, dtype=bool)

            # We go over onsets (t, f) in the unaligned midi. For each onset, we find its approximate time based on DTW,
            # then find its precise time with likelihood local max
            for t, f in zip(*unaligned_onsets.nonzero()):
                t_comp = t // DTW_FACTOR
                t_src = matches2[t_comp]
                t_sources = list(range(DTW_FACTOR * min(t_src), DTW_FACTOR * max(t_src) + 1))
                # we extend the search area of local max to be ~0.5 second:
                t_sources_extended = get_margin(t_sources, len(aligned_onsets))
                # eliminate occupied positions. Allow only a single onset per 5 frames:
                existing_eliminated = [t_source for t_source in t_sources_extended if (aligned_onsets[t_source - 2: t_source + 3, f] == 0).all()]
                if len(existing_eliminated) > 0:
                    t_sources_extended = existing_eliminated

                t_src = max(t_sources_extended, key=lambda x: onset_pred_np[x, f])  # t_src is the most likely time in the local neighborhood for this note onset
                f_pitch = (len(self.instruments) * N_KEYS) + (f % N_KEYS)
                aligned_onsets[t_src, f] = 1  # set the label
                # Now we need to decide note duration and offset time. Find note length in unaligned midi:
                t_off = t
                while t_off < len(unaligned_frames) and unaligned_frames[t_off, f]:
                    t_off += 1
                note_len = t_off - t  # this is the note length in the unaligned midi. We need note length in the audio.

                # option 1: use mapping, traverse note length in the unaligned midi, and then use the reverse mapping:
                try:
                    t_off_src1 = max(matches2[(DTW_FACTOR * max(matches1[t_src // DTW_FACTOR]) + note_len) // DTW_FACTOR]) * DTW_FACTOR
                    t_off_src1 = max(t_src + 1, t_off_src1)
                except Exception as e:
                    t_off_src1 = len(aligned_offsets)
                # option 2: use relative note length
                t_off_src2 = t_src + int(note_len * (len(aligned_onsets) / len(unaligned_onsets)))
                t_off_src2 = min(len(aligned_onsets), t_off_src2)

                t_off_src = t_off_src2  # we choose option 2
                aligned_frames[t_src: t_off_src, f] = 1

                if t_off_src < len(aligned_offsets):
                    aligned_offsets[t_off_src, f] = 1

            # eliminate instruments that do not exist in the unaligned midi
            inactive_instruments, active_instruments_list = get_inactive_instruments(unaligned_onsets, len(aligned_onsets))
            onset_pred_np[inactive_instruments] = 0

            onset_label = aligned_onsets
            frame_label = aligned_frames
            offset_label = aligned_offsets
            label = np.maximum(2 * frame_label, offset_label)
            label = np.maximum(3 * onset_label, label).astype(np.uint8)

            if to_save is not None:
                inst_only = len(self.instruments) * N_KEYS
                time_now = datetime.now().strftime('%y%m%d-%H%M%S')
                frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_alignment_' + time_now + '.mid',
                            aligned_onsets[:, : inst_only], aligned_frames[:, : inst_only],
                            64. * aligned_onsets[:, : inst_only],
                            inst_mapping=self.instruments)
                frames2midi_pitch(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_alignment_pitch_' + time_now + '.mid',
                                  aligned_onsets[:, -N_KEYS:], aligned_frames[:, -N_KEYS:],
                                  64. * aligned_onsets[:, -N_KEYS:])
                predicted_onsets = onset_pred_np >= 0.5
                predicted_frames = frame_pred_np >= 0.5
                frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_' + time_now + '.mid',
                            predicted_onsets[:, : inst_only], predicted_frames[:, : inst_only],
                            64. * predicted_onsets[:, : inst_only],
                            inst_mapping=self.instruments)
                frames2midi_pitch(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_pitch_' + time_now + '.mid',
                                  predicted_onsets[:, -N_KEYS:], predicted_frames[:, -N_KEYS:],
                                  64. * predicted_onsets[:, -N_KEYS:])
                if len(self.instruments) > 1:
                    max_pred_onsets = max_inst(onset_pred_np)
                    frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_max_' + time_now + '.mid',
                                max_pred_onsets[:, : inst_only], predicted_frames[:, : inst_only],
                                64. * max_pred_onsets[:, : inst_only],
                                inst_mapping=self.instruments)
            if update:
                data['label'] = torch.from_numpy(label).byte()

            velocity_pred = velocity_pred.detach().squeeze().cpu()
            # velocity_pred = torch.from_numpy(new_vels)
            velocity_pred = (128. * velocity_pred)
            velocity_pred[velocity_pred < 0.] = 0.
            velocity_pred[velocity_pred > 127.] = 127.
            velocity_pred = velocity_pred.byte()
            if update:
                data['velocity'] = velocity_pred

            del audio_inp
            try:
                del mel
            except:
                pass
            del onset_pred
            del offset_pred
            del frame_pred
            del velocity_pred
            torch.cuda.empty_cache()
