import numpy as np
import librosa
import torch
import torch.nn.functional as F
from torch import nn
from .mel import melspectrogram
from .lstm import BiLSTM
from onsets_and_frames.constants import *

class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=13):
        nn.Module.__init__(self)
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        onset_model_size = int(onset_complexity * model_size)
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, onset_model_size),
            sequence_model(onset_model_size, onset_model_size),
            nn.Linear(onset_model_size, output_features * n_instruments),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features * n_instruments)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)

        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, parallel_model=None, multi=False, positive_weight=2., inv_positive_weight=2.):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)

        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        if multi:
            onset_pred = onset_pred[..., : N_KEYS]
            offset_pred = offset_pred[..., : N_KEYS]
            frame_pred = frame_pred[..., : N_KEYS]
            velocity_pred = velocity_pred[..., : N_KEYS]

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
        if 'velocity' in batch:
            losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        onset_mask = 1. * onset_label
        onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
        onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        onset_mask += 1
        if 'onset_mask' in batch:
            onset_mask = onset_mask * batch['onset_mask']

        offset_mask = 1. * offset_label
        offset_positive_weight = 2.
        offset_mask *= (offset_positive_weight - 1)
        offset_mask += 1.

        frame_mask = 1. * frame_label
        frame_positive_weight = 2.
        frame_mask *= (frame_positive_weight - 1)
        frame_mask += 1.

        for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, offset_mask, frame_mask]):
            losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator


def duplicate_linear(linear, n):
    A, b = linear.parameters()
    in_features, out_features = linear.in_features, linear.out_features
    layer_new = torch.nn.Linear(in_features, n * out_features)
    A_new, b_new = layer_new.parameters()
    A_new.requires_grad, b_new.requires_grad = False, False
    for j in range(n):
        A_new[j * out_features: (j + 1) * out_features, :] = A.detach().clone()
        b_new[j * out_features: (j + 1) * out_features] = b.detach().clone()
    A_new.requires_grad, b_new.requires_grad = True, True
    return layer_new


def load_weights(model, old_model, n_instruments):
    for i in range(len(model.onset_stack)):
        if i < len(model.onset_stack) - 2:
            model.onset_stack[i].load_state_dict(old_model.onset_stack[i].state_dict())
        elif i < len(model.onset_stack) - 1:
            linear = old_model.onset_stack[i]
            layer_new = duplicate_linear(linear, n_instruments)
            model.onset_stack[i].load_state_dict(layer_new.state_dict())

    for i in range(len(model.frame_stack)):
        if i < len(model.frame_stack) - 1:
            model.frame_stack[i].load_state_dict(old_model.frame_stack[i].state_dict())

    for i in range(len(model.combined_stack)):
        if i < len(model.combined_stack) - 1:
            model.combined_stack[i].load_state_dict(old_model.combined_stack[i].state_dict())

    for i in range(len(model.offset_stack)):
        if i < len(model.offset_stack) - 1:
            model.offset_stack[i].load_state_dict(old_model.offset_stack[i].state_dict())

    for i in range(len(model.velocity_stack)):
        if i < len(model.velocity_stack) - 1:
            model.velocity_stack[i].load_state_dict(old_model.velocity_stack[i].state_dict())
        elif i < len(model.velocity_stack):
            linear = old_model.velocity_stack[i]
            layer_new = duplicate_linear(linear, n_instruments)
            model.velocity_stack[i].load_state_dict(layer_new.state_dict())


class AMTAdapter(nn.Module):
    """Wraps EffNetb0 to provide an OnsetsAndFrames-compatible interface for the EM loop.

    The backbone expects CQT features at 44.1 kHz; this adapter handles resampling,
    feature extraction, output conversion, and time-axis alignment internally so that
    the rest of the training pipeline is unaffected.
    """

    _SR_AST = 44100
    _HOP_AST = 1024
    _N_BINS = 168          # 84 * 2 CQT bins
    _BINS_PER_OCT = 24     # 12 * 2 bins per octave
    _WIN_FRAMES = 11       # context window fed to EffNetb0 (5 left + center + 5 right)
    _MIDI_LOW = 36         # lowest MIDI pitch covered by AST model (C2)
    _MIDI_HIGH = 83        # highest MIDI pitch covered by AST model (B5)

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        # Named sub-modules required by set_diff() in train.py.
        # frame/offset/combined/velocity point to Identity (no parameters) so those
        # set_diff() calls are no-ops; actual freezing is done explicitly in train.py.
        self.onset_stack = backbone.effnet.blocks
        self.frame_stack = nn.Identity()
        self.offset_stack = nn.Identity()
        self.combined_stack = nn.Identity()
        self.velocity_stack = nn.Identity()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _audio_to_windows(self, audio_1d, cqt=None):
        """Convert 1-D float audio at 16 kHz into batched CQT context windows.

        audio_1d : (L,) float tensor, range [-1, 1], any device
        cqt      : (168, T_cqt) float tensor (pre-computed); if provided, skips
                   resample + CQT and uses this directly (the fast path).
        returns  : (T_cqt, 1, 11, 168) float tensor on the backbone's device
        """
        device = next(self.backbone.parameters()).device

        if cqt is None:
            audio_np = audio_1d.float().cpu().numpy()
            audio_44k = librosa.resample(audio_np, orig_sr=SAMPLE_RATE, target_sr=self._SR_AST)
            fmin = librosa.midi_to_hz(self._MIDI_LOW)
            cqt_mag = np.abs(librosa.cqt(
                audio_44k,
                sr=self._SR_AST,
                hop_length=self._HOP_AST,
                fmin=fmin,
                n_bins=self._N_BINS,
                bins_per_octave=self._BINS_PER_OCT,
            ))
            cqt = torch.from_numpy(cqt_mag).float()

        cqt = cqt.to(device)  # (168, T_cqt)

        # Zero-pad time axis by 5 on each side, then unfold into overlapping windows
        pad = self._WIN_FRAMES // 2  # = 5
        cqt_padded = F.pad(cqt.unsqueeze(0), (pad, pad), value=0.0)  # (1, 168, T_cqt+10)
        windows = cqt_padded.unfold(2, self._WIN_FRAMES, 1)          # (1, 168, T_cqt, 11)
        windows = windows.permute(2, 0, 3, 1)                        # (T_cqt, 1, 11, 168)
        return windows

    def _per_pitch_probs(self, onset_logits, offset_logits, oct_logits, cls_logits):
        """Convert EffNetb0 frame-level logits to per-MIDI-pitch probability tensors.

        onset_logits  : (T,)
        offset_logits : (T,)
        oct_logits    : (T, pitch_octave+1=5)  — index 4 is the "no pitch" octave
        cls_logits    : (T, pitch_class+1=13)  — index 12 is the "no pitch" class

        Returns onset_pp, frame_pp, offset_pp each (T, N_KEYS=88).
        onset_pp[t, k]  = onset_sig[t] * P(octave_of_k | t) * P(class_of_k | t)
        frame_pp[t, k]  = P(octave_of_k | t) * P(class_of_k | t)   (voiced proxy)
        offset_pp[t, k] = offset_sig[t] broadcast over all keys
        Pitches outside MIDI 36-83 remain zero.
        """
        T = onset_logits.shape[0]
        onset_sig = torch.sigmoid(onset_logits)    # (T,)
        offset_sig = torch.sigmoid(offset_logits)  # (T,)
        oct_probs = torch.softmax(oct_logits, dim=-1)  # (T, 5)
        cls_probs = torch.softmax(cls_logits, dim=-1)  # (T, 13)

        onset_pp = torch.zeros(T, N_KEYS, device=onset_logits.device)
        frame_pp = torch.zeros(T, N_KEYS, device=onset_logits.device)

        for midi in range(self._MIDI_LOW, self._MIDI_HIGH + 1):
            key_idx = midi - MIN_MIDI                  # index in 88-key space (0-87)
            oct_idx = (midi - self._MIDI_LOW) // 12   # 0-3
            cls_idx = midi % 12                        # 0-11
            pitch_p = oct_probs[:, oct_idx] * cls_probs[:, cls_idx]  # (T,)
            onset_pp[:, key_idx] = onset_sig * pitch_p
            frame_pp[:, key_idx] = pitch_p

        # Offset: scalar per frame, broadcast across all pitches
        offset_pp = offset_sig.unsqueeze(-1).expand(T, N_KEYS).clone()
        return onset_pp, frame_pp, offset_pp

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, audio, cqt=None):
        """Run the adapter forward pass.

        audio : (B, L) or (L,) float tensor at 16 kHz, range [-1, 1]
        cqt   : (B, 168, T_cqt) optional pre-computed CQT (skips resample+CQT when set)

        Returns 5-tuple matching OnsetsAndFrames.forward() output shapes:
            onset_pred    : (B, T_mel, N_KEYS)
            offset_pred   : (B, T_mel, N_KEYS)
            activation    : (B, T_mel, N_KEYS)  — 3rd output, ignored by callers
            frame_pred    : (B, T_mel, N_KEYS)
            velocity_pred : (B, T_mel, N_KEYS)  — zeros (AST has no velocity output)

        T_mel is computed to match the mel-spectrogram frame count for the same audio,
        ensuring pseudo-label shapes from the dataset are compatible.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        B, L = audio.shape
        T_mel = (L - 1) // HOP_LENGTH + 1

        onset_list, frame_list, offset_list = [], [], []
        for b in range(B):
            cqt_b = cqt[b] if cqt is not None else None
            windows = self._audio_to_windows(audio[b], cqt=cqt_b)  # (T_cqt, 1, 11, 168)

            # Process in chunks to avoid OOM on long clips
            _chunk = 32
            o_chunks, of_chunks, oct_chunks, cls_chunks = [], [], [], []
            for _i in range(0, windows.shape[0], _chunk):
                _w = windows[_i:_i + _chunk]
                _o, _of, _oct, _cls = self.backbone(_w)
                o_chunks.append(_o); of_chunks.append(_of)
                oct_chunks.append(_oct); cls_chunks.append(_cls)
            o_log   = torch.cat(o_chunks)
            of_log  = torch.cat(of_chunks)
            oct_log = torch.cat(oct_chunks)
            cls_log = torch.cat(cls_chunks)
            onset_pp, frame_pp, offset_pp = self._per_pitch_probs(
                o_log, of_log, oct_log, cls_log
            )  # each (T_cqt, N_KEYS)

            # Resample time axis from CQT rate to mel rate — differentiable via F.interpolate
            def _interp(x):
                # x: (T_cqt, N_KEYS) → treat pitch as channel dim for 1-D interpolation
                return F.interpolate(
                    x.T.unsqueeze(0), size=T_mel, mode='linear', align_corners=False
                ).squeeze(0).T  # (T_mel, N_KEYS)

            onset_list.append(_interp(onset_pp))
            frame_list.append(_interp(frame_pp))
            offset_list.append(_interp(offset_pp))

        onset_pred = torch.stack(onset_list)   # (B, T_mel, N_KEYS)
        frame_pred = torch.stack(frame_list)
        offset_pred = torch.stack(offset_list)
        velocity_pred = torch.zeros_like(onset_pred)

        return onset_pred, offset_pred, frame_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, parallel_model=None, multi=False,
                     positive_weight=2., inv_positive_weight=2.):
        """M-step: compute predictions and weighted BCE losses against pseudo-labels.

        Calls backward() per clip so only one clip's gradient graph lives in VRAM
        at a time (~2.5 GB peak vs ~20 GB for a naive batch of 8).
        Returns detached scalar losses — caller must NOT call loss.backward() again.
        """
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        cqt_batch = batch.get('cqt')   # (B, 168, T_cqt_seq) or None
        B = audio_label.shape[0]

        # Precompute loss masks (no model involvement)
        onset_mask = 1. * onset_label
        onset_mask[..., :-N_KEYS] *= (positive_weight - 1)
        onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        onset_mask += 1
        if 'onset_mask' in batch:
            onset_mask = onset_mask * batch['onset_mask']
        offset_mask = 1. * offset_label * (2. - 1) + 1.
        frame_mask = 1. * frame_label * (2. - 1) + 1.

        total_onset = 0.
        total_offset = 0.
        total_frame = 0.
        onset_preds, frame_preds, offset_preds = [], [], []

        for b in range(B):
            cqt_b = cqt_batch[b:b + 1] if cqt_batch is not None else None
            o, of, _, f, _ = self.forward(audio_label[b:b + 1], cqt_b)

            o_loss = (onset_mask[b:b + 1] * F.binary_cross_entropy(
                o, onset_label[b:b + 1], reduction='none')).mean()
            of_loss = (offset_mask[b:b + 1] * F.binary_cross_entropy(
                of, offset_label[b:b + 1], reduction='none')).mean()
            f_loss = (frame_mask[b:b + 1] * F.binary_cross_entropy(
                f, frame_label[b:b + 1], reduction='none')).mean()

            # Divide by B so gradient magnitude matches a full-batch average
            clip_loss = (o_loss + of_loss + f_loss) / B
            clip_loss.backward()

            total_onset += o_loss.item()
            total_offset += of_loss.item()
            total_frame += f_loss.item()

            onset_preds.append(o.detach())
            frame_preds.append(f.detach())
            offset_preds.append(of.detach())

        predictions = {
            'onset': torch.cat(onset_preds, dim=0).reshape(*onset_label.shape),
            'offset': torch.cat(offset_preds, dim=0).reshape(*offset_label.shape),
            'frame': torch.cat(frame_preds, dim=0).reshape(*frame_label.shape),
        }
        if 'velocity' in batch:
            predictions['velocity'] = torch.zeros_like(batch['velocity'])

        # Non-differentiable scalars — backward is already done above
        losses = {
            'loss/onset':  torch.tensor(total_onset  / B),
            'loss/offset': torch.tensor(total_offset / B),
            'loss/frame':  torch.tensor(total_frame  / B),
        }
        return predictions, losses
