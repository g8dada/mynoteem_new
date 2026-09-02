import torch
import torch.nn as nn


class EffNetb0(nn.Module):
    """EfficientNet-B0 singing transcription model (ICASSP 2021).

    Outputs per-frame logits for onset, offset, pitch octave, and pitch class.
    Input: (B, 1, 11, 168) CQT context windows.
    """

    def __init__(self, pitch_class=12, pitch_octave=4):
        super().__init__()
        self.pitch_octave = pitch_octave
        self.pitch_class = pitch_class
        self.effnet = torch.hub.load(
            '/root/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master',
            'efficientnet_b0', pretrained=False, verbose=False, source='local',
        )
        self.effnet.conv_stem = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.effnet.classifier = nn.Linear(
            self.effnet.classifier.in_features,
            2 + pitch_class + pitch_octave + 2,
        )

    def forward(self, x):
        out = self.effnet(x)
        onset_logits = out[:, 0]
        offset_logits = out[:, 1]
        pitch_out = out[:, 2:]
        pitch_octave_logits = pitch_out[:, : self.pitch_octave + 1]
        pitch_class_logits = pitch_out[:, self.pitch_octave + 1 :]
        return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits
