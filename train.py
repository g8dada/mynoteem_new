import os
import random
from datetime import datetime
import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from onsets_and_frames import *
from onsets_and_frames.dataset import EMDATASET
from torch.nn import DataParallel
from onsets_and_frames.transcriber import load_weights


def set_diff(model, diff=True):
    for p in model.parameters():
        p.requires_grad = diff


ex = Experiment('train_transcriber')


@ex.config
def config():
    adapter_mode = False  # Set True to use EffNetb0 (AST) as the EM backbone instead of OnsetsAndFrames
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S') + f'-adapter_{adapter_mode}' # ckpts and midi will be saved here
    transcriber_ckpt = '/data/hakka/singing_transcription_ICASSP2021/AST/models/1005_e_4' if adapter_mode else 'ckpts/model_64.pt'
    multi_ckpt = False # Flag if the ckpt was trained on pitch only or instrument-sensitive. The provided checkpoints were trained on pitch only.
    visualize = True  # save piano-roll PNGs and notes_log.json each epoch
    gt_json_path = '/data/hakka/singing_transcription_ICASSP2021/MIR-ST500_20210206/MIR-ST500_corrected.json'
    max_groups = None  # limit # of training groups (None = all); set small for fast change-validation runs
    random_seed = 42   # pin all RNGs so A/B runs differ only by the intended change, not data order

    # Adapter EM-loop knobs. Exported to env vars in train() so the E-step (dataset) and M-step
    # (transcriber) pick them up; defaults fall back to the env var for backward compat; and being
    # config params they are logged to config.json and settable via `with DURATION_DTW=1`.
    ONSET_ONLY     = int(os.environ.get('ONSET_ONLY', 0))         # 1 = train only onset head (zero offset/pitch losses) — diagnostic
    USE_FOCAL      = int(os.environ.get('USE_FOCAL', 0))           # 1 = focal onset/offset loss; 0 = pos-weighted BCE (adapter only)
    POS_WEIGHT     = float(os.environ.get('POS_WEIGHT', 20))       # BCE positive weight for onset/offset when USE_FOCAL=0 (adapter only)
    FOCAL_ALPHA    = float(os.environ.get('FOCAL_ALPHA', 0.4))     # onset/offset focal-loss alpha when USE_FOCAL=1 (adapter only)
    DURATION_DTW   = int(os.environ.get('DURATION_DTW', 0))        # 1 = duration-aware DTW offsets (adapter only)
    DURATION_DTW_W = float(os.environ.get('DURATION_DTW_W', 0.5))  # sustain weight in the DTW feature (adapter only)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_interval = 1 # how often to save checkpoint
    batch_size = 8
    sequence_length = SEQ_LEN #if HOP_LENGTH == 512 else 3 * SEQ_LEN // 4

    iterations = None # per epoch; None = auto (~1 pass/epoch based on dataset size). Full set -> 1592.
    learning_rate = 0.00001
    learning_rate_decay_steps = 10000
    clip_gradient_norm = False #3
    epochs = 15

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, checkpoint_interval, batch_size, sequence_length, learning_rate, learning_rate_decay_steps,
          clip_gradient_norm, epochs, transcriber_ckpt, multi_ckpt, adapter_mode, visualize, gt_json_path, max_groups,
          random_seed, ONSET_ONLY, USE_FOCAL, POS_WEIGHT, FOCAL_ALPHA, DURATION_DTW, DURATION_DTW_W):

    print_config(ex.current_run)
    os.makedirs(logdir, exist_ok=True)

    # Pin every RNG so two runs (e.g. duration-DTW on vs off) differ ONLY by the intended change,
    # not by random data order. Seed the global `random` BEFORE building the dataset (EMDATASET
    # does random.shuffle(self.data) at load time). The DataLoader shuffle is pinned per-epoch below.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    print(f'[seed] pinned RNGs to {random_seed} (dataset data-order, dropout, DataLoader shuffle)')

    # Bridge the adapter knobs to the env vars that the E-step (dataset.update_pts) and M-step
    # (AMTAdapter.run_on_batch) read internally — so `python train.py with DURATION_DTW=1` works
    # and the values are logged in config.json. (No effect in non-adapter mode: both are _is_adapter-gated.)
    os.environ['ONSET_ONLY']     = str(ONSET_ONLY)
    os.environ['USE_FOCAL']      = str(USE_FOCAL)
    os.environ['POS_WEIGHT']     = str(POS_WEIGHT)
    os.environ['FOCAL_ALPHA']    = str(FOCAL_ALPHA)
    os.environ['DURATION_DTW']   = str(DURATION_DTW)
    os.environ['DURATION_DTW_W'] = str(DURATION_DTW_W)
    print(f'[adapter knobs] ONSET_ONLY={ONSET_ONLY} USE_FOCAL={USE_FOCAL} POS_WEIGHT={POS_WEIGHT} '
          f'FOCAL_ALPHA={FOCAL_ALPHA} DURATION_DTW={DURATION_DTW} DURATION_DTW_W={DURATION_DTW_W}')

    # data_name = 'mirst500_15sec_data_full_quantized'
    data_name = 'mirst500_15sec_data_full_zeroshot'
    train_data_path = f'/data/hakka/mynoteem_new/data/{data_name}_NoteEM_audio'
    labels_path = f'/data/hakka/mynoteem_new/data/{data_name}_NoteEM_tsv'

    os.makedirs(labels_path, exist_ok=True)

    train_groups = sorted([g for g in os.listdir(labels_path) if os.path.isdir(os.path.join(labels_path, g))])
    if max_groups:
        train_groups = train_groups[:max_groups]
        print(f'[max_groups] limiting to {len(train_groups)} groups for fast testing')

    if visualize:
        viz_dir = os.path.join(logdir, 'viz')
        viz_keys = {
            os.path.join(train_data_path, f'{g}#0', f'{g}#0.flac')
            for g in train_groups[:10]
        }
        notes_json_path = os.path.join(logdir, 'notes_log.json')
    else:
        viz_dir = None
        viz_keys = None
        notes_json_path = None

    conversion_map = None
    # adapter_mode uses pitch-only (no instrument channels) — force empty instrument map
    instrument_map = [] if adapter_mode else None
    dataset = EMDATASET(audio_path=train_data_path,
                           labels_path=labels_path,
                           groups=train_groups,
                            sequence_length=sequence_length,
                            seed=random_seed,
                           device=DEFAULT_DEVICE,
                            instrument_map=instrument_map,
                            conversion_map=conversion_map
                        )
    print('len dataset', len(dataset), len(dataset.data))

    # Tie iterations to dataset size so subset runs keep the SAME per-epoch intensity as the
    # full-set default (1592 iters * 8 / 12738 clips ~= 1 pass/epoch). Prevents the silent
    # oversampling footgun (e.g. max_groups=60 + iterations=1592 => 19x passes/epoch -> onset
    # collapse). Explicit iterations still honored, but a far-from-1x value is warned loudly.
    if iterations is None:
        iterations = max(1, len(dataset) // batch_size)
        print(f'[iterations] auto={iterations} (~1 pass/epoch over {len(dataset)} clips)')
    else:
        passes_per_epoch = iterations * batch_size / max(1, len(dataset))
        print(f'[iterations] explicit={iterations} => {passes_per_epoch:.1f} passes/epoch over {len(dataset)} clips')
        if not (0.5 <= passes_per_epoch <= 2.0):
            print(f'[WARN] {passes_per_epoch:.1f} passes/epoch is far from ~1; oversampling can collapse '
                  f'the onset signal (see the 19x-oversampling failure). Consider iterations=None (auto).')

    #####
    if adapter_mode:
        from onsets_and_frames.ast_model import EffNetb0
        from onsets_and_frames.transcriber import AMTAdapter
        backbone = EffNetb0()
        backbone.load_state_dict(
            torch.load(transcriber_ckpt, map_location='cpu', weights_only=False),
            strict=False,
        )
        transcriber = AMTAdapter(backbone).to(device)
        # Freeze the early encoder blocks; keep later blocks and classifier trainable.
        set_diff(transcriber.backbone.effnet.conv_stem, False)
        set_diff(transcriber.backbone.effnet.blocks[:4], False)
    elif not multi_ckpt:
        model_complexity = 64 if '64' in transcriber_ckpt else 48
        saved_transcriber = torch.load(transcriber_ckpt).cpu()
        # We create a new transcriber with N_KEYS classes for each instrument:
        transcriber = OnsetsAndFrames(N_MELS, (MAX_MIDI - MIN_MIDI + 1),
                                              model_complexity,
                                    onset_complexity=1., n_instruments=len(dataset.instruments) + 1).to(device)
        # We load weights from the saved pitch-only checkkpoint and duplicate the final layer as an initialization:
        load_weights(transcriber, saved_transcriber, n_instruments=len(dataset.instruments) + 1)
    else:
        # The checkpoint is already instrument-sensitive
        transcriber = torch.load(transcriber_ckpt).to(device)

    if not adapter_mode:
        # We recommend to train first only onset detection. This will already give good note durations because the combined stack receives
        # information from the onset stack
        set_diff(transcriber.frame_stack, False)
        set_diff(transcriber.offset_stack, False)
        set_diff(transcriber.combined_stack, False)
        set_diff(transcriber.velocity_stack, False)

    parallel_transcriber = DataParallel(transcriber, device_ids=[0])
    optimizer = torch.optim.Adam(list(transcriber.parameters()), lr=learning_rate, weight_decay=1e-5)
    transcriber.zero_grad()
    optimizer.zero_grad()
    for epoch in range(1, epochs + 1):
        print('epoch', epoch)
        if epoch > 1:
            del loader
            del batch
        torch.cuda.empty_cache()

        POS = 1.1 # Pseudo-label positive threshold (value > 1 means no pseudo label).
        NEG = -0.1 # Pseudo-label negative threshold (value < 0 means no pseudo label).
        with torch.no_grad():
            dataset.update_pts(parallel_transcriber,
                               POS=POS,
                               NEG=NEG,
                               to_save=logdir + '/alignments',
                               first=epoch == 1,
                               update=True,
                               BEST_BON=epoch > 1,
                               viz_dir=viz_dir,
                               viz_keys=viz_keys,
                               viz_tag=f'epoch{epoch:02d}',
                               gt_json_path=gt_json_path,
                               notes_json_path=notes_json_path,
                               )
        # per-epoch generator seeded by (random_seed, epoch): the shuffle for each epoch depends
        # only on these, so it is identical across runs regardless of other RNG consumption.
        loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(random_seed + epoch))

        total_loss = []
        transcriber.train()
        if adapter_mode:
            # Keep the pretrained backbone's BatchNorm in eval() during the M-step:
            #   - running_mean/var stay fixed (no drift under noisy pseudo-labels,
            #     and no incoherent stat updates on the weight-frozen early blocks),
            #   - normalization matches the E-step (which also runs in eval), so we
            #     optimize exactly the model that generated the labels.
            # Weights and affine gamma/beta still train; dropout stays active.
            for module in transcriber.backbone.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()

        onset_total_tp = 0.
        onset_total_pp = 0.
        onset_total_p = 0.

        torch.cuda.empty_cache()

        loader_cycle = cycle(loader)
        for _ in tqdm(range(iterations)):
            batch = next(loader_cycle)
            optimizer.zero_grad()

            n_weight = 20 if adapter_mode else (1 if HOP_LENGTH == 512 else 2)  # decoupled onset: ~neg:pos ratio
            transcription, transcription_losses = transcriber.run_on_batch(batch, parallel_transcriber,
                                                                           positive_weight=n_weight,
                                                                           inv_positive_weight=n_weight,
                                                                           )
            onset_threshold = 0.05 if adapter_mode else 0.5
            onset_pred = transcription['onset'].detach() > onset_threshold
            onset_total_pp += onset_pred
            onset_tp = onset_pred * batch['onset'].detach()
            onset_total_tp += onset_tp
            onset_total_p += batch['onset'].detach()

            onset_recall = (onset_total_tp.sum() / onset_total_p.sum()).item()
            onset_precision = (onset_total_tp.sum() / onset_total_pp.sum()).item()

            pitch_onset_recall = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_p[..., -N_KEYS:].sum()).item()
            pitch_onset_precision = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_pp[..., -N_KEYS:].sum()).item()

            transcription_loss = sum(transcription_losses.values())
            loss = transcription_loss
            # AMTAdapter.run_on_batch calls backward() per clip internally;
            # returned losses are detached scalars, so grad_fn is None.
            if loss.grad_fn is not None:
                loss.backward()

            if clip_gradient_norm:
                clip_grad_norm_(transcriber.parameters(), clip_gradient_norm)

            optimizer.step()
            total_loss.append(loss.item())
            print('loss:', sum(total_loss) / len(total_loss), 'Onset Precision:', onset_precision, 'Onset Recall', onset_recall,
                                                            'Pitch Onset Precision:', pitch_onset_precision, 'Pitch Onset Recall', pitch_onset_recall)

        save_condition = epoch % checkpoint_interval == 0
        if save_condition:
            torch.save(transcriber, os.path.join(logdir, 'transcriber_{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
            torch.save({'instrument_mapping': dataset.instruments},
                       os.path.join(logdir, 'instrument_mapping.pt'.format(epoch)))


