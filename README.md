# Eliciting Anticipatory Latent Dynamics in Robot Policies via Next-Chunk Prediction

**CNCPolicy** is a cascaded next-chunk prediction framework that elicits anticipatory
latent dynamics purely from action supervision, without explicit future supervision in
pixel or latent space. An Online Action Head jointly produces the current action chunk
and an evolved latent state, on which a cascaded Unrolled Action Head forecasts the next
chunk so that the resulting loss shapes the latent dynamics of the online policy. The
unroll operates only during training, making CNCPolicy plug-and-play with zero
inference-time overhead.

This directory is the MetaWorld imitation-learning release: joint imitation learning on
50 MetaWorld tasks.


### Installation

```bash
conda create -n cncpolicy python=3.10 -y
conda activate cncpolicy

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

pip install transformers==5.0.0rc0 omegaconf accelerate h5py opencv-python

# MetaWorld / MuJoCo
pip install metaworld==3.1.1 mujoco gymnasium

pip install scipy matplotlib openpyxl imageio tqdm
```


## Dataset

The training data (HDF5) is hosted on ModelScope:
[yangfan97/CNCPolicy-Metaworld-Data-hdf5](https://modelscope.cn/datasets/yangfan97/CNCPolicy-Metaworld-Data-hdf5)

```bash
pip install modelscope

# (optional) move the cache to a data disk
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache

# parallel download
modelscope download --dataset yangfan97/CNCPolicy-Metaworld-Data-hdf5 --max-workers 10

# unzip all archives into the current directory
cd $MODELSCOPE_CACHE/datasets/yangfan97/CNCPolicy-Metaworld-Data-hdf5
for f in *.zip; do unzip -q -o "$f"; done
```

Data pulled from ModelScope needs a one-time cleanup pass (outlier removal and
stationary-step filtering):

```bash
python utils/calc_stat_remove_outlier.py
python utils/clean_dataset_stationary.py
```

Then set `dataset.dataset_dir` in `configs/metaworld.yaml` to the path of your
unpacked dataset root. To collect your own demonstrations instead, run
`python scripts/gen_data_hdf5.py`.

## Preparation

1. **DINOv3 weights**: download `dinov3-vitb16-pretrain-lvd1689m` into
   `./dinov3_pretrain/` (`model.vision_encoder.checkpoint_path` in the config
   points to it). 
2. **Demonstration data**: see the **Dataset** section above; make sure
   `dataset.dataset_dir` in `configs/metaworld.yaml` points to your dataset root
3. **Normalization stats**: `python utils/compute_norm_stats_metaworld.py`
   (`utils/stat-metaworld.json` is included and can be reused directly if the
   data matches)

## Training

```bash
./run_train_ema.sh all 40       # early stop after epoch 40 (decay schedule still spans 60)
./run_train_ema.sh resume <ckpt>  # resume training (incl. EMA state)
```

Checkpoints are saved to `checkpoints_vla/ema_ig_all_<timestamp>/` and contain
`model_state_dict` and `ema_state_dict`; saved every 5 epochs, with the training
`config.yaml` stored alongside.

Key config of (`configs/metaworld.yaml` → `training`):
`use_next_chunk_pred: true` (next-chunk prediction), `lambda_next_chunk: 0.8`,
`use_ema: true`, `ema_decay: 0.99`, `batch_size: 128`,
`common.action_chunk_size: 8` (first 8 steps executed at inference).

## Evaluation

```bash
./run_train_ema.sh eval checkpoints_vla/<run>/checkpoint_epoch_40.pt all 50
# or:
python scripts/eval_metaworld_parallel.py \
    --config <ckpt_dir>/config.yaml --checkpoint <ckpt.pt> \
    --task all --num-episodes 50 --num-envs 50
```
