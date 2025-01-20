# Cosmos事後学習手順
## 学習用データの準備
MP4かつ720Pの動画を含むフォルダを作成する
nuscenesは1600x900なので，4/5に圧縮する？
仮想環境を作成
```bash
python3 -m venv make_data
source make_data/bin/activate
pip install nuscenes-devkit
pip install "numpy<2"
```
`make_data.ipynb`を実行
## dockerコンテナのダウンロードとスタート
```bash
docker run --ipc=host -it --gpus=all -v /data/home/kaneyoshi.hiratsuka/SourceCode/cosmos_ft/Cosmos:/workspace/Cosmos nvcr.io/nvidia/nemo:cosmos.1.0 bash
mkdir hf_home
```
## 環境変数の設定
```bash
export HF_TOKEN="<your/HF/access/token>"
export HF_HOME="/workspace/hf_home"
```
## モデルのダウンロード
先に[こちらのページ](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)で利用規約をsubmitしておく．
```bash
cd Cosmos
python cosmos1/models/autoregressive/nemo/download_autoregressive_nemo.py
```

## データの前処理
```bash
export HF_TOKEN="<your/HF/access/token>"
export HF_HOME="/workspace/hf_home"
export RAW_DATA="/workspace/Cosmos/nuscenes_mp4"
export OUTPUT_PREFIX="./indexed_videos"
```
```bash
cd /workspace/Cosmos
git config --global --add safe.directory /workspace/Cosmos
git lfs pull --include=$RAW_DATA

python cosmos1/models/autoregressive/nemo/post_training/prepare_dataset.py \
--input_videos_dir $RAW_DATA \
--output_prefix $OUTPUT_PREFIX
```
## ポストトレーニング
```bash
export HF_TOKEN="<your/HF/access/token>"
export HF_HOME="/workspace/hf_home"
# Number of GPU devices available for post-training. At least 2 for 4B and 8 for 12B.
export NUM_DEVICES=2

# Optionally, you can monitor training progress with Weights and Biases (wandb).
export WANDB_API_KEY="5767e2baca4de66a67547e79fdf0e61f3be358bd"
export WANDB_PROJECT_NAME="cosmos-ft"
export WANDB_RUN_ID="cosmos_autoregressive_4b_finetune"
```
```bash
torchrun --nproc-per-node $NUM_DEVICES cosmos1/models/autoregressive/nemo/post_training/general.py \
--data_path $OUTPUT_PREFIX \
--split_string 4,1,1 \
--log_dir ./logs \
--max_steps 100 --save_every_n_steps 10 \
--tensor_model_parallel_size $NUM_DEVICES \
--model_path nvidia/Cosmos-1.0-Autoregressive-4B
```
## 推論
```bash
export HF_TOKEN="<your/HF/access/token>"
export HF_HOME="/workspace/hf_home"
# Inference with post-trained model.
export NEMO_CHECKPOINT=./logs/default/checkpoints/epoch\=0-step\=19
# Path to the the mp4 file (In git-lfs)
export INPUT_DATA=/workspace/Cosmos/nuscenes_mp4/output0.mp4
```
```bash
cd /workspace/Cosmos
git lfs pull $INPUT_DATA

# change --ar_model_dir to a post-trained checkpoint under ./logs/default/checkpoints/
NVTE_FLASH_ATTN=1 \
NVTE_FUSED_ATTN=0 \
NVTE_UNFUSED_ATTN=0 \
torchrun --nproc-per-node 1 cosmos1/models/autoregressive/nemo/inference/general.py \
--input_image_or_video_path $INPUT_DATA \
--video_save_name "Cosmos-1.0-Autoregressive-4B.mp4" \
--ar_model_dir "$NEMO_CHECKPOINT"
```