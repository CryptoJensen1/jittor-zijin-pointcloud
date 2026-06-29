# 点云降噪赛题 Baseline

## 环境安装
```bash
# 安装计图
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 -y # 确保gcc、g++版本不高于10
conda install -c conda-forge libgomp -y # 确保OpenMP runtime存在

# 安装依赖
python -m pip install -r requirements.txt
pip install jittor numpy trimesh scipy omegaconf point-cloud-utils
```

## 数据准备
1. 将训练数据 `dataset_train.tar.gz` 解压到本目录下：
   ```bash
   tar xzf dataset_train.tar.gz
   ```
   解压后目录：`dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj`

2. 将测试数据 `dataset_test_noisy.zip` 解压到本目录下：
   ```bash
   unzip dataset_test_noisy.zip
   ```
   解压后目录：`dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy`

## 训练
```bash
python run.py --task configs/task/train_vm.yaml
```
训练权重保存在 `experiments/` 目录下。

## 推理（生成提交文件）
修改 `configs/task/predict_vm.yaml` 中的 `load_ckpt` 为你的最佳权重路径，然后运行：
```bash
python run.py --task configs/task/predict_vm.yaml
```
降噪结果保存在 `results/` 目录下，格式为 `.npy` (float32, shape (N,3))。

## 打包提交
```bash
cd results/dataset_test_noisy
zip -r ../../result.zip shapenet/
```

## 提交格式
每个测试样本一个 `denoised.npy`，目录结构与测试集一致，打包为 `result.zip`：
```
result.zip
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy    # np.float32, shape (N, 3)
```

## DRD 扩散模型推理

基于 TVCG 2026 "Deterministic Point Cloud Diffusion for Denoising" (DRD) 的改进实验。

### 配置
```bash
# 修改 configs/task/predict_drd.yaml 中的 load_ckpt 指向 DRD checkpoint
python run.py --task predict_drd
```

### 模型文件
| 文件 | 说明 |
|------|------|
| `src/model/drd_net.py` | DRD 扩散模型（DrdDenoiseNet + DrdDecoder） |
| `src/model/time_emb.py` | 时间嵌入模块（正弦位置编码 + MLP） |
| `configs/model/drd.yaml` | DRD 模型配置（T=30, schedule=decreased） |
| `configs/task/predict_drd.yaml` | DRD 推理配置 |

## 实验版本历史

### 点云去噪（正赛赛道二）

| 版本 | 核心改动 | CD | P2S | 总分 | 说明 |
|------|---------|-----|------|------|------|
| Baseline | VM 100ep，单步推理 | 47.94 | 75.03 | 61.48 | 官方基线 |
| V3 (最佳) | VM 200ep + Warmup Cosine LR | 50.19 | 75.00 | **62.59** | 训练侧改进，稳定有效 |
| V2 | 多步x3 + patch 2000 | 33.81 | 84.45 | 59.13 | 推理侧：迭代过冲 |
| V4 | 单步 + patch 2000 | 48.99 | 72.77 | 60.88 | 大patch单步无增益 |
| DM | V3 + DistanceModule | 36.63 | 54.23 | 45.43 | Jittor兼容性受限 |
| Fusion | 三尺度KNN平均 | 50.09 | 74.80 | 62.44 | 三尺度结果一致 |
| V6 | 5角度旋转投票 | 49.49 | 74.28 | 61.89 | 旋转对去噪影响小 |
| V8 | 10步Langevin | 49.17 | 74.13 | 61.65 | 精细步长无增益 |
| **DRD** | **30步扩散，SCnet DCU训练254ep** | **33.81** | **84.45** | **59.13** | **扩散范式探索：P2S显著提升，CD受限于训练不完整** |

### 点云分类（热身赛二）

| 版本 | 核心改进 | A榜准确率 | 说明 |
|------|---------|-----------|------|
| Baseline | 基础PCT | 84.89% | 基线 |
| V2 | 受控增强+2048pt+Label Smooth | 86.30% | 核心改进 |
| V3 (最佳) | V2 + 10次旋转投票 | **86.71%** | 投票推理 |

## 本地评测（需要 GT 数据，仅组委会持有）
```bash
python evaluate.py \
    --pred_dir ./results/dataset_test_noisy \
    --gt_dir ./test_gt \
    --noisy_dir ./dataset_test_noisy \
    --mesh_dir ./dataset_train \
    --workers 8
```
