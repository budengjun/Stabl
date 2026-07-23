# STABL 实验 A 与实验 B 直接运行指南

## 1. 文件放置

把整个 `stabl_experiments_A_B` 文件夹放到 STABL repository 根目录。目录建议如下：

```text
Stabl/
  stabl/
  setup.py
  Sample Data/
  stabl_experiments_A_B/
```

进入 repository 根目录并激活你现有的 `stabl` 环境：

```bash
cd /data/yhu94/Stabl
source /path/to/stabl/bin/activate
```

让 Python 同时找到本地 STABL package 和新实验代码：

```bash
export PYTHONPATH="$PWD:$PWD/stabl_experiments_A_B:$PYTHONPATH"
```

先检查环境：

```bash
python stabl_experiments_A_B/check_environment_ab.py
```

Gaussian 实验需要 `knockpy`。PLSKO 实验还需要 `Rscript` 和官方 R package `PLSKO`。

## 2. 实验 A 的设计

实验 A 比较四种 completion：

```text
oracle_complete
median
exact_gaussian_posterior
bayesianridge_posterior
```

四条分支严格使用同一预处理顺序：

```text
completion
fit scaler on completed X
standardize X
generate knockoff
no post generation scaling
```

每个 replicate 和 completion 只生成一次 completed X。不同 knockoff draw 只改变 knockoff seed。

同时比较三种选择规则：

```text
stabl_min
stabl_q
knockoff_plus
```

### 2.1 两个 replicate 的 smoke test

```bash
mkdir -p logs results

python -u stabl_experiments_A_B/run_experiment_a_oracle_completion.py \
  --out-dir results/experiment_A_smoke \
  --n-replicates 2 \
  --n 150 \
  --p 200 \
  --n-signal 10 \
  --block-size 25 \
  --rho 0.60 \
  --task classification \
  --signal-strength 1.0 \
  --mechanisms MCAR,MAR \
  --missing-rate 0.20 \
  --completion-methods oracle_complete,median,exact_gaussian_posterior,bayesianridge_posterior \
  --generators gaussian_equicorrelated \
  --n-knockoff-draws 1 \
  --n-bootstraps 20 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --skip-precision-recovery \
  --overwrite \
  2>&1 | tee logs/experiment_A_smoke.log
```

运行成功后应出现：

```text
results/experiment_A_smoke/selection_results.csv
results/experiment_A_smoke/draw_diagnostics.csv
results/experiment_A_smoke/imputation_recovery.csv
results/experiment_A_smoke/summary.csv
```

### 2.2 正式实验 A

```bash
nice -n 10 python -u stabl_experiments_A_B/run_experiment_a_oracle_completion.py \
  --out-dir results/experiment_A_main \
  --n-replicates 100 \
  --n 150 \
  --p 200 \
  --n-signal 10 \
  --block-size 25 \
  --rho 0.60 \
  --task classification \
  --signal-strength 1.0 \
  --mechanisms MCAR,MAR \
  --missing-rate 0.20 \
  --completion-methods oracle_complete,median,exact_gaussian_posterior,bayesianridge_posterior \
  --generators gaussian_equicorrelated \
  --n-knockoff-draws 1 \
  --n-bootstraps 50 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  2>&1 | tee logs/experiment_A_main.log
```

推荐在 tmux 中运行：

```bash
tmux new -s stabl_A
```

进入 tmux 后执行上面的正式命令。离开但不中断：

```text
Ctrl+B
D
```

重新进入：

```bash
tmux attach -t stabl_A
```

### 2.3 中断后继续

使用完全相同的科学参数，把 `--overwrite` 换成 `--resume`：

```bash
python -u stabl_experiments_A_B/run_experiment_a_oracle_completion.py \
  --out-dir results/experiment_A_main \
  --n-replicates 100 \
  --n 150 \
  --p 200 \
  --n-signal 10 \
  --block-size 25 \
  --rho 0.60 \
  --task classification \
  --signal-strength 1.0 \
  --mechanisms MCAR,MAR \
  --missing-rate 0.20 \
  --completion-methods oracle_complete,median,exact_gaussian_posterior,bayesianridge_posterior \
  --generators gaussian_equicorrelated \
  --n-knockoff-draws 1 \
  --n-bootstraps 50 \
  --target-fdr 0.10 \
  --n-jobs 8 \
  --resume \
  2>&1 | tee -a logs/experiment_A_main.log
```

### 2.4 实验 A 后续敏感性分析

在主实验完成后，再分别改变一个维度：

```text
rho = 0.2, 0.6, 0.8
missing_rate = 0.1, 0.2, 0.4
p = 100, 200, 500
signal_strength = 0.5, 1.0, 1.5
```

每个配置使用新的 `--out-dir`。

## 3. 实验 B 的设计

实验 B 使用真实 omic covariate matrix，并人工生成 sparse outcome，因此 true support 已知。

同一套 STABL score paths 同时用于：

```text
original STABL minimum FDP plus
Stabl_q target FDR threshold
standard knockoff plus threshold
```

### 3.1 SSI Gaussian baseline smoke test

SSI 每行对应一个患者，适合作为主要 semi synthetic benchmark。SSI 有 native missingness，所以这里先用固定 median completion，只比较 generator 和 selection rule。

```bash
python -u stabl_experiments_A_B/run_experiment_b_semisynthetic.py \
  --dataset ssi \
  --data-path "Sample Data/Biobank SSI" \
  --omic CyTOF \
  --complete-strategy median \
  --p-values 250 \
  --out-dir results/experiment_B_SSI_smoke \
  --n-replicates 2 \
  --n-signal 10 \
  --task classification \
  --signal-strength 1.0 \
  --generators gaussian_equicorrelated,gaussian_mvr \
  --n-knockoff-draws 1 \
  --n-bootstraps 20 \
  --target-fdr 0.05 \
  --n-jobs 8 \
  --overwrite \
  2>&1 | tee logs/experiment_B_SSI_smoke.log
```

### 3.2 SSI Gaussian 正式 baseline

```bash
nice -n 10 python -u stabl_experiments_A_B/run_experiment_b_semisynthetic.py \
  --dataset ssi \
  --data-path "Sample Data/Biobank SSI" \
  --omic CyTOF \
  --complete-strategy median \
  --p-values 250,500,full \
  --out-dir results/experiment_B_SSI_gaussian \
  --n-replicates 50 \
  --n-signal 10 \
  --task classification \
  --signal-strength 1.0 \
  --generators gaussian_equicorrelated,gaussian_mvr \
  --n-knockoff-draws 1 \
  --n-bootstraps 50 \
  --target-fdr 0.05 \
  --n-jobs 8 \
  2>&1 | tee logs/experiment_B_SSI_gaussian.log
```

### 3.3 OOL iid aligned benchmark

OOL 有 repeated measurements。主分析建议每位受试者只保留第一行：

```bash
python -u stabl_experiments_A_B/run_experiment_b_semisynthetic.py \
  --dataset ool \
  --data-path "Sample Data" \
  --omic CyTOF \
  --subject-mode first \
  --complete-strategy drop-columns \
  --p-values 250,500,full \
  --out-dir results/experiment_B_OOL_first \
  --n-replicates 50 \
  --n-signal 8 \
  --task classification \
  --signal-strength 1.0 \
  --generators gaussian_equicorrelated,gaussian_mvr \
  --n-knockoff-draws 1 \
  --n-bootstraps 50 \
  --target-fdr 0.05 \
  --n-jobs 8 \
  2>&1 | tee logs/experiment_B_OOL_first.log
```

`--subject-mode all` 只作为 repeated measures stress test，不作为 PLSKO 论文复现主结论。

## 4. PLSKO 调参实验

先确认官方 R package：

```bash
Rscript -e "stopifnot(requireNamespace('PLSKO', quietly=TRUE))"
```

复制示例配置：

```bash
cp stabl_experiments_A_B/plsko_generator_configs.example.json \
   stabl_experiments_A_B/plsko_generator_configs.json
```

第一轮不要直接跑 50 replicates 和 full p。先在 p=250 下用 10 到 20 replicates 筛选配置：

```bash
python -u stabl_experiments_A_B/run_experiment_b_semisynthetic.py \
  --dataset ssi \
  --data-path "Sample Data/Biobank SSI" \
  --omic CyTOF \
  --complete-strategy median \
  --p-values 250 \
  --out-dir results/experiment_B_SSI_plsko_screen \
  --n-replicates 20 \
  --n-signal 10 \
  --task classification \
  --signal-strength 1.0 \
  --generator-configs-json stabl_experiments_A_B/plsko_generator_configs.json \
  --n-knockoff-draws 1 \
  --n-bootstraps 50 \
  --target-fdr 0.05 \
  --n-jobs 8 \
  2>&1 | tee logs/experiment_B_SSI_plsko_screen.log
```

查看 `summary.csv`，筛出 empirical FDR 接近或低于 0.05 且 power 最高的一个或两个配置。然后用精简后的 JSON 跑 50 replicates 和 p=500。

## 5. 结果判读

实验 A 主要看：

```text
summary.csv
imputation_recovery.csv
```

核心列：

```text
empirical_fdr
power_mean
fdp_exceedance_probability
masked_rmse
sample_cov_fro_relative
sample_corr_fro_relative
ridge_precision_fro_relative
```

实验 B 主要看：

```text
summary.csv
```

先比较同一 generator 下三种 selection rule，再比较同一 selection rule 下不同 generator。

最关键的判定：

```text
standard knockoff plus 控制 FDR，但 stabl_min 不控制
```

这支持 threshold 是 STABL 的主要问题。

```text
oracle complete 与 exact posterior 控制 FDR，BayesianRidge posterior 不控制
```

这支持 approximate completion model misspecification 是主要问题。

```text
同一 selection rule 下 PLSKO 控制 FDR，而 Gaussian 不控制
```

这支持 generator 是主要问题。

## 6. 当前不要做的事

暂时不要加入 derandomized aggregation。先证明单 draw procedure 的 empirical FDR 基本受控。Derandomization 可以减少有效 procedure 的随机性，但不能修复 invalid knockoff pair。

# 缓存复用与避免重复计算

详细说明见 `CACHE_REUSE_GUIDE.md`。

建议为所有新实验使用一个长期保留的 pair cache：

```bash
mkdir -p /data/yhu94/stabl_pair_cache
```

Experiment A 命令增加：

```bash
--pair-cache-dir /data/yhu94/stabl_pair_cache
```

Experiment B 默认使用共享 knockoff pair bank。建议至少使用 5 个 pair：

```bash
--n-knockoff-draws 5 \
--pair-bank-scope shared \
--pair-cache-dir /data/yhu94/stabl_pair_cache
```

这样 PLSKO 每个 p 和参数配置只生成 5 次，而不是 `n_replicates × 5` 次。

旧 OOL 审计的 score cache 可直接重新套用新阈值规则：

```bash
python -u stabl_experiments_A_B_cached/reanalyze_legacy_ool_scores.py \
  --score-dir stabl_knockoff_experiments/OOL_all_generators_formal_v2/scores \
  --out-csv results/legacy_ool_threshold_reanalysis.csv \
  --target-fdr 0.10 \
  --include-selected-features
```

## V10 external omic generalization

See `RUN_V10_EXTERNAL_GENERALIZATION.md`. Run independent Oracle calibration,
then the SSI Proteomics + DREAM Phylotype external pilot using a fresh seed.


## Generator Track G0 to G3

The integrated PLSKO generator study is documented in `RUN_GENERATOR_TRACK_G0_G3.md`.
For primary PLSKO comparisons, use `--pair-bank-scope per-replicate`; the legacy
shared pair bank is a low cost diagnostic and does not represent generator
variability across independent semi synthetic outcomes.

```bash
bash generator_track/scripts/run_g0_ssi.sh
bash generator_track/scripts/run_g1_ssi_pilot.sh
bash generator_track/scripts/run_g2_ssi_pilot.sh
```
