# SnapKV 复现 — Pythia-70M

基于 kvpress 框架，在 Pythia-70M（GPT-NeoX 架构）上复现 SnapKV 缓存压缩算法，并测试 PPL 与推理吞吐。

## 文件说明

| 文件          | 说明                                    |
| ------------- | --------------------------------------- |
| `SnapNV.py`   | SnapKV 实现，继承 `kvpress.ScorerPress` |
| `run_eval.py` | 评测脚本，支持 PPL 和吞吐两种模式       |

## 环境

```
pip install torch transformers datasets kvpress
```

## 运行

```bash
# PPL 测试（baseline）
python run_eval.py --mode ppl --press none --block-size 128

# PPL 测试（SnapKV，50% 压缩率）
python run_eval.py --mode ppl --press snapkv --compression-ratio 0.5 --block-size 128

# 吞吐测试
python run_eval.py --mode speed --press snapkv --compression-ratio 0.5 --block-size 128
```

## 测试结果

模型：EleutherAI/pythia-70m，数据集：WikiText-2 test，block_size=128

### PPL

| 压缩率        | window_size | PPL    |
| ------------- | ----------- | ------ |
| 0（baseline） | —           | 68.11  |
| 0.3           | 32          | 91.42  |
| 0.5           | 32          | 110.71 |
| 0.5           | 64          | 112.43 |

压缩率越高 PPL 越大，符合预期。Pythia-70M 本身参数量很小，对压缩更敏感。

### 吞吐

| 配置               | tokens/s |
| ------------------ | -------- |
| baseline           | 705      |
| snapkv (ratio=0.5) | 182      |

## 实现要点

- GPT-NeoX 使用融合 QKV 投影（`query_key_value`），原生 kvpress 不支持，需自定义 `_prerope_queries` 做 fallback
- GPT-NeoX 的 attention 层接收 `layer_past` 而非 `past_key_values`，`forward_hook` 需兼容两种命名
- kvpress 的 `cache_position[-1] > q_len` 检查在分块 prefill 场景下会误跳过压缩，改为 `q_len <= 1`
