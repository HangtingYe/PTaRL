<h1 align="center">
[ICLR 2024]PTaRL: Prototype-based Tabular Representation Learning via Space Calibration


Links: [<a href="https://openreview.net/forum?id=G32oY4Vnm8&noteId=G32oY4Vnm8">OpenReview</a>]
</h1>


**📥 Contact Email Update:** Please use **yeht22@mails.jlu.edu.cn** for all future communications. The previous email (yeht2118@mails.jlu.edu.cn) is no longer active.

## 1. Environment Setup

Recommended for readers: create environment from `environment.yml` (designed to work on both local machines and Linux servers, following the same style as LLM_DAS).

```bash
cd PTaRL

conda env remove -n ptarl -y 2>/dev/null || true
conda env create -f environment.yml
conda activate ptarl

# Environment check
python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
```

Expected from the environment check:
- `torch` should be `2.4.1`.
- `transformers` should be `4.46.3`.
- The current training code calls `.cuda()` directly, so running experiments as-is still requires a CUDA-capable server environment even though the environment file itself is cross-platform.
- If `./tokenizer` does not exist, the first run will automatically download `bert-base-uncased` and save it to `./tokenizer`.

If you see `No matching distribution found`:
- Usually this is an index/network issue, not package absence.
- Check with `python -m pip config list`.
- Then retry with an explicit index such as `-i https://pypi.org/simple` or your regional mirror.

## 📚 Instructions

Run PTaRL with MLP as backbone on CA and JA datasets by:
* python train_final_version.py --model_type MLP_ot --dataname california_housing
* python train_final_version.py --model_type MLP_ot --dataname jannis

## 🤗 Citing the paper
If our work is useful for your own, you can cite us with the following BibTex entry:

    @inproceedings{
    ye2024ptarl,
    title={{PT}a{RL}: Prototype-based Tabular Representation Learning via Space Calibration},
    author={Hangting Ye and Wei Fan and Xiaozhuang Song and Shun Zheng and He Zhao and Dan dan Guo and Yi Chang},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=G32oY4Vnm8}
    }
