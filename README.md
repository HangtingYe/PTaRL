# PTaRL: Prototype-based Tabular Representation Learning via Space Calibration

## Official implementation of the experiments in the [PTaRL paper](https://openreview.net/forum?id=G32oY4Vnm8&noteId=G32oY4Vnm8).

## Instructions

Run PTaRL with MLP as backbone on CA and JA datasets by:
* python train_final_version.py --model_type MLP_ot --dataname california_housing
* python train_final_version.py --model_type MLP_ot --dataname jannis

## Citing the paper
If our work is useful for your own, you can cite us with the following BibTex entry:

    @inproceedings{
    ye2024ptarl,
    title={{PT}a{RL}: Prototype-based Tabular Representation Learning via Space Calibration},
    author={Hangting Ye and Wei Fan and Xiaozhuang Song and Shun Zheng and He Zhao and Dan dan Guo and Yi Chang},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=G32oY4Vnm8}
    }
