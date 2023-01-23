# Soft Actor-Critic Discrete in Tensorflow
Soft Actor-Critic is a deep reinforcement learning framework for training maximum entropy policies in discrete domains. The algorithm is based on the paper **[SOFT ACTOR-CRITIC FOR DISCRETE ACTION SETTINGS](https://arxiv.org/abs/1910.07207)**

This repo is implemented based on Tensorflow framwork, and inspried by **[this](https://github.com/toshikwa/sac-discrete.pytorch)** repo.

## Setup
```bash
pip install -r requirements.txt
```
For game environment compilation, please refer to **[submodule repo](https://github.com/djjbxxz/ColorLine_Environment/tree/tensorflow)**.

Don't forget to rename the compilated file to `gen_colorline_data_tensorflow`, with a suffix `.dll` in Windows, and `.so` in Linux.

Then point the module import path in [Env.py](Env.py) to the directory of the env submodule.

## Training
```python3
python3 train.py
```

## Results

