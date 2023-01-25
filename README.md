# Soft Actor-Critic Discrete in Tensorflow
Soft Actor-Critic is a deep reinforcement learning framework for training maximum entropy policies in discrete domains. The algorithm is based on the paper **[SOFT ACTOR-CRITIC FOR DISCRETE ACTION SETTINGS](https://arxiv.org/abs/1910.07207)**

This repo is implemented based on Tensorflow framwork, and inspried by **[this](https://github.com/toshikwa/sac-discrete.pytorch)** repo.

## Setup
```bash
pip install -r requirements.txt
```

Game environment code is written in C++ to boost up training process.

To compile the game env, please refer to the **[submodule repo](https://github.com/djjbxxz/ColorLine_Environment/tree/tensorflow)**.

Don't forget to rename the compilated file to `gen_colorline_data_tensorflow`, with a suffix `.dll` in Windows, and `.so` in Linux.

Then, point the game env module import path in [Env.py](Env.py) to the directory of the game env submodule.


## Training
```python3
python3 train.py
```

## Results

<img width="761" alt="result" src="https://user-images.githubusercontent.com/41796656/213947925-55341859-1d88-4891-ad27-f6fdfb122cfc.png">

Ps: negative rewards mean that the model is making illegal moves that are not allowed by the game rule.


At the begining, the model learnt how to avoid making illgel move (learned the basic game rule) quickly within the first 100k iteration, but after that, the model seems to not making any progess in the following training iteration.

The follwing image shows that the variance of actions made by the model dramaticlly drop as the training process going.
<img width="759" alt="action variance" src="https://user-images.githubusercontent.com/41796656/213948256-156db940-244a-4a57-842f-ab13c7ff4b73.png">


Look at the action distribution collected in the training, the model is more likely to make certain move as the training process going. It is abnormal that a few actions dominate among  all the action possiblity.

<img width="505" alt="action distribution" src="https://user-images.githubusercontent.com/41796656/213948471-d30a53cb-48d7-4c05-9b6d-d7998a0df7d4.png">

Ps: There are 625 possible action in total in the case.

### My guess on what cause this:
1) The entropy is not guiding the training properly.
2) The model is overfitting.

Don't worry! I am trying to fix this! :)

## Other

I am a grad student in AI, and this is my first reinforcement learning project.

The purpose I created this repo is to record my learning process, and document any mistakes I made. 
If you have any advice on this project, please create an issue. Any advice would be appreciated!
