# DreamerV2 PyTorch Implementation

This project is a PyTorch implementation of the DreamerV2 agent, a reinforcement learning algorithm that learns a world model from pixels and uses it to train an actor-critic agent in imagined trajectories. This implementation is based on the paper [Mastering Diverse Domains through World Models](https://arxiv.org/pdf/2010.02193) and was inspired by the original implementation [here](https://github.com/danijar/dreamerv2) and another excellent implementation [here](https://github.com/jsikyoon/dreamer-torch). 

## Project Structure

The repository is organized as follows:

```
.
├── configs/            # YAML configuration files for different environments.
│   ├── bankheist.yaml
│   ├── freeway.yaml
│   └── pong.yaml
├── src/                # Main source code directory.
│   ├── envs/           # Environment wrappers.
│   │   └── atari.py    # Wrapper for Atari environments.
│   ├── models/         # Core neural network models.
│   │   ├── rssm.py     # Recurrent State-Space Model (the world model).
│   │   ├── vision.py   # Encoder and Decoder for image processing.
│   │   ├── actor_critic.py # Actor and Critic networks.
│   │   ├── behavior.py # Logic for training in imagined trajectories.
│   │   └── heads.py    # Reward and Discount predictors.
│   ├── replay/         # Replay buffer implementation.
│   │   └── replay_buffer.py
│   ├── utils/          # Utility functions.
│   │   ├── image_saver.py
│   │   └── tools.py
│   └── train.py        # Main script for training the agent.
...

## Usage

To start training, run the `train.py` script with the desired configuration file.

```bash
python -m src.train --config configs/pong.yaml
```

You can monitor training progress and results if you have `wandb` configured.

## Dependencies

```bash
pip install -r requirements.txt
```
