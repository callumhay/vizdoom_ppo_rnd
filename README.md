# vizdoom_ppo_rnd
A doom agent that can beat levels in vizdoom using a combination of Proximal Policy Optimization (PPO) and Random Network Distillation (RND).




[Installing Vizdoom](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/Building.md)

## Optional Dependancies

**Wandb**

For online tracking information (integrates seemlessly with the Tensorboard data tracking). Pass the command line arg `--track` to activate it.
```bash
conda install -c conda-forge wandb
```
