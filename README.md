# Vizdoom PPO + RND: A General Doom Playing Agent using Proximal Policy Optimization and Random Network Distillation

This code base makes use of [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) and [Random Network Distillation (RND)](https://arxiv.org/pdf/1810.12894.pdf) to enable a Doom agent to beat levels in Doom and Doom 2 through the use of pytorch and the vizdoom environment.

<p style="text-align:center;margin:0 1em" align="center">
<a style="margin:0 auto" href="https://youtube.com/watch?v=mPff0B6wNSs">
<img src="https://img.youtube.com/vi/mPff0B6wNSs/0.jpg" alt="Passing level E1M2" title="Passing level E1M2" width="400">
</a>
</p>

The agent has currently been tested on *Map01 in Doom 2* and *E1M2 in Doom* and is able to learn to beat each level within approximately 6M global steps (across all running environments), this takes around 8 hours on a 12GB Nvidia RTX 3080 with 20 simultaneous environments. More testing is required to get the agent to generalize across further levels. The agent is able to approach perfect play in all basic vizdoom .cfg scenarios in <1M global steps per scenario. For reproducibility, all training was done using a seed of 42, the seed can be set via the command line (e.g., `--seed 42`).

## Getting Started

### Dependancies
All required dependancies can be installed via the environment.yml file via
```bash
conda env create -f environment.yml
```
**Important Note:** You may need to install additional dependancies for vizdoom to run in your environment, [see the vizdoom repository for details](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/Building.md).

You will need to copy your `doom.wad` and `doom2.wad` files into the `bin` directory in order to train on levels in either of those games.

### Optional Dependancies

**Wandb**

Can be used for online tracking information (integrates seemlessly with the Tensorboard data tracking). Pass the command line arg `--track` to activate it.
```bash
conda install -c conda-forge wandb
```

### Running / Training an Agent

To train the agent on doom level E1M2 you can run the following command:
```bash
python doom_ppo_rnd.py --map E1M2 --num-envs 20
```
I strongly recommend changing the `num-envs` to the highest possible number you can fit in your GPU memory for faster training.

*On 12GB of GPU memory, the current implementation is able to run 20 simultaneous training environments. The amount of GPU memory (and the speed of training) should scale linearly with the number of agents.*

If you want to train on a specific vizdoom scenario, a list of them can be found in [doom_gym_wrappers.py](doom_gym_wrappers.py) and they can be enabled using the `--gym-id` flag (e.g., `--gym-id VizdoomCorridor-v0`). Note that each scenario will have specific constraints on the action space of the agent. I've found that the best way to train in these scenarios is to enable multi-discrete action spaces (`--multidiscrete-actions True`).

For example, to run the corridor scenaraio, use the following command:
```bash
python doom_ppo_rnd.py --gym-id VizdoomCorridor-v0 --multidiscrete-actions True
```

###  Model Saving and Loading

During training the model will automatically save at global step intervals, determined by the command line argument `--save-timesteps`. When saving takes place, a model checkpoint file will be generated in a unique run directory under `runs/<gym_env_id>/<unique_run_dir>`. This directory will be created at the start of training and all tensorboard stats data will also be placed and updated in this same directory over the course of training.

To load a saved model for further training you can use the command line argument `--model <path_to_my_saved_model>`.


## Implementation Details

The code makes use of the following networks for each agent:

- A convolutional network to encode the RGB and label buffers as a smaller, latent representation of the visual observation
- Embeddings for the x,y position and the orientation/angle, health, and ammo
- Cropped pixels from the HUD containing the currently held keycards
- RND target and predictor networks (includes a combination of a convolutional network and fully connected layers) for determining the ‘novelty’ of a given observation (RGB pixels, keycard pixels, and location + orientation game variables as one-hot vectors)
- Long-Short Term Memory (LSTM) which is given the output of the convolutional network (latent representation), position, keycard pixels and game variable embeddings
- An actor network which takes the output of the LSTM to produce the distribution of actions for the agent to take
- A critic network that produces a single valued output for each agent to compare against generated rewards

There are many tweaks and gotchas across this implementation. It's difficult to enumerate them all, but here are some of the key details to consider:

- The weighting of the intrinsic vs. extrinsic reward is important, this can be modified via the command line argument `--reward-i-coeff`, which will set the coefficient to the intrinsic reward. By default this is set to a low value (0.01) so that the agent doesn't overweigh the rewards it gets from RND (i.e., intrinsic rewards) vs. the rewards its getting from the environment (i.e., extrinsic rewards)
- Extrinsic reward advantages are double weighted over intrinsic advantages, this can be found in [doom_ppo_rnd.py](doom_ppo_rnd.py), this implementation detail comes directly from [OpenAI's implementation of RND](https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/run_atari.py#L107). For future work, It might be worth playing with this weighting to see if it can be improved for faster training.
- Intrinsic rewards are normalized, whereas extrinsic rewards are *not*. This turned out to be a crucial implementation detail: I've found that normalizing both rewards keeps  the agent from learning properly. This detail is also used by OpenAI in their RND implementation.
- I've choosen a small pixel observation width and height (80,60). I've found that this size is a good trade-off between memory constraints and providing the input for sufficient for learning. I'm positive that the agent could be improved via a larger screenbuffer size, but I haven't been able to test it due to GPU memory constraints. If you plan on changing this, be warned that you should also check to make sure the pixel crop for the keycards is still working as intended (i.e., that it's cropping the correct HUD pixels). This code can be found in [net_utils.py](https://github.com/callumhay/vizdoom_ppo_rnd/blob/01dded87b1661b9a45ed481e6b331b46cdbbf200/net_utils.py#L39).
- I've found that increases to the LSTM input size should result in a proportional change to the hidden size of the LSTM to maintain a 4:1 ratio (input:hidden). In my experience having a hidden size around 1/4 the size of the input has resulted in the best results, though I have not played around with significantly larger hidden size ratios (e.g., > 1/3 the input size) due to GPU memory constriants. This might be worth experimenting with. In my experience, hidden sizes smaller than 1/4 the input size result in significantly worse agent behaviour.
- There is an exploration period before training that is necessary to populate the reward normalization buffers so that it can bootstrap an appropriate variance. The number of exploration steps can be set via the command line argument `--num-explore-steps`, the default is currently 1000.
- The number of steps each environment takes can be set with `--num-steps` and is currently set to 256 by default. This can improve training if set to a larger number, however it comes with the trade-off of more GPU memory.
- The weighting of the RND network loss via `--pt-coef` was determined empirically to be in the ballpark of [0.001, 0.05] and is currently set by default to 0.005. This is probably worth playing around with more to see if the RND predictor network can be trained more effectively in sync with the other networks.
- Extrinsic reward structuring is something I've played around with a lot and have had to revisit many times throughout this project. You can find the structure of these in [doom_general_env_config.py](doom_general_env_config.py) and [doom_reward_vars.py](doom_reward_vars.py). The current reward structure appears to work fairly well:
  - There is currently a small living penalty (i.e., the agent is punished for every step of gameplay), I've found this to be necessary to get the agent to be more expedient and to drive it towards significant rewards (e.g., passing levels). Training may benefit from a larger living penalty, but I haven't tested this with the latest code. Likely the best penalty is one that balances well with the intrinsic reward such that seeing new things overcomes the penalty, but eventually decays back to a penalty once the agent has been exposed to that same situation repeatedly.
  - There is currently a small ammo use penalty, this is tricky to balance: If the penalty is too high the agent will avoid shooting entirely, if it's too low it will waste ammo indiscriminately.
  - The agent currently gets rewarded for doing damage but this also includes damage to environmental objects, specifically barrels (this has to do with how vizdoom events work). As a result, it likes abusing this reward by blowing up barrels for better or worse; don't worry, the agent eventually learns to stop blowing up barrels in its face ;) 
  - Feel free to modify these rewards; however, *I strongly recommend not tying any extrinsic rewards to exploration-based activities* (i.e., don't give rewards or penalties for visiting or avoiding locations/coordinates in the environment). Tying extrinsic rewards to exploration-specific events results in all sorts of poor agent behaviour (aimless circling, stopping/staying, etc.). The code makes use of RND to remedy this problem, all exploration-related reward should be deferred to the RND network whenever possible.


## Thanks and Acknowledgements

Thank you to the OpenAI team (John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, Yuri Burda, Harrison Edwards, Amos Storkey) responsible for the development of both PPO and RND algorithms. Furthermore, their [RND github repo](https://github.com/openai/random-network-distillation) was incredibly useful for this implementation.

Many thanks to Huang, Shengyi; Dossa, Rousslan Fernand Julien; Raffin, Antonin; Kanervisto, Anssi; and Wang, Weixun for their [super-comprehensive implementations and tips for PPO baselines](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/). Their [atari LSTM PPO implementation](https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari_lstm.py) was used as a starting point for this implementation.

Thanks to Marek Wydmuch and the Farama-Foundation for their great work on the [vizdoom environment](https://github.com/Farama-Foundation/ViZDoom).

Thanks to the vizdoomgym team and its contributors for providing a starting point for the [vizdoom gym environment code](https://github.com/shakenes/vizdoomgym) used in this implementation.

Thanks to the Stable Diffusion team and its contributors for open sourcing their [code](https://github.com/CompVis/stable-diffusion), which was used in part for the implementation of the convolutional network in this implementation. 
