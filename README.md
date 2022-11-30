# Vizdoom PPO + RND: General Doom Playing Agent using Proximal Policy Optimization and Random Network Distillation

This code base makes use of [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) and [Random Network Distillation (RND)](https://arxiv.org/pdf/1810.12894.pdf) to enable a Doom agent to beat levels in Doom and Doom 2 through the use of the vizdoom environment. 

<div style="text-align:center;margin:0 1em">
<a style="margin:0 auto" href="https://youtube.com/watch?v=mPff0B6wNSs">
<img src="https://img.youtube.com/vi/mPff0B6wNSs/0.jpg" width="400">
</a>
</div>

The agent has currently been tested on *Map01 in Doom 2* and *E1M2 in Doom* and is able to learn to beat each level within approximately 6M global steps (across all running environments), this takes around 8 hours on a 12GB Nvidia RTX 3080 with 20 simultaneous environments. More testing is required to get the agent to generalize across further levels. The agent is able to approach perfect play in all basic vizdoom .cfg scenarios in <1M global steps in each scenario.

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

For online tracking information (integrates seemlessly with the Tensorboard data tracking). Pass the command line arg `--track` to activate it.
```bash
conda install -c conda-forge wandb
```

### Running / Training an Agent

To train the agent on doom level E1M2 you can run the following command:
```bash
python doom_ppo_rnd.py --map E1M2 --num-envs 20
```
I strongly recommend changing the `num-envs` to the highest possible number you can fit in your GPU memory for faster training.

If you want to train on a specific vizdoom scenario, a list of them can be found in `doom_gym_wrappers.py` and they can be enabled using the `--gym-id` flag (e.g., `--gym-id VizdoomCorridor-v0`). Note that each scenario will have specific constraints on the action space of the agent. I've found that the best way to train in these scenarios is to enable multi-discrete action spaces (`--multidiscrete-actions True`).

For example, to run the corridor scenaraio, use the following command:
```bash
python doom_ppo_rnd.py --gym-id VizdoomCorridor-v0 --multidiscrete-actions True
```

###  Model Saving and Loading

During training the model will automatically save at global step intervals, determined by the command line argument `--save-timesteps`. When saving takes place, a model checkpoint file will be generated in a unique run directory under `runs/<gym_env_id>/<unique_run_dir>`. All tensorboard stats data will also be placed in this same directory.

To load a saved model for further training you can use the command line argument `--model <path_to_my_saved_model>`.


## Implementation Details

The code makes use of the following networks for each agent:

- A convolutional network to encode the RGB and label buffers as a smaller, latent representation of the visual observation
- Embeddings for the x,y position and the orientation/angle, health, and ammo
- Cropped pixels from the HUD containing the currently held keycards
- RND target and predictor networks for determining the ‘novelty’ of a given observation (RGB pixels, keycard pixels, and location + orientation game variables as one-hot vectors)
- Long-Short Term Memory (LSTM) which is given the output of the convolutional network (latent representation), position, keycard pixels and game variable embeddings
- An actor network which takes the output of the LSTM to produce the distribution of actions for the agent to take
- A critic network that produces a single valued output for each agent to compare against generated rewards


On 12GB of GPU memory, the current implementation is able to run 20 simultaneous training environments. The amount of GPU memory (and the speed of training) should scale linearly with the number of agents.


## Thanks and Acknowledgements

Thank you to the OpenAI team (John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, Yuri Burda, Harrison Edwards, Amos Storkey) responsible for the development of both PPO and RND algorithms.

Many thanks to Huang, Shengyi; Dossa, Rousslan Fernand Julien; Raffin, Antonin; Kanervisto, Anssi; and Wang, Weixun for their [super comprehensive implementations and tips for PPO baselines](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

Thanks to Marek Wydmuch and the Farama-Foundation for their great work on the [vizdoom environment](https://github.com/Farama-Foundation/ViZDoom).

Thanks to the vizdoomgym team and its contributors for providing a starting point for the [vizdoom gym environment code](https://github.com/shakenes/vizdoomgym) used in this implementation.

Thanks to the Stable Diffusion team and its contributors for open sourcing their [code](https://github.com/CompVis/stable-diffusion), which was used in part for the implementation of the convolutional network in this repo. 
