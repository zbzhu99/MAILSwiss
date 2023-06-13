This is a multi-agent version of [ILSwiss](https://github.com/Ericonaldo/ILSwiss), which contains various reinforcement learning and imitation learning algorithms. Now the multi-agent interface only supports independent training among agents, and more complicated interacting interface such as CTDE (Centralized Training with Decentralized Execution) will be implemented in the future.

# Current Available Algorithm

+ [x] Soft Actor Critic

+ [x] Proximal Policy Optimization

+ [x] Generative Adversarial Imitation Learning (GAIL)

+ [x] Behavior Cloning (BC)

+ [ ] QMIX

+ [ ] MADDPG


# Setup

1. Configurate conda environment:

```bash
conda env create --name mail --file=env.yml
pip install -e .
```

2. Run RL algorithms to obtain expert policy:

```bash
python run_experiments.py -e exp_specs/sac/sac_mpe_spread.yaml
```

3. Generate expert demostrations using trained policy:

First change `policy_log_dir` in `exp_specs/gen_expert/mpe_spread.yaml` to the path of rl training log directory, then run:

```bash
python run_experiments.py -e exp_specs/gen_expert/mpe_spread.yaml
```

Add the path of generated demonstrations in `demos_listing.yaml`.

4. Run GAIL:

Make sure `expert_name` in `exp_specs/gail/gail_mpe_spread.yaml` is the same as the name in `demos_listing.yaml`, then run:

```bash
python run_experiments.py -e exp_specs/gail/gail_mpe_spread.yaml
```

5. Run BC:

Make sure `expert_name` in `exp_specs/bc/bc_mpe_spread.yaml` is the same as the name in `demos_listing.yaml`, then run:

```bash
python run_experiments.py -e exp_specs/bc/bc_mpe_spread.yaml
```
