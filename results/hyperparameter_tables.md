Budget: 500,000 environment frames per run, 3 training seeds per configuration. Mean Reward = mean validation return over 50 held-out validation episodes (± std across training seeds). All runs share γ=0.99 and a 2×128 tanh MLP unless stated. Bold = selected configuration.

### 4.1 DQN

| # | Learning Rate | Gamma | Replay Buffer Size | Batch Size | Exploration Strategy | Target Update | Mean Reward | On-time | Critical svc. | Crash % | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 0.0005 | 0.99 | 100,000 | 128 | ε-greedy 1→0.05 over 100k frames | hard / 1000 steps | 35.11 ± 2.47 | 9.79 | 0.342 | 0.7 | Reference |
| 02 | 0.0001 | 0.99 | 100,000 | 128 | ε-greedy 1→0.05 over 100k frames | hard / 1000 steps | 27.16 ± 1.92 | 9.03 | 0.312 | 3.3 | Learning rate 1e-4 (slower, more stable value fitting?) |
| 03 | 0.002 | 0.99 | 100,000 | 128 | ε-greedy 1→0.05 over 100k frames | hard / 1000 steps | 32.19 ± 2.99 | 9.49 | 0.335 | 2.0 | Learning rate 2e-3 (faster but risk of divergence) |
| 04 | 0.0005 | 0.95 | 100,000 | 128 | ε-greedy 1→0.05 over 100k frames | hard / 1000 steps | 34.23 ± 1.82 | 9.25 | 0.351 | 2.0 | Discount 0.95 (shorter planning horizon, ~20 decisions) |
| 05 | 0.0005 | 0.99 | 10,000 | 128 | ε-greedy 1→0.05 over 100k frames | hard / 1000 steps | 34.17 ± 3.14 | 9.63 | 0.345 | 1.3 | Replay buffer 10k (more on-policy, less decorrelated replay) |
| 06 | 0.0005 | 0.99 | 100,000 | 32 | ε-greedy 1→0.05 over 100k frames | hard / 1000 steps | 31.03 ± 1.88 | 9.62 | 0.314 | 2.7 | Batch size 32 (noisier gradients) |
| 07 | 0.0005 | 0.99 | 100,000 | 128 | ε-greedy 1→0.01 over 25k frames | hard / 1000 steps | 32.74 ± 2.24 | 9.38 | 0.353 | 4.0 | Fast exploration decay |
| 08 | 0.0005 | 0.99 | 100,000 | 128 | ε-greedy 1→0.1 over 250k frames | hard / 1000 steps | 36.12 ± 0.88 | 9.77 | 0.348 | 1.3 | Slow exploration decay |
| 09 | 0.0005 | 0.99 | 100,000 | 128 | ε-greedy 1→0.05 over 100k frames | soft τ=0.005 | 33.60 ± 1.56 | 9.62 | 0.333 | 2.7 | Soft (Polyak) target update, tau 0.005, instead of hard updates |
| **10** | 0.0005 | 0.99 | 100,000 | 128 | ε-greedy 1→0.05 over 100k frames | hard / 1000 steps, vanilla | **36.76 ± 0.49** | 9.78 | 0.349 | 0.0 | Vanilla DQN target (Double DQN off) |
| 11 | 0.0005 | 0.99 | 100,000 | 128 | ε-greedy 1→0.05 over 100k frames | hard / 1000 steps | 19.71 ± 4.24 | 8.65 | 0.291 | 4.0 | no action mask; invalid actions/ep 2.9 |

### 4.2 REINFORCE

| # | Learning Rate | Gamma | Episodes / Update | Baseline | Return Norm. | Entropy Coef. | Mean Reward | On-time | Critical svc. | Crash % | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 0.001 | 0.99 | 8 | none | yes | 0 | 22.01 ± 3.82 | 8.62 | 0.274 | 1.3 | Reference |
| 02 | 0.0003 | 0.99 | 8 | none | yes | 0 | 18.58 ± 0.31 | 8.59 | 0.254 | 0.0 | Learning rate 3e-4 |
| 03 | 0.003 | 0.99 | 8 | none | yes | 0 | 22.35 ± 3.54 | 8.59 | 0.287 | 1.3 | Learning rate 3e-3 |
| 04 | 0.001 | 0.95 | 8 | none | yes | 0 | 22.53 ± 4.48 | 8.60 | 0.285 | 0.7 | Discount 0.95 |
| 05 | 0.001 | 0.999 | 8 | none | yes | 0 | 22.10 ± 3.65 | 8.37 | 0.280 | 0.7 | Discount 0.999 (near-undiscounted shift return, higher variance) |
| 06 | 0.001 | 0.99 | 8 | V(s) | yes | 0 | 18.25 ± 1.12 | 8.35 | 0.256 | 0.0 | Learned state-value baseline |
| 07 | 0.001 | 0.99 | 8 | none | no | 0 | 18.97 ± 1.57 | 8.27 | 0.259 | 0.0 | Raw (unnormalised) returns |
| **08** | 0.001 | 0.99 | 2 | none | yes | 0 | **22.77 ± 4.04** | 8.61 | 0.285 | 1.3 | 2 episodes per update (high-variance gradient) |
| 09 | 0.001 | 0.99 | 32 | none | yes | 0 | 19.02 ± 0.80 | 8.72 | 0.242 | 0.0 | 32 episodes per update (low-variance, fewer updates) |
| 10 | 0.001 | 0.99 | 8 | none | yes | 0.01 | 19.12 ± 3.14 | 8.20 | 0.263 | 0.0 | Entropy bonus 0.01 |
| 11 | 0.001 | 0.99 | 8 | none | yes | 0 | -46.32 ± 0.00 | 0.00 | 0.000 | 0.0 | no action mask; invalid actions/ep 0.0 |

### 4.3 PPO

| # | Learning Rate | Clip ε | Epochs | GAE λ | Entropy Coef. | Rollout / Minibatch | Mean Reward | On-time | Critical svc. | Crash % | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 0.0003 | 0.2 | 4 | 0.95 | 0.01 | 2048 / 256 | 24.35 ± 3.07 | 8.69 | 0.289 | 0.0 | Reference |
| 02 | 0.0001 | 0.2 | 4 | 0.95 | 0.01 | 2048 / 256 | 18.98 ± 1.97 | 8.33 | 0.257 | 0.0 | Learning rate 1e-4 |
| 03 | 0.001 | 0.2 | 4 | 0.95 | 0.01 | 2048 / 256 | 28.04 ± 1.29 | 8.86 | 0.323 | 2.7 | Learning rate 1e-3 |
| 04 | 0.0003 | 0.1 | 4 | 0.95 | 0.01 | 2048 / 256 | 28.12 ± 1.49 | 8.98 | 0.319 | 0.7 | Clip epsilon 0.1 (more conservative updates) |
| 05 | 0.0003 | 0.3 | 4 | 0.95 | 0.01 | 2048 / 256 | 24.57 ± 1.23 | 8.47 | 0.292 | 0.0 | Clip epsilon 0.3 (larger policy steps) |
| **06** | 0.0003 | 0.2 | 10 | 0.95 | 0.01 | 2048 / 256 | **30.95 ± 1.23** | 9.30 | 0.339 | 2.0 | 10 optimisation epochs per rollout (more sample reuse) |
| 07 | 0.0003 | 0.2 | 4 | 0.8 | 0.01 | 2048 / 256 | 26.31 ± 2.70 | 8.99 | 0.311 | 1.3 | GAE lambda 0.80 (lower variance, more bias) |
| 08 | 0.0003 | 0.2 | 4 | 0.95 | 0 | 2048 / 256 | 23.58 ± 0.69 | 8.69 | 0.273 | 0.0 | No entropy bonus |
| 09 | 0.0003 | 0.2 | 4 | 0.95 | 0.05 | 2048 / 256 | 28.58 ± 0.64 | 9.07 | 0.335 | 2.7 | Entropy coefficient 0.05 (stronger exploration) |
| 10 | 0.0003 | 0.2 | 4 | 0.95 | 0.01 | 512 / 128 | 24.56 ± 2.90 | 8.60 | 0.288 | 0.0 | Short rollouts |
| 11 | 0.0003 | 0.2 | 4 | 0.95 | 0.01 | 2048 / 256 | -15.03 ± 21.68 | 3.46 | 0.172 | 0.7 | no action mask; invalid actions/ep 0.3 |

### 4.4 A2C (Actor-Critic)

| # | Learning Rate | Rollout (envs × steps) | GAE λ | Entropy Coef. | Critic Coef. | Adv. Norm. | Mean Reward | On-time | Critical svc. | Crash % | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **01** | 0.0007 | 8 × 8 | 1 | 0.01 | 0.5 | no | **25.21 ± 0.52** | 8.87 | 0.289 | 0.0 | Reference |
| 02 | 0.0002 | 8 × 8 | 1 | 0.01 | 0.5 | no | 25.03 ± 0.73 | 8.83 | 0.297 | 0.7 | Learning rate 2e-4 |
| 03 | 0.002 | 8 × 8 | 1 | 0.01 | 0.5 | no | 23.55 ± 1.54 | 8.86 | 0.292 | 2.7 | Learning rate 2e-3 |
| 04 | 0.0007 | 8 × 8 | 0.95 | 0.01 | 0.5 | no | 24.53 ± 1.76 | 8.85 | 0.296 | 1.3 | GAE lambda 0.95 advantages |
| 05 | 0.0007 | 8 × 8 | 0 | 0.01 | 0.5 | no | 16.00 ± 2.14 | 7.85 | 0.244 | 0.0 | GAE lambda 0 = one-step TD actor-critic (lowest variance, highest bias) |
| 06 | 0.0007 | 8 × 32 | 1 | 0.01 | 0.5 | no | 21.11 ± 3.78 | 8.75 | 0.262 | 0.0 | Longer rollouts |
| 07 | 0.0007 | 8 × 8 | 1 | 0 | 0.5 | no | 24.82 ± 1.18 | 8.80 | 0.292 | 0.0 | No entropy bonus |
| 08 | 0.0007 | 8 × 8 | 1 | 0.05 | 0.5 | no | 23.32 ± 1.19 | 8.85 | 0.292 | 2.7 | Entropy coefficient 0.05 |
| 09 | 0.0007 | 8 × 8 | 1 | 0.01 | 0.25 | no | 24.44 ± 1.12 | 8.71 | 0.296 | 0.7 | Critic loss coefficient 0.25 (weaker value fitting) |
| 10 | 0.0007 | 8 × 8 | 1 | 0.01 | 0.5 | yes | 24.40 ± 0.70 | 8.82 | 0.293 | 1.3 | Per-batch advantage normalisation on |
| 11 | 0.0007 | 8 × 8 | 1 | 0.01 | 0.5 | no | -46.32 ± 0.00 | 0.00 | 0.000 | 0.0 | no action mask; invalid actions/ep 0.0 |
