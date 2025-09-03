
# Adaptive Resource Optimization in Blockchain using Reinforcement Learning

This project presents a novel approach for optimizing resource utilization in blockchain networks using **Deep Reinforcement Learning (DRL)**. It leverages intelligent, adaptive policy learning to improve network efficiency in terms of latency, throughput, and orphan block rate.

## 🔍 Overview

Traditional blockchain networks suffer from inefficient use of bandwidth, high latency, and transaction congestion. This project introduces a Reinforcement Learning-based decision-making framework that:

- Monitors real-time local metrics (latency, mempool depth, bandwidth)
- Learns optimal peer selection and block assembly strategies
- Continuously improves through feedback from the environment

## ✅ Features

- ✅ Actor-Critic based DRL (DDPG and P-DQN)
- ✅ SimPy-based discrete-event blockchain simulator
- ✅ Prioritized Experience Replay Buffer
- ✅ Modular training pipeline with benchmarking
- ✅ Comparative analysis with static and rule-based policies
- ✅ Detailed metrics tracking and visualization

## 🧠 Algorithms Used

### 1. Deep Deterministic Policy Gradient (DDPG)
- Actor-Critic architecture for continuous control
- Actor: Outputs neighbor selection weights, block size, and interval
- Critic: Estimates Q-value for state-action pairs
- Uses soft target updates and exploration noise

### 2. Parameterized DQN (P-DQN)
- Handles mixed discrete + continuous action spaces
- Discrete head chooses sampling strategy
- Parameter networks output continuous sub-actions

### 3. Prioritized Experience Replay
- Samples transitions based on temporal-difference (TD) error
- Improves sample efficiency and learning stability

### 4. SimPy Discrete-Event Simulation
- Models block propagation, peer communication, and transaction arrival
- Custom environment with `reset()` and `step(action)` interface

## 🏗️ Project Structure

```
adaptive_rl_blockchain/
│
├── agent.py              # DDPG agent definition
├── sim_env.py            # SimPy-based blockchain network simulator
├── replay_buffer.py      # Prioritized experience replay memory
├── metrics.py            # Logging and plotting for metrics
├── train.py              # Training script
├── compare.py            # Policy evaluation and comparison
├── config.yaml           # Environment and training configuration
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

## 📈 Metrics and Results

The following metrics are tracked during training and evaluation:
- Average Latency
- Average Throughput
- Orphan Block Rate

Generated visualizations:
- `metrics.png`: Learning curve plots (latency, throughput, orphan rate)
- `comparison_bar.png`: Policy comparison chart (Static, Rule-based, RL)

## 🧪 Evaluation

Three policies are compared:
- **Static**: Always selects all peers with fixed block size and interval
- **Rule-based**: Selects peers based on latency and mempool heuristics
- **RL-Orch**: Learns optimal strategy through continuous environment feedback

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- SimPy
- NumPy
- Matplotlib
- PyYAML
- tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Run Training

```bash
python train.py
```

## 📊 Run Evaluation

```bash
python compare.py
```

## 📂 Output

- `metrics.csv`: Logged metrics per episode
- `metrics.png`: Learning curves
- `comparison_bar.png`: Bar chart comparing policies

