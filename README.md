# 🚀 Deep Reinforcement Learning Journey

Welcome to my Deep Reinforcement Learning (DRL) experimentation repository! This is where I explore, learn, and implement various RL algorithms from the ground up. Whether it's coding Q-Learning from scratch or leveraging modern libraries like Stable Baselines3, this repo is all about **learning by doing**.

⚠️ **Educational Purpose Only**: All implementations here are purely for learning and experimentation. They're not production-ready and are meant to deepen understanding of RL concepts.

## 🎯 What's Inside

This repository contains my journey through the fascinating world of reinforcement learning, from basic tabular methods to deep neural network approaches.

### 📚 Learning Progression

#### **Foundational RL (Tabular Methods)**
- **Multi-Armed Bandits**: Sample averaging on 10-armed testbed
- **Q-Learning**: GridWorld, FrozenLake, and BlackJack implementations
- **SARSA**: WindyGridWorld navigation
- **Dyna-Q**: Model-based planning in GridWorld

#### Deep RL (Neural Network Methods)
- **DQN (Deep Q-Networks)**: Custom PyTorch implementations
- **PPO (Proximal Policy Optimization)**: Using Stable Baselines3
- **Advanced DQN variants**: Exploring different architectures and techniques

### 🏗️ Repository Structure

```
📦 DRL/
├── 📁 notebooks/           # Jupyter notebooks for exploration & visualization
│   ├── QLearn-*.ipynb     # Q-Learning experiments
│   ├── SARSA-*.ipynb      # SARSA implementations
│   ├── DynaQ-*.ipynb      # Model-based learning
│   └── SampleAverage-*.ipynb # Multi-armed bandit solutions
│
├── 📁 scripts/            # Standalone Python implementations
│   ├── 📁 agents/         # Custom RL agent implementations
│   │   ├── q_agents.py    # Tabular Q-Learning agents
│   │   ├── dnn_agents.py  # Deep neural network agents
│   │   ├── planners.py    # Model-based planners (Dyna-Q)
│   │   └── sample_averaging.py # Bandit algorithms
│   │
│   ├── 📁 environments/   # Custom environment implementations
│   │   └── simple.py      # GridWorld and simple envs
│   │
│   ├── 📁 structures/     # Neural network architectures & utilities
│   │   ├── q_network.py   # DQN architectures
│   │   └── replay_buffer.py # Experience replay implementation
│   │
│   ├── 📁 utils/          # Plotting, evaluation, and helper functions
│   │   ├── evaluation.py  # Performance metrics
│   │   └── plot.py        # Visualization utilities
│   │
│   └── *.py              # Individual experiment scripts
│
├── 📁 rlzoo_configs/      # RL Zoo3 configuration files
│   └── *.yml             # Hyperparameter configurations
│
├── requirements.txt       # Dependencies
└── README.md             # You are here! 👋
```

## 🛠️ Technologies & Libraries

### Core Libraries
- **PyTorch**: For deep learning implementations
- **Stable Baselines3**: State-of-the-art RL algorithms
- **RL Zoo3**: Pre-tuned hyperparameters and configurations
- **Gymnasium**: OpenAI Gym's successor for RL environments

### Environments Explored
- **Classic Control**: LunarLander, CartPole
- **Atari Games**: SpaceInvaders, Tetris (using ALE)
- **Tabular Envs**: GridWorld, FrozenLake, BlackJack, WindyGridWorld
- **Custom Environments**: Simple navigation tasks

### Visualization & Analysis
- **Matplotlib & Seaborn**: Performance plots and analysis
- **TensorBoard**: Training monitoring and logging
- **Jupyter Notebooks**: Interactive experimentation

## 🚀 Getting Started

### Prerequisites
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

# Install dependencies
```bash
pip install -r requirements.txt
```

### Running Experiments

#### Tabular Methods
```bash
# Run Q-Learning on FrozenLake
python scripts/QLearn-FrozenLake-V1.py

# Explore SARSA on WindyGridWorld
python scripts/SARSA-WindGridWorld-V0.py

# Try multi-armed bandit
python scripts/SampleAverage-10ArmedTestbed-V0.py
```

#### Deep RL Methods
```bash
# Train DQN on LunarLander
python scripts/DQN-LunarLander-V3.py

# Run PPO on LunarLander
python scripts/PPO-LunarLander-V3.py

# Train on Atari games
python scripts/DQN-ALESpaceInvaders-V5.py
```

#### Using RL Zoo3 Configurations
```bash
# Train using pre-configured hyperparameters
python -m rl_zoo3.train --algo dqn --env ALE/SpaceInvaders-v5 --conf-file rlzoo_configs/DQN-ALESpaceInvaders-V5.yml
```

### Jupyter Notebooks
Explore the interactive notebooks for step-by-step learning:
```bash
jupyter lab
# Navigate to notebooks/ directory
```

## 🧠 Learning Philosophy

This repository embodies a **learn-by-doing** approach:

1. **Start Simple**: Begin with tabular methods to understand core RL concepts
2. **Build From Scratch**: Implement algorithms manually to grasp the mechanics  
3. **Experiment Freely**: Try different hyperparameters, architectures, and environments
4. **Use Modern Tools**: Leverage Stable Baselines3 for state-of-the-art implementations
5. **Document Everything**: Track progress, insights, and lessons learned

## 📈 Current Focus Areas

- [ ] **Double DQN & Dueling DQN**: Enhancing value-based methods
- [ ] **Policy Gradient Methods**: Moving beyond value-based approaches
- [ ] **Actor-Critic Methods**: Combining value and policy methods
- [ ] **Multi-Agent RL**: Exploring competitive and cooperative scenarios
- [ ] **Hierarchical RL**: Learning complex behaviors through abstraction

## 🤝 Feedback & Collaboration

I'm always eager to learn and improve! If you spot any issues, have suggestions, or want to discuss RL concepts, please feel free to:

- **Open an Issue**: For bugs, questions, or suggestions
- **Start a Discussion**: Share ideas or alternative approaches
- **Contribute**: Pull requests are welcome for improvements

## 📚 Learning Resources

Some resources that have been invaluable in this journey:
- **Sutton & Barto**: "Reinforcement Learning: An Introduction"
- **OpenAI Spinning Up**: Comprehensive RL educational resource
- **Stable Baselines3 Documentation**: Excellent practical implementations
- **Deep RL Bootcamp**: UC Berkeley's deep RL course materials

## ⚡ Quick Tips for Fellow Learners

1. **Start with tabular methods** - they build intuition for RL fundamentals
2. **Visualize everything** - plots help understand convergence and behavior
3. **Experiment with hyperparameters** - small changes can have big impacts
4. **Read the original papers** - understand the theory behind implementations
5. **Don't worry about perfect code** - focus on learning and understanding first

---
*Happy Learning!* 🎓

*Remember: The goal isn't to create production code, but to deeply understand how these algorithms work. Every bug is a learning opportunity, and every experiment brings new insights!*

## 📄 License (MIT)
This project is for educational purposes. Feel free to use, modify, and learn from any code here!
