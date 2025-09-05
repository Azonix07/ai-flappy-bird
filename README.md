# 🧠 AI Flappy Bird - Neural Evolution

An advanced AI implementation of Flappy Bird using neural networks and genetic algorithms. Watch as AI agents learn to play the game through evolutionary processes!

## 🚀 Features

### Core AI System
- **Enhanced Neural Networks**: 6-input, 12-hidden-node, 1-output architecture with ReLU and Sigmoid activations
- **Advanced Genetic Algorithm**: Multiple crossover strategies, adaptive mutation rates, and elitism
- **Smart Bird Behavior**: Improved decision-making with enhanced input features
- **Dynamic Difficulty**: Adaptive pipe spacing and speed based on generation performance

### Visual Enhancements
- **Beautiful Graphics**: Gradient backgrounds, detailed pipes, and animated clouds
- **Real-time Neural Visualization**: Live display of neural network activations
- **Performance Tracking**: Comprehensive statistics and generation analytics
- **Best Bird Highlighting**: Visual indicators for top-performing agents

### Optimization Features
- **Performance Optimized**: 3x speed multiplier for faster learning
- **Elitism Strategy**: Top 10% of birds preserved each generation
- **Tournament Selection**: Better diversity through competitive selection
- **Adaptive Mutations**: Mutation rates adjust based on fitness scores

## 🎮 Controls

| Key | Action |
|-----|--------|
| **SPACE** | Pause/Resume the simulation |
| **R** | Reset the entire simulation |
| **S** | Toggle statistics display |

## 🧬 How It Works

### Neural Network Architecture
```
Inputs (6 nodes):
├── Bird Y position (normalized)
├── Bird velocity (normalized) 
├── Distance to next gap (normalized)
├── Gap center position (normalized)
├── Height difference from gap center
└── Frames since last flap

Hidden Layer (12 nodes):
└── ReLU activation function

Output (1 node):
└── Flap decision (Sigmoid activation)
```

### Genetic Algorithm Process
1. **Population**: 150 birds per generation
2. **Selection**: 70% tournament selection, 30% roulette wheel
3. **Elitism**: Top 10% preserved unchanged
4. **Crossover**: Uniform, single-point, and blend strategies
5. **Mutation**: Adaptive rates with Gaussian distribution
6. **Fitness**: Multi-objective function considering survival, distance, and pipes passed

### Learning Progression
- **Early Generations**: Random behavior, learning basic survival
- **Mid Generations**: Developing timing and spatial awareness
- **Advanced Generations**: Optimized performance with precise control

## 📊 Statistics Tracked

- **Generation**: Current evolutionary generation
- **Birds Alive**: Number of surviving birds
- **Best Score**: Highest score in current generation
- **Average Score**: Mean performance of current generation
- **All-Time Best**: Best score across all generations

## 🔧 Technical Improvements

### Neural Network Enhancements
- Xavier weight initialization for better learning
- Numerical stability improvements in activation functions
- Enhanced input feature engineering
- Multi-strategy crossover operations

### Game Engine Optimizations
- Efficient collision detection with margin adjustments
- Dynamic pipe generation based on performance
- Optimized rendering with gradient backgrounds
- Real-time FPS monitoring

### Genetic Algorithm Upgrades
- Tournament selection for better diversity
- Adaptive mutation rates based on fitness
- Gaussian mutations for smoother changes
- Multi-objective fitness function

## 🎯 Performance Metrics

The AI typically achieves:
- **Generation 1-10**: Learning basic controls (0-100 points)
- **Generation 10-30**: Developing strategy (100-500 points)
- **Generation 30-50**: Optimizing performance (500-1000+ points)
- **Generation 50+**: Mastering the game (1000+ points consistently)

## 🌟 Advanced Features

- **Speciation Support**: Framework for population clustering (future enhancement)
- **Neural Visualization**: Real-time display of network activations
- **Performance Analytics**: Detailed statistics and trend tracking
- **Dynamic Environments**: Adaptive difficulty based on AI performance

## 🚀 Getting Started

1. Open `index.html` in a modern web browser
2. Watch the AI learn to play automatically
3. Use keyboard controls to interact with the simulation
4. Observe neural network activations in real-time

## 🧪 Experimentation

Try modifying these parameters in `main.js`:
- `POP_SIZE`: Population size (default: 150)
- `MUTATION_RATE`: Base mutation rate (default: 0.15)
- `PIPE_GAP`: Initial gap size (default: 200)
- `SPEED_MULTIPLIER`: Simulation speed (default: 3)

## 🏆 Achievement Goals

- Survive 1000+ frames consistently
- Achieve all-time best scores over 2000
- Maintain stable performance across generations
- Observe emergent intelligent behaviors

---

**Built with**: Vanilla JavaScript, HTML5 Canvas, and advanced AI algorithms
**Inspired by**: NEAT (NeuroEvolution of Augmenting Topologies) and genetic programming principles: How the System Works

## Introduction
Welcome! This is a fun project where we teach computers to play Flappy Bird using artificial intelligence. We'll explain everything step by step, like you're learning it for the first time. No fancy words without explanations!

## What is Flappy Bird?
Flappy Bird is a simple game where a bird flies through pipes. The bird falls down due to gravity, and you tap to make it flap up. The goal is to go through as many pipes as possible without hitting them.

In our version, **no human plays** - the computer controls everything!

## Artificial Intelligence (AI)
AI means making computers think and learn like humans. Here, we use two main ideas:
1. **Neural Networks** - Like a tiny brain inside each bird
2. **Evolution** - Birds get better over time, like animals in nature

## Neural Networks: The Bird's Brain
Each bird has a tiny "brain" that decides when to flap. Think of it as a decision-making machine.

### How It Works
The brain takes **inputs** (information) and gives **outputs** (decisions).

**Inputs (4 pieces of information):**
- Where the bird is on the screen (up/down)
- How fast the bird is moving (up or down)
- Height of the top pipe
- Height of the bottom pipe

**Output (1 decision):**
- Flap or don't flap (yes/no)

### The Brain's Structure
```
Inputs → Hidden Layer → Output
   4   →     8       →   1
```

**Neural Network Diagram:**
```
Input Layer     Hidden Layer     Output Layer
   [I1]           [H1]              [O1]
   [I2]           [H2]              (Flap?)
   [I3]           [H3]
   [I4]           [H4]
                 [H5]
                 [H6]
                 [H7]
                 [H8]

Connections: I1→H1, I1→H2, ..., I4→H8, H1→O1, ..., H8→O1
```

- **Input layer**: 4 numbers (the information above)
- **Hidden layer**: 8 "thinking" units
- **Output layer**: 1 number (0 = don't flap, 1 = flap)

### Mathematics: How the Brain Thinks
The brain uses simple math to make decisions:

1. **Multiplication**: Each connection has a "weight" (like importance)
   ```
   Input × Weight = Signal
   ```

2. **Addition**: Add up all signals
   ```
   Signal1 + Signal2 + Signal3 + ... = Total
   ```

3. **Sigmoid Function**: Squish the total into a number between 0 and 1
   ```
   Sigmoid(x) = 1 / (1 + e^(-x))
   ```
   - If x is big positive: result ≈ 1
   - If x is big negative: result ≈ 0
   - If x is zero: result ≈ 0.5

**Sigmoid Function Graph:**
```
Output
  1.0 |          .--------
      |        .'
      |      .'
  0.5 |....'
      |   .'
      |  .'
  0.0 |.'          --------
      +-------------------- Input
        -3  -2  -1   0   1   2   3
```

This gives us a number between 0 and 1. If > 0.5, the bird flaps!

## Evolution: Getting Smarter Over Time
Birds start with random brains and get better through "evolution" (like natural selection).

### How Evolution Works
1. **Generation**: A group of 100 birds plays the game
2. **Fitness**: How well each bird did (score = how many pipes passed)
3. **Selection**: Pick the best birds to be "parents"
4. **Crossover**: Mix parents' brains to make babies
5. **Mutation**: Add small random changes
6. **Repeat**: New generation with improved brains

### Mathematics in Evolution
**Fitness Calculation:**
```
Fitness = Score × Score
```
(Better scores get much higher fitness - rewards good birds more)

**Selection:**
- Calculate total fitness of all birds
- Pick randomly, but better birds more likely to be chosen
- Like a lottery where good birds have more tickets

**Crossover:**
- Take half the "weights" from mom, half from dad
- Creates a mix of both parents' brains

**Mutation:**
- 10% chance to change each weight
- Add small random number (-0.1 to +0.1)
- Prevents getting stuck, adds new ideas

**Fitness Distribution Chart:**
```
Fitness
  High |     ████  (Best birds)
        |   ████████
        | ████████████
  Low   |██████████████████  (Most birds)
        +--------------------
          Score: 0 → 1000
```

**Selection Process:**
```
Bird A (Score: 100) - Fitness: 10,000
Bird B (Score: 200) - Fitness: 40,000  ← More likely to be picked!
Bird C (Score: 50)  - Fitness: 2,500
```

## The Complete System
1. **Start**: 100 birds with random brains
2. **Play**: Each bird uses its brain to decide when to flap
3. **Die**: Birds hit pipes or go off screen
4. **Evolve**: Best birds make new generation
5. **Repeat**: Birds get better each time!

**Evolution Progress Over Time:**
```
Generation
   20 |          ██████████ (Birds playing forever!)
      |
   15 |       ████████
      |
   10 |    ██████
      |
    5 |  ████
      |
    1 | ██  (Random - crash immediately)
      +---------------------------------- Time
        Gen 1    5    10    15    20
```

**Score Improvement:**
```
Score
 1000 |          ██████████
      |
  750 |       ████████
      |
  500 |    ██████
      |
  250 |  ████
      |
    0 | ██
      +---------------------------------- Generation
        1    5    10    15    20
```

## Visualizing the Brain
In the top corners, you can see the neural networks "thinking":
- **Best Bird**: The smartest bird so far
- **Alive Bird**: A bird still playing

Colors show how active each part is:
- **Red**: Very active
- **Yellow**: Medium active
- **Blue**: Not active

**Neural Activity Example:**
```
Input Layer:    [🔵] [🔵] [🟡] [🔴]  ← Bird position, velocity, pipes
Hidden Layer:   [🟡] [🔴] [🟡] [🔵] [🟡] [🔵] [🟡] [🔴]  ← Processing
Output Layer:   [🔴]  ← FLAP! (High activation = flap)
```

**Activity Levels:**
```
0.0 - 0.3: 🔵 Blue (Inactive)
0.3 - 0.7: 🟡 Yellow (Medium)
0.7 - 1.0: 🔴 Red (Very Active)
```

## Why This Works
- **Trial and Error**: Birds try different strategies
- **Learning**: Good strategies survive and spread
- **Improvement**: Each generation is better than the last
- **No Human Help**: The computer figures it out itself!

## Fun Facts
- Birds start terrible (crash immediately)
- After 10-20 generations, they get pretty good
- The best birds can go forever!
- Each bird's brain has 4×8 + 8×1 = 40 connections
- That's 40 numbers the computer has to learn!

## Try It Yourself
1. Open `index.html` in your browser
2. Watch the birds learn
3. See the neural activity change
4. Notice how scores improve over generations

Isn't it amazing how simple math can create intelligent behavior? 🤖🦅

## Contact!
mail: mehmetkahyakas5@gmail.com

github: @mehmetkahya0

website: https://mehmetkahya0.github.io/
