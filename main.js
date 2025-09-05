const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const neural1 = document.getElementById('neural1');
const ctx1 = neural1.getContext('2d');
const neural2 = document.getElementById('neural2');
const ctx2 = neural2.getContext('2d');

// New graph canvases
const fitnessGraph = document.getElementById('fitnessGraph');
const ctxFitness = fitnessGraph.getContext('2d');
const diversityGraph = document.getElementById('diversityGraph');
const ctxDiversity = diversityGraph.getContext('2d');
const activityHeatmap = document.getElementById('activityHeatmap');
const ctxActivity = activityHeatmap.getContext('2d');
const genePoolGraph = document.getElementById('genePoolGraph');
const ctxGenePool = genePoolGraph.getContext('2d');

// Game constants - optimized for better AI learning
const WIDTH = 400;
const HEIGHT = 600;
const BIRD_SIZE = 20;
const PIPE_WIDTH = 60;
const PIPE_GAP = 200; // Increased for easier early learning
const GRAVITY = 0.6; // Reduced for more control
const JUMP = -12; // Adjusted for new gravity
const POP_SIZE = 150; // Reduced for faster generations
const MUTATION_RATE = 0.15; // Increased for more exploration
const SPEED_MULTIPLIER = 3; // Game speed multiplier

// Performance tracking
let totalGenerations = 0;
let allTimeBest = 0;
let generationBests = [];

// Enhanced Neural Network class with better architecture
class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        // Xavier initialization for better learning
        this.weightsIH = this.xavierMatrix(hiddenNodes, inputNodes);
        this.weightsHO = this.xavierMatrix(outputNodes, hiddenNodes);
        this.biasH = this.randomMatrix(hiddenNodes, 1);
        this.biasO = this.randomMatrix(outputNodes, 1);
        
        // Learning rate for adaptive mutations
        this.learningRate = 0.1;
    }

    xavierMatrix(rows, cols) {
        let matrix = [];
        let scale = Math.sqrt(2.0 / cols);
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = (Math.random() * 2 - 1) * scale;
            }
        }
        return matrix;
    }

    randomMatrix(rows, cols) {
        let matrix = [];
        for (let i = 0; i < rows; i++) {
            matrix[i] = [];
            for (let j = 0; j < cols; j++) {
                matrix[i][j] = Math.random() * 2 - 1;
            }
        }
        return matrix;
    }

    feedforward(inputs) {
        // Convert inputs to matrix format
        let inputMatrix = inputs.map(x => [x]);
        
        let hidden = this.matrixMultiply(this.weightsIH, inputMatrix);
        hidden = this.addMatrix(hidden, this.biasH);
        hidden = this.applyActivation(hidden, this.relu); // Using ReLU for hidden layer

        let output = this.matrixMultiply(this.weightsHO, hidden);
        output = this.addMatrix(output, this.biasO);
        output = this.applyActivation(output, this.sigmoid); // Sigmoid for output

        return { output: output, hidden: hidden };
    }

    // ReLU activation function
    relu(x) {
        return Math.max(0, x);
    }

    // Improved sigmoid with better numerical stability
    sigmoid(x) {
        if (x > 500) return 1;
        if (x < -500) return 0;
        return 1 / (1 + Math.exp(-x));
    }

    applyActivation(matrix, activationFunc) {
        return matrix.map(row => row.map(activationFunc));
    }

    matrixMultiply(a, b) {
        let result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < b[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < a[0].length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    addMatrix(a, b) {
        let result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    copy() {
        let nn = new NeuralNetwork(this.inputNodes, this.hiddenNodes, this.outputNodes);
        nn.weightsIH = this.copyMatrix(this.weightsIH);
        nn.weightsHO = this.copyMatrix(this.weightsHO);
        nn.biasH = this.copyMatrix(this.biasH);
        nn.biasO = this.copyMatrix(this.biasO);
        nn.learningRate = this.learningRate;
        return nn;
    }

    copyMatrix(matrix) {
        return matrix.map(row => row.slice());
    }

    // Enhanced mutation with adaptive rates
    mutate(fitness = 0) {
        // Adaptive mutation rate based on fitness
        let adaptiveRate = MUTATION_RATE * (1 + Math.exp(-fitness / 1000));
        
        this.mutateMatrix(this.weightsIH, adaptiveRate);
        this.mutateMatrix(this.weightsHO, adaptiveRate);
        this.mutateMatrix(this.biasH, adaptiveRate);
        this.mutateMatrix(this.biasO, adaptiveRate);
    }

    mutateMatrix(matrix, rate = MUTATION_RATE) {
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                if (Math.random() < rate) {
                    // Gaussian mutation for smoother changes
                    let gaussian = this.randomGaussian(0, 0.2);
                    matrix[i][j] += gaussian;
                    // Clamp values to prevent extreme weights
                    matrix[i][j] = Math.max(-5, Math.min(5, matrix[i][j]));
                }
            }
        }
    }

    // Box-Muller transform for Gaussian random numbers
    randomGaussian(mean = 0, std = 1) {
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        return z * std + mean;
    }

    // Improved crossover with multiple strategies
    crossover(partner) {
        let child = new NeuralNetwork(this.inputNodes, this.hiddenNodes, this.outputNodes);
        
        // Randomly choose crossover strategy
        let strategy = Math.floor(Math.random() * 3);
        
        switch(strategy) {
            case 0: // Uniform crossover
                child.weightsIH = this.uniformCrossover(this.weightsIH, partner.weightsIH);
                child.weightsHO = this.uniformCrossover(this.weightsHO, partner.weightsHO);
                child.biasH = this.uniformCrossover(this.biasH, partner.biasH);
                child.biasO = this.uniformCrossover(this.biasO, partner.biasO);
                break;
            case 1: // Single point crossover
                child.weightsIH = this.singlePointCrossover(this.weightsIH, partner.weightsIH);
                child.weightsHO = this.singlePointCrossover(this.weightsHO, partner.weightsHO);
                child.biasH = this.singlePointCrossover(this.biasH, partner.biasH);
                child.biasO = this.singlePointCrossover(this.biasO, partner.biasO);
                break;
            case 2: // Blend crossover
                child.weightsIH = this.blendCrossover(this.weightsIH, partner.weightsIH);
                child.weightsHO = this.blendCrossover(this.weightsHO, partner.weightsHO);
                child.biasH = this.blendCrossover(this.biasH, partner.biasH);
                child.biasO = this.blendCrossover(this.biasO, partner.biasO);
                break;
        }
        
        return child;
    }

    uniformCrossover(a, b) {
        let result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < a[i].length; j++) {
                result[i][j] = Math.random() < 0.5 ? a[i][j] : b[i][j];
            }
        }
        return result;
    }

    singlePointCrossover(a, b) {
        let result = [];
        let crossoverPoint = Math.floor(Math.random() * a.length);
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < a[i].length; j++) {
                result[i][j] = i < crossoverPoint ? a[i][j] : b[i][j];
            }
        }
        return result;
    }

    blendCrossover(a, b) {
        let result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < a[i].length; j++) {
                let alpha = Math.random();
                result[i][j] = alpha * a[i][j] + (1 - alpha) * b[i][j];
            }
        }
        return result;
    }
}

// Enhanced Bird class with improved AI behavior
class Bird {
    constructor(brain) {
        this.x = 50;
        this.y = HEIGHT / 2;
        this.velocity = 0;
        this.brain = brain || new NeuralNetwork(6, 12, 1); // More inputs and hidden nodes
        this.score = 0;
        this.fitness = 0;
        this.dead = false;
        this.framesSinceFlap = 0;
        this.distanceTraveled = 0;
        this.pipesPassed = 0;
    }

    think(pipes) {
        // Find the closest upcoming pipe
        let closest = null;
        let closestD = Infinity;
        for (let pipe of pipes) {
            let d = pipe.x + PIPE_WIDTH - this.x;
            if (d > -PIPE_WIDTH && d < closestD) {
                closest = pipe;
                closestD = d;
            }
        }

        let inputs;
        if (closest) {
            // Enhanced input features for better learning
            let distanceToGap = closest.x + PIPE_WIDTH - this.x;
            let gapCenter = (closest.top + closest.bottom) / 2;
            let heightDiff = this.y - gapCenter;
            
            inputs = [
                this.y / HEIGHT,                           // Normalized bird height
                this.velocity / 20,                        // Normalized velocity
                distanceToGap / WIDTH,                     // Distance to next gap
                gapCenter / HEIGHT,                        // Gap center position
                heightDiff / HEIGHT,                       // Height difference from gap center
                this.framesSinceFlap / 60                  // Frames since last flap
            ];
        } else {
            // Default values when no pipes present
            inputs = [
                this.y / HEIGHT,
                this.velocity / 20,
                1.0,
                0.5,
                0.0,
                this.framesSinceFlap / 60
            ];
        }

        let result = this.brain.feedforward(inputs);
        this.hiddenActivations = result.hidden;
        
        // Decision threshold with some randomness for exploration
        let threshold = 0.5;
        if (result.output[0][0] > threshold) {
            this.flap();
        }
        
        this.framesSinceFlap++;
    }

    flap() {
        this.velocity = JUMP;
        this.framesSinceFlap = 0;
    }

    update() {
        this.velocity += GRAVITY;
        this.y += this.velocity;
        this.score++;
        this.distanceTraveled++;
        
        // Bonus for staying alive longer
        if (this.score % 100 === 0) {
            this.fitness += 50;
        }
    }

    show() {
        // Dynamic color based on performance
        let intensity = Math.min(255, this.score / 10);
        ctx.fillStyle = `rgb(255, ${255 - intensity}, 0)`;
        ctx.fillRect(this.x - BIRD_SIZE/2, this.y - BIRD_SIZE/2, BIRD_SIZE, BIRD_SIZE);
        
        // Add a small trail effect for the best bird
        ctx.fillStyle = `rgba(255, ${255 - intensity}, 0, 0.3)`;
        ctx.fillRect(this.x - BIRD_SIZE/2 - 2, this.y - BIRD_SIZE/2, BIRD_SIZE + 4, BIRD_SIZE);
    }

    offscreen() {
        return this.y > HEIGHT || this.y < 0;
    }

    hits(pipe) {
        // More forgiving collision detection
        let margin = 3;
        if (this.x + margin > pipe.x + PIPE_WIDTH || this.x + BIRD_SIZE - margin < pipe.x) return false;
        if (this.y - margin < pipe.top || this.y + BIRD_SIZE + margin > pipe.bottom) return true;
        return false;
    }

    // Enhanced fitness calculation
    calculateFitness() {
        // Multi-objective fitness function
        let survivalBonus = this.score * this.score;
        let distanceBonus = this.distanceTraveled;
        let pipeBonus = this.pipesPassed * 1000;
        
        this.fitness = survivalBonus + distanceBonus + pipeBonus;
        
        // Penalty for dying early
        if (this.dead && this.score < 100) {
            this.fitness *= 0.1;
        }
        
        return this.fitness;
    }
}

// Enhanced Pipe class with dynamic difficulty
class Pipe {
    constructor(generation = 1) {
        this.x = WIDTH;
        // Dynamic gap size based on generation
        let dynamicGap = Math.max(150, PIPE_GAP - (generation - 1) * 2);
        this.gap = dynamicGap;
        
        this.top = Math.random() * (HEIGHT - this.gap - 100) + 50;
        this.bottom = this.top + this.gap;
        this.width = PIPE_WIDTH;
        this.speed = 2.5 + (generation - 1) * 0.1; // Gradual speed increase
        this.passed = false;
        this.scored = false;
    }

    update() {
        this.x -= this.speed;
    }

    show() {
        // Gradient pipe colors
        let gradient = ctx.createLinearGradient(this.x, 0, this.x + this.width, 0);
        gradient.addColorStop(0, '#2d5016');
        gradient.addColorStop(0.5, '#4a7c59');
        gradient.addColorStop(1, '#2d5016');
        
        ctx.fillStyle = gradient;
        
        // Top pipe
        ctx.fillRect(this.x, 0, this.width, this.top);
        // Pipe cap
        ctx.fillRect(this.x - 5, this.top - 20, this.width + 10, 20);
        
        // Bottom pipe
        ctx.fillRect(this.x, this.bottom, this.width, HEIGHT - this.bottom);
        // Pipe cap
        ctx.fillRect(this.x - 5, this.bottom, this.width + 10, 20);
        
        // Add some visual details
        ctx.strokeStyle = '#1a3009';
        ctx.lineWidth = 2;
        ctx.strokeRect(this.x, 0, this.width, this.top);
        ctx.strokeRect(this.x, this.bottom, this.width, HEIGHT - this.bottom);
    }

    offscreen() {
        return this.x + this.width < 0;
    }

    // Check if bird passed this pipe
    checkPassed(bird) {
        if (!this.passed && bird.x > this.x + this.width) {
            this.passed = true;
            return true;
        }
        return false;
    }
}

// Enhanced Population class with better selection and statistics
class Population {
    constructor(size) {
        this.birds = [];
        this.generation = 1;
        this.bestScore = 0;
        this.avgScore = 0;
        this.species = []; // For speciation (advanced feature)
        
        for (let i = 0; i < size; i++) {
            this.birds.push(new Bird());
        }
    }

    update(pipes) {
        for (let bird of this.birds) {
            if (!bird.dead) {
                bird.think(pipes);
                bird.update();
                
                // Check pipe collisions
                for (let pipe of pipes) {
                    if (bird.hits(pipe)) {
                        bird.dead = true;
                        break;
                    }
                    
                    // Check if bird passed a pipe
                    if (pipe.checkPassed(bird)) {
                        bird.pipesPassed++;
                        bird.fitness += 100; // Bonus for passing pipes
                    }
                }
                
                // Check boundaries
                if (bird.offscreen()) {
                    bird.dead = true;
                }
            }
        }
    }

    show() {
        // Show only living birds, with best bird highlighted
        let best = this.getBestBird();
        
        for (let bird of this.birds) {
            if (!bird.dead) {
                if (bird === best) {
                    // Highlight best bird
                    ctx.strokeStyle = 'red';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(bird.x - BIRD_SIZE/2 - 2, bird.y - BIRD_SIZE/2 - 2, BIRD_SIZE + 4, BIRD_SIZE + 4);
                }
                bird.show();
            }
        }
    }

    allDead() {
        return this.birds.every(bird => bird.dead);
    }

    // Enhanced fitness calculation with multiple strategies
    calculateFitness() {
        let sum = 0;
        let maxFitness = 0;
        
        // Calculate individual fitness
        for (let bird of this.birds) {
            bird.calculateFitness();
            sum += bird.fitness;
            maxFitness = Math.max(maxFitness, bird.fitness);
        }
        
        // Normalize fitness
        if (sum > 0) {
            for (let bird of this.birds) {
                bird.fitness /= sum;
            }
        }
        
        // Calculate statistics
        this.avgScore = this.birds.reduce((sum, bird) => sum + bird.score, 0) / this.birds.length;
        this.bestScore = Math.max(...this.birds.map(b => b.score));
        
        // Update all-time best
        allTimeBest = Math.max(allTimeBest, this.bestScore);
        generationBests.push(this.bestScore);
    }

    // Tournament selection for better diversity
    tournamentSelection(tournamentSize = 5) {
        let tournament = [];
        for (let i = 0; i < tournamentSize; i++) {
            let randomIndex = Math.floor(Math.random() * this.birds.length);
            tournament.push(this.birds[randomIndex]);
        }
        
        tournament.sort((a, b) => b.fitness - a.fitness);
        return tournament[0];
    }

    // Roulette wheel selection (original method)
    pickOne() {
        let index = 0;
        let r = Math.random();
        while (r > 0 && index < this.birds.length) {
            r -= this.birds[index].fitness;
            index++;
        }
        index = Math.max(0, Math.min(this.birds.length - 1, index - 1));
        return this.birds[index];
    }

    // Enhanced next generation with elitism
    nextGeneration() {
        this.calculateFitness();
        
        let newBirds = [];
        
        // Elitism: Keep best 10% unchanged
        let sortedBirds = [...this.birds].sort((a, b) => b.fitness - a.fitness);
        let eliteCount = Math.floor(this.birds.length * 0.1);
        
        for (let i = 0; i < eliteCount; i++) {
            let elite = new Bird(sortedBirds[i].brain.copy());
            newBirds.push(elite);
        }
        
        // Generate remaining population
        for (let i = eliteCount; i < this.birds.length; i++) {
            let parentA, parentB;
            
            // Use tournament selection 70% of the time, roulette 30%
            if (Math.random() < 0.7) {
                parentA = this.tournamentSelection();
                parentB = this.tournamentSelection();
            } else {
                parentA = this.pickOne();
                parentB = this.pickOne();
            }
            
            let child = parentA.brain.crossover(parentB.brain);
            child.mutate(parentA.fitness + parentB.fitness);
            newBirds.push(new Bird(child));
        }
        
        this.birds = newBirds;
        this.generation++;
        totalGenerations++;
        
        // Reset all birds
        for (let bird of this.birds) {
            bird.dead = false;
            bird.score = 0;
            bird.fitness = 0;
            bird.y = HEIGHT / 2;
            bird.velocity = 0;
            bird.framesSinceFlap = 0;
            bird.distanceTraveled = 0;
            bird.pipesPassed = 0;
        }
    }

    getBestBird() {
        let best = this.birds[0];
        for (let bird of this.birds) {
            if (!bird.dead && bird.score > best.score) {
                best = bird;
            }
        }
        return best;
    }

    getAliveBirds() {
        return this.birds.filter(bird => !bird.dead);
    }
}

// Enhanced Game class with better performance and features
class Game {
    constructor() {
        this.population = new Population(POP_SIZE);
        this.pipes = [];
        this.frameCount = 0;
        this.gameSpeed = 1;
        this.paused = false;
        this.showStats = true;
        this.backgroundGradient = this.createBackgroundGradient();
    }

    createBackgroundGradient() {
        let gradient = ctx.createLinearGradient(0, 0, 0, HEIGHT);
        gradient.addColorStop(0, '#87CEEB');
        gradient.addColorStop(1, '#98FB98');
        return gradient;
    }

    update() {
        if (this.paused) return;
        
        // Dynamic pipe spawning based on population performance
        let spawnRate = Math.max(60, 120 - this.population.generation);
        if (this.frameCount % spawnRate === 0 && this.frameCount > 30) {
            this.pipes.push(new Pipe(this.population.generation));
        }

        this.population.update(this.pipes);
        
        for (let pipe of this.pipes) {
            pipe.update();
        }
        
        this.pipes = this.pipes.filter(pipe => !pipe.offscreen());

        if (this.population.allDead()) {
            this.population.nextGeneration();
            this.pipes = [];
            this.frameCount = 0;
            
            // Add first pipe for new generation
            setTimeout(() => {
                this.pipes.push(new Pipe(this.population.generation));
            }, 30);
        }

        this.frameCount++;
    }

    show() {
        // Clear canvas with gradient background
        ctx.fillStyle = this.backgroundGradient;
        ctx.fillRect(0, 0, WIDTH, HEIGHT);

        // Draw clouds for ambiance
        this.drawClouds();

        // Draw pipes
        for (let pipe of this.pipes) {
            pipe.show();
        }

        // Draw population
        this.population.show();

        // Update info display
        this.updateInfoDisplay();

        // Visualize neural networks
        let bestBird = this.population.getBestBird();
        let aliveBirds = this.population.getAliveBirds();
        
        this.visualizeNeural(bestBird, ctx1, 'Best Bird');
        if (aliveBirds.length > 1) {
            this.visualizeNeural(aliveBirds[1], ctx2, 'Other Bird');
        } else if (aliveBirds.length > 0) {
            this.visualizeNeural(aliveBirds[0], ctx2, 'Last Bird');
        }
    }

    drawClouds() {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        let cloudOffset = (this.frameCount * 0.2) % (WIDTH + 100);
        
        // Simple cloud shapes
        for (let i = 0; i < 3; i++) {
            let x = (i * 150 - cloudOffset) % (WIDTH + 100) - 50;
            let y = 50 + i * 80;
            
            ctx.beginPath();
            ctx.arc(x, y, 20, 0, Math.PI * 2);
            ctx.arc(x + 25, y, 35, 0, Math.PI * 2);
            ctx.arc(x + 50, y, 20, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    updateInfoDisplay() {
        let aliveBirds = this.population.getAliveBirds();
        document.getElementById('generation').textContent = this.population.generation;
        document.getElementById('alive').textContent = aliveBirds.length;
        document.getElementById('bestScore').textContent = this.population.bestScore;
        
        // Update stat value classes based on values
        const aliveElement = document.getElementById('alive');
        if (aliveBirds.length > 50) {
            aliveElement.className = 'stat-value';
        } else if (aliveBirds.length > 10) {
            aliveElement.className = 'stat-value warning';
        } else {
            aliveElement.className = 'stat-value danger';
        }
        
        // Add additional stats
        let avgScoreElement = document.getElementById('avgScore');
        if (avgScoreElement) {
            avgScoreElement.textContent = Math.round(this.population.avgScore);
        }
        
        let allTimeBestElement = document.getElementById('allTimeBest');
        if (allTimeBestElement) {
            allTimeBestElement.textContent = allTimeBest;
        }

        // Update game status
        const statusIndicator = document.getElementById('statusIndicator');
        const gameStatus = document.getElementById('gameStatus');
        
        if (this.paused) {
            statusIndicator.className = 'status-indicator paused';
            gameStatus.textContent = 'Paused';
            gameStatus.className = 'stat-value warning';
        } else if (aliveBirds.length === 0) {
            statusIndicator.className = 'status-indicator stopped';
            gameStatus.textContent = 'Evolving';
            gameStatus.className = 'stat-value danger';
        } else {
            statusIndicator.className = 'status-indicator running';
            gameStatus.textContent = 'Learning';
            gameStatus.className = 'stat-value';
        }
    }

    visualizeNeural(bird, context, label) {
        context.clearRect(0, 0, 200, 150);
        
        if (!bird || !bird.hiddenActivations) {
            // Show "No Data" message when bird is not available
            context.fillStyle = 'rgba(255, 255, 255, 0.3)';
            context.font = '12px Inter';
            context.fillText('No Active Bird', 60, 75);
            return;
        }

        // Background with subtle gradient
        const gradient = context.createLinearGradient(0, 0, 200, 150);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        context.fillStyle = gradient;
        context.fillRect(0, 0, 200, 150);

        // Title and score with better typography
        context.fillStyle = '#00d4ff';
        context.font = 'bold 11px Orbitron, monospace';
        context.fillText(label, 5, 15);
        
        context.fillStyle = '#2ecc71';
        context.font = '10px Inter';
        context.fillText(`Score: ${bird.score}`, 5, 30);
        context.fillText(`Fitness: ${Math.round(bird.fitness * 1000)}`, 5, 42);

        // Network visualization with improved layout
        let inputPositions = [
            {x: 25, y: 60}, {x: 25, y: 75}, {x: 25, y: 90},
            {x: 25, y: 105}, {x: 25, y: 120}, {x: 25, y: 135}
        ];
        
        let hiddenPositions = [];
        for (let i = 0; i < 12; i++) {
            hiddenPositions.push({
                x: 80 + (i % 4) * 18,
                y: 60 + Math.floor(i / 4) * 20
            });
        }
        
        let outputPosition = {x: 170, y: 90};

        // Draw connections with activation-based opacity
        context.lineWidth = 1;
        
        // Input to hidden connections (simplified)
        for (let i = 0; i < Math.min(inputPositions.length, 3); i++) {
            for (let j = 0; j < Math.min(hiddenPositions.length, 8); j++) {
                const opacity = 0.1 + Math.abs(bird.hiddenActivations[j] ? bird.hiddenActivations[j][0] : 0) * 0.3;
                context.strokeStyle = `rgba(100, 150, 255, ${opacity})`;
                context.beginPath();
                context.moveTo(inputPositions[i].x, inputPositions[i].y);
                context.lineTo(hiddenPositions[j].x, hiddenPositions[j].y);
                context.stroke();
            }
        }

        // Draw nodes with improved visuals
        // Input nodes
        context.fillStyle = '#3498db';
        for (let pos of inputPositions) {
            context.beginPath();
            context.arc(pos.x, pos.y, 3, 0, 2 * Math.PI);
            context.fill();
            
            // Add glow effect
            context.shadowColor = '#3498db';
            context.shadowBlur = 5;
            context.fill();
            context.shadowBlur = 0;
        }

        // Hidden nodes with activation visualization
        for (let i = 0; i < hiddenPositions.length && i < bird.hiddenActivations.length; i++) {
            let activation = Math.abs(bird.hiddenActivations[i][0]);
            let intensity = Math.min(255, activation * 255);
            
            // Create pulsing effect for high activation
            let radius = 3 + (activation > 0.7 ? Math.sin(Date.now() * 0.01) * 1 : 0);
            
            context.fillStyle = `rgb(${intensity}, ${255 - intensity}, 50)`;
            context.beginPath();
            context.arc(hiddenPositions[i].x, hiddenPositions[i].y, radius, 0, 2 * Math.PI);
            context.fill();
            
            // Add glow for highly active neurons
            if (activation > 0.5) {
                context.shadowColor = `rgb(${intensity}, ${255 - intensity}, 50)`;
                context.shadowBlur = 8;
                context.fill();
                context.shadowBlur = 0;
            }
        }

        // Output node with decision visualization
        const isFlapping = bird.brain && bird.framesSinceFlap < 5;
        context.fillStyle = isFlapping ? '#e74c3c' : '#95a5a6';
        context.beginPath();
        context.arc(outputPosition.x, outputPosition.y, isFlapping ? 5 : 4, 0, 2 * Math.PI);
        context.fill();
        
        if (isFlapping) {
            context.shadowColor = '#e74c3c';
            context.shadowBlur = 10;
            context.fill();
            context.shadowBlur = 0;
        }

        // Add input labels
        const inputLabels = ['Y', 'V', 'D', 'G', 'H', 'T'];
        context.fillStyle = 'rgba(255, 255, 255, 0.6)';
        context.font = '8px Inter';
        inputLabels.forEach((label, i) => {
            if (i < inputPositions.length) {
                context.fillText(label, inputPositions[i].x - 15, inputPositions[i].y + 2);
            }
        });

        // Add output label
        context.fillText('FLAP', outputPosition.x - 15, outputPosition.y - 10);
    }

    togglePause() {
        this.paused = !this.paused;
    }

    resetGame() {
        this.population = new Population(POP_SIZE);
        this.pipes = [];
        this.frameCount = 0;
        totalGenerations = 0;
        allTimeBest = 0;
        generationBests = [];
    }
}

// Initialize game
const game = new Game();

// Add keyboard controls for user interaction
document.addEventListener('keydown', (event) => {
    switch(event.code) {
        case 'Space':
            event.preventDefault();
            game.togglePause();
            // Add visual feedback
            const pauseBtn = document.querySelector('.control-button.primary');
            pauseBtn.style.transform = 'scale(0.95)';
            setTimeout(() => {
                pauseBtn.style.transform = '';
            }, 150);
            break;
        case 'KeyR':
            event.preventDefault();
            game.resetGame();
            // Add visual feedback
            const resetBtn = document.querySelector('.control-button.secondary');
            resetBtn.style.transform = 'scale(0.95)';
            setTimeout(() => {
                resetBtn.style.transform = '';
            }, 150);
            break;
        case 'KeyS':
            event.preventDefault();
            game.showStats = !game.showStats;
            break;
    }
});

// Add mouse wheel support for speed control
document.addEventListener('wheel', (event) => {
    if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        // This could be used to adjust game speed in the future
    }
});

// Add click effects to canvas
document.getElementById('gameCanvas').addEventListener('click', (event) => {
    const rect = event.target.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Create a ripple effect at click position
    createRippleEffect(x, y);
});

function createRippleEffect(x, y) {
    const ripple = document.createElement('div');
    ripple.style.cssText = `
        position: absolute;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: rgba(0, 212, 255, 0.3);
        border: 2px solid rgba(0, 212, 255, 0.6);
        pointer-events: none;
        left: ${x - 10}px;
        top: ${y - 10}px;
        animation: ripple 0.6s ease-out forwards;
        z-index: 1000;
    `;
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
    
    const canvasContainer = document.querySelector('.canvas-container');
    canvasContainer.style.position = 'relative';
    canvasContainer.appendChild(ripple);
    
    setTimeout(() => {
        ripple.remove();
        style.remove();
    }, 600);
}

// Performance optimized animation loop
let lastTime = 0;
let frameCount = 0;
let fps = 0;

function animate(currentTime) {
    // Calculate FPS
    if (currentTime - lastTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastTime = currentTime;
    }
    frameCount++;

    // Run multiple updates per frame for faster learning
    for (let i = 0; i < SPEED_MULTIPLIER; i++) {
        game.update();
    }
    
    game.show();
    
    // Display FPS if stats are enabled
    if (game.showStats) {
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(`FPS: ${fps}`, WIDTH - 60, 20);
        ctx.fillText(`Speed: ${SPEED_MULTIPLIER}x`, WIDTH - 80, 35);
    }
    
    requestAnimationFrame(animate);
}

// Start the game
animate(0);

// Add instructions to the page
window.addEventListener('load', () => {
    console.log('🎮 AI Flappy Bird - Neural Evolution Loaded!');
    console.log('  SPACE - Pause/Resume');
    console.log('  R - Reset Game');
    console.log('  S - Toggle Stats');
    console.log('');
    console.log('🧠 Watch the neural networks learn to play!');
    console.log('The best performing birds will produce offspring for the next generation.');
    
    // Add a loading animation effect
    const title = document.querySelector('.title');
    title.style.opacity = '0';
    title.style.transform = 'translateY(-20px)';
    
    setTimeout(() => {
        title.style.transition = 'all 0.8s ease-out';
        title.style.opacity = '1';
        title.style.transform = 'translateY(0)';
    }, 100);
    
    // Stagger the appearance of panels
    const panels = document.querySelectorAll('.stats-panel, .neural-panel, .canvas-container');
    panels.forEach((panel, index) => {
        panel.style.opacity = '0';
        panel.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            panel.style.transition = 'all 0.6s ease-out';
            panel.style.opacity = '1';
            panel.style.transform = 'translateY(0)';
        }, 200 + index * 150);
    });

    // Add subtle floating animation to neural panels
    const neuralPanels = document.querySelectorAll('.neural-panel');
    neuralPanels.forEach((panel, index) => {
        panel.style.animation = `float ${3 + index * 0.5}s ease-in-out infinite alternate`;
    });

    // Add the floating animation keyframes
    const style = document.createElement('style');
    style.textContent = `
        @keyframes float {
            from { transform: translateY(0px); }
            to { transform: translateY(-5px); }
        }
    `;
    document.head.appendChild(style);
    
    // Initialize tooltips for better UX
    addTooltips();
});

function addTooltips() {
    const tooltips = [
        { element: '#generation', text: 'Current evolutionary generation' },
        { element: '#alive', text: 'Number of birds still alive in this generation' },
        { element: '#bestScore', text: 'Highest score achieved in current generation' },
        { element: '#avgScore', text: 'Average score of all birds in current generation' },
        { element: '#allTimeBest', text: 'Best score ever achieved across all generations' }
    ];

    tooltips.forEach(({ element, text }) => {
        const el = document.querySelector(element);
        if (el) {
            el.title = text;
            el.style.cursor = 'help';
        }
    });
}
