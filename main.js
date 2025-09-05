// AI Flappy Bird - Version 2.2 - Cache busting fix
const canvas = document.getElementById('gameCanvas');
const ctx = canvas ? canvas.getContext('2d') : null;

// Check if canvas is properly loaded
if (!canvas) {
    console.error('❌ gameCanvas element not found!');
} else if (!ctx) {
    console.error('❌ Failed to get 2D context from gameCanvas!');
} else {
    console.log('✅ Game canvas loaded successfully');
}

const neural1 = document.getElementById('neural1');
const ctx1 = neural1 ? neural1.getContext('2d') : null;
const neural2 = document.getElementById('neural2');
const ctx2 = neural2 ? neural2.getContext('2d') : null;

// New graph canvases with null checks
const fitnessGraph = document.getElementById('fitnessGraph');
const ctxFitness = fitnessGraph ? fitnessGraph.getContext('2d') : null;
const diversityGraph = document.getElementById('diversityGraph');
const ctxDiversity = diversityGraph ? diversityGraph.getContext('2d') : null;
const activityHeatmap = document.getElementById('activityHeatmap');
const ctxActivity = activityHeatmap ? activityHeatmap.getContext('2d') : null;
const genePoolGraph = document.getElementById('genePoolGraph');
const ctxGenePool = genePoolGraph ? genePoolGraph.getContext('2d') : null;
const learningRateGraph = document.getElementById('learningRateGraph');
const ctxLearning = learningRateGraph ? learningRateGraph.getContext('2d') : null;
const speciesGraph = document.getElementById('speciesGraph');
const ctxSpecies = speciesGraph ? speciesGraph.getContext('2d') : null;
const performanceHist = document.getElementById('performanceHist');
const ctxPerformance = performanceHist ? performanceHist.getContext('2d') : null;
const complexityGraph = document.getElementById('complexityGraph');
const ctxComplexity = complexityGraph ? complexityGraph.getContext('2d') : null;
const survivalGraph = document.getElementById('survivalGraph');
const ctxSurvival = survivalGraph ? survivalGraph.getContext('2d') : null;
const confidenceGraph = document.getElementById('confidenceGraph');
const ctxConfidence = confidenceGraph ? confidenceGraph.getContext('2d') : null;
const trainingProgress = document.getElementById('trainingProgress');
const ctxTraining = trainingProgress ? trainingProgress.getContext('2d') : null;

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
let learningRates = [];
let speciesCounts = [];
let survivalRates = [];
let confidenceLevels = [];
let complexityScores = [];

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
        if (!ctx) return;
        
        // Simple yellow circle
        ctx.fillStyle = '#FFD700';
        ctx.beginPath();
        ctx.arc(this.x, this.y, BIRD_SIZE/2, 0, 2 * Math.PI);
        ctx.fill();
        
        // Optional: add a subtle outline
        ctx.strokeStyle = '#FFA500';
        ctx.lineWidth = 1;
        ctx.stroke();
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
        if (!ctx) return;
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
        if (!ctx) return;
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

        // Collect additional metrics for graphs
        const aliveCount = this.birds.filter(bird => !bird.dead).length;
        const survivalRate = aliveCount / this.birds.length;
        survivalRates.push(survivalRate);

        // Learning rate (mutation rate) - could be adaptive
        const currentMutationRate = MUTATION_RATE * (1 + Math.sin(totalGenerations * 0.1) * 0.1);
        learningRates.push(currentMutationRate);

        // Species diversity (simplified)
        const speciesDiversity = Math.max(1, Math.floor(this.bestScore / 50));
        speciesCounts.push(speciesDiversity);

        // Neural complexity (based on connections and activations)
        let totalComplexity = 0;
        this.birds.forEach(bird => {
            if (bird.brain && bird.hiddenActivations) {
                const activationSum = bird.hiddenActivations.reduce((sum, act) => sum + Math.abs(act[0]), 0);
                totalComplexity += activationSum;
            }
        });
        const avgComplexity = totalComplexity / this.birds.length;
        complexityScores.push(avgComplexity);

        // Keep arrays at reasonable size
        if (learningRates.length > 100) learningRates.shift();
        if (speciesCounts.length > 100) speciesCounts.shift();
        if (survivalRates.length > 100) survivalRates.shift();
        if (complexityScores.length > 100) complexityScores.shift();
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
        if (!ctx) return '#87CEEB'; // Fallback color
        let gradient = ctx.createLinearGradient(0, 0, 0, HEIGHT);
        gradient.addColorStop(0, '#87CEEB');
        gradient.addColorStop(1, '#98FB98');
        return gradient;
    }

    update() {
        if (this.paused) return;
        
        try {
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
        } catch (error) {
            console.error('❌ Error in game update:', error);
        }
    }

    show() {
        if (!ctx) {
            console.warn('⚠️ Canvas context not available for rendering');
            return;
        }
        
        try {
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

            // Draw additional graphs
            this.drawFitnessGraph();
            this.drawDiversityGraph();
            this.drawActivityHeatmap();
            this.drawGenePoolGraph();
            this.drawLearningRateGraph();
            this.drawSpeciesGraph();
            this.drawPerformanceHistogram();
            this.drawComplexityGraph();
            this.drawSurvivalGraph();
            this.drawConfidenceGraph();
            this.drawTrainingProgress();
        } catch (error) {
            console.error('❌ Error in game show:', error);
        }
    }

    drawClouds() {
        if (!ctx) return;
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
        try {
            let aliveBirds = this.population.getAliveBirds();
            
            // Safely update DOM elements
            const generationEl = document.getElementById('generation');
            const aliveEl = document.getElementById('alive');
            const bestScoreEl = document.getElementById('bestScore');
            
            if (generationEl) generationEl.textContent = this.population.generation;
            if (aliveEl) aliveEl.textContent = aliveBirds.length;
            if (bestScoreEl) bestScoreEl.textContent = this.population.bestScore;
            
            // Update stat value classes based on values
            if (aliveEl) {
                if (aliveBirds.length > 50) {
                    aliveEl.className = 'stat-value';
                } else if (aliveBirds.length > 10) {
                    aliveEl.className = 'stat-value warning';
                } else {
                    aliveEl.className = 'stat-value danger';
                }
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
                if (statusIndicator) statusIndicator.className = 'status-indicator paused';
                if (gameStatus) {
                    gameStatus.textContent = 'Paused';
                    gameStatus.className = 'stat-value warning';
                }
            } else if (aliveBirds.length === 0) {
                if (statusIndicator) statusIndicator.className = 'status-indicator stopped';
                if (gameStatus) {
                    gameStatus.textContent = 'Evolving';
                    gameStatus.className = 'stat-value danger';
                }
            } else {
                if (statusIndicator) statusIndicator.className = 'status-indicator running';
                if (gameStatus) {
                    gameStatus.textContent = 'Learning';
                    gameStatus.className = 'stat-value';
                }
            }
        } catch (error) {
            console.error('❌ Error in updateInfoDisplay:', error);
        }
    }    visualizeNeural(bird, context, label) {
        if (!context) return;
        context.clearRect(0, 0, 400, 300);
        
        if (!bird || !bird.hiddenActivations) {
            // Show "No Data" message when bird is not available
            context.fillStyle = 'rgba(255, 255, 255, 0.3)';
            context.font = '16px Inter';
            context.fillText('No Active Bird', 150, 150);
            return;
        }

        // Background with subtle gradient
        const gradient = context.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        context.fillStyle = gradient;
        context.fillRect(0, 0, 400, 300);

        // Title and score with better typography
        context.fillStyle = '#00d4ff';
        context.font = 'bold 16px Orbitron, monospace';
        context.fillText(label, 10, 25);
        
        context.fillStyle = '#2ecc71';
        context.font = '14px Inter';
        context.fillText(`Score: ${bird.score}`, 10, 45);
        context.fillText(`Fitness: ${Math.round(bird.fitness * 1000)}`, 10, 65);

        // Network visualization with improved layout for larger canvas
        let inputPositions = [
            {x: 50, y: 100}, {x: 50, y: 120}, {x: 50, y: 140},
            {x: 50, y: 160}, {x: 50, y: 180}, {x: 50, y: 200}
        ];
        
        let hiddenPositions = [];
        for (let i = 0; i < 12; i++) {
            hiddenPositions.push({
                x: 150 + (i % 4) * 30,
                y: 100 + Math.floor(i / 4) * 30
            });
        }
        
        let outputPosition = {x: 330, y: 160};

        // Draw connections with activation-based opacity
        context.lineWidth = 2;
        
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
            context.arc(pos.x, pos.y, 6, 0, 2 * Math.PI);
            context.fill();
            
            // Add glow effect
            context.shadowColor = '#3498db';
            context.shadowBlur = 8;
            context.fill();
            context.shadowBlur = 0;
        }

        // Hidden nodes with activation visualization
        for (let i = 0; i < hiddenPositions.length && i < bird.hiddenActivations.length; i++) {
            let activation = Math.abs(bird.hiddenActivations[i][0]);
            let intensity = Math.min(255, activation * 255);
            
            // Create pulsing effect for high activation
            let radius = 6 + (activation > 0.7 ? Math.sin(Date.now() * 0.01) * 2 : 0);
            
            context.fillStyle = `rgb(${intensity}, ${255 - intensity}, 50)`;
            context.beginPath();
            context.arc(hiddenPositions[i].x, hiddenPositions[i].y, radius, 0, 2 * Math.PI);
            context.fill();
            
            // Add glow for highly active neurons
            if (activation > 0.5) {
                context.shadowColor = `rgb(${intensity}, ${255 - intensity}, 50)`;
                context.shadowBlur = 12;
                context.fill();
                context.shadowBlur = 0;
            }
        }

        // Output node with decision visualization
        const isFlapping = bird.brain && bird.framesSinceFlap < 5;
        context.fillStyle = isFlapping ? '#e74c3c' : '#95a5a6';
        context.beginPath();
        context.arc(outputPosition.x, outputPosition.y, isFlapping ? 8 : 6, 0, 2 * Math.PI);
        context.fill();
        
        if (isFlapping) {
            context.shadowColor = '#e74c3c';
            context.shadowBlur = 15;
            context.fill();
            context.shadowBlur = 0;
        }

        // Add input labels
        const inputLabels = ['Y', 'V', 'D', 'G', 'H', 'T'];
        context.fillStyle = 'rgba(255, 255, 255, 0.8)';
        context.font = '12px Inter';
        inputLabels.forEach((label, i) => {
            if (i < inputPositions.length) {
                context.fillText(label, inputPositions[i].x - 25, inputPositions[i].y + 4);
            }
        });

        // Add output label
        context.fillText('FLAP', outputPosition.x - 20, outputPosition.y - 15);
    }

    // New graph visualization functions
    drawFitnessGraph() {
        if (!ctxFitness) return;
        ctxFitness.clearRect(0, 0, 400, 300);

        // Background
        const gradient = ctxFitness.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        ctxFitness.fillStyle = gradient;
        ctxFitness.fillRect(0, 0, 400, 300);

        // Title
        ctxFitness.fillStyle = '#00d4ff';
        ctxFitness.font = 'bold 16px Orbitron, monospace';
        ctxFitness.fillText('Fitness Evolution', 10, 25);

        if (generationBests.length === 0) {
            ctxFitness.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctxFitness.font = '14px Inter';
            ctxFitness.fillText('Waiting for data...', 140, 150);
            return;
        }

        // Graph area dimensions
        const graphLeft = 50;
        const graphTop = 40;
        const graphWidth = 320;
        const graphHeight = 220;
        const graphBottom = graphTop + graphHeight;
        const graphRight = graphLeft + graphWidth;

        // Draw grid lines
        ctxFitness.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctxFitness.lineWidth = 1;

        // Vertical grid lines (generations)
        for (let i = 0; i <= 10; i++) {
            const x = graphLeft + (i * graphWidth) / 10;
            ctxFitness.beginPath();
            ctxFitness.moveTo(x, graphTop);
            ctxFitness.lineTo(x, graphBottom);
            ctxFitness.stroke();

            // Generation labels
            if (i % 2 === 0) {
                ctxFitness.fillStyle = 'rgba(255, 255, 255, 0.6)';
                ctxFitness.font = '10px Inter';
                const genNum = Math.round((i * generationBests.length) / 10);
                ctxFitness.fillText(genNum.toString(), x - 5, graphBottom + 15);
            }
        }

        // Horizontal grid lines (fitness values)
        const maxFitness = Math.max(...generationBests);
        for (let i = 0; i <= 10; i++) {
            const y = graphBottom - (i * graphHeight) / 10;
            ctxFitness.beginPath();
            ctxFitness.moveTo(graphLeft, y);
            ctxFitness.lineTo(graphRight, y);
            ctxFitness.stroke();

            // Fitness value labels
            ctxFitness.fillStyle = 'rgba(255, 255, 255, 0.6)';
            ctxFitness.font = '10px Inter';
            const fitnessValue = Math.round((i * maxFitness) / 10);
            ctxFitness.fillText(fitnessValue.toString(), graphLeft - 35, y + 3);
        }

        // Draw axis labels
        ctxFitness.fillStyle = '#00d4ff';
        ctxFitness.font = '12px Inter';
        ctxFitness.fillText('Generations', graphLeft + graphWidth/2 - 30, graphBottom + 30);
        ctxFitness.save();
        ctxFitness.translate(15, graphTop + graphHeight/2);
        ctxFitness.rotate(-Math.PI/2);
        ctxFitness.fillText('Fitness Score', 0, 0);
        ctxFitness.restore();

        // Draw fitness line graph
        ctxFitness.strokeStyle = '#00d4ff';
        ctxFitness.lineWidth = 3;
        ctxFitness.beginPath();

        const scaleX = graphWidth / Math.max(generationBests.length - 1, 1);
        const scaleY = graphHeight / Math.max(maxFitness, 1);

        generationBests.forEach((fitness, i) => {
            const x = graphLeft + i * scaleX;
            const y = graphBottom - fitness * scaleY;

            if (i === 0) {
                ctxFitness.moveTo(x, y);
            } else {
                ctxFitness.lineTo(x, y);
            }
        });

        ctxFitness.stroke();

        // Add data points with values
        generationBests.forEach((fitness, i) => {
            const x = graphLeft + i * scaleX;
            const y = graphBottom - fitness * scaleY;

            // Draw point
            ctxFitness.fillStyle = '#00d4ff';
            ctxFitness.beginPath();
            ctxFitness.arc(x, y, 4, 0, Math.PI * 2);
            ctxFitness.fill();

            // Show value on hover points (every 5th point or last point)
            if (i % 5 === 0 || i === generationBests.length - 1) {
                ctxFitness.fillStyle = '#ffffff';
                ctxFitness.font = '10px Inter';
                ctxFitness.fillText(Math.round(fitness).toString(), x - 8, y - 8);
            }
        });

        // Current generation marker
        if (generationBests.length > 0) {
            const currentX = graphLeft + (generationBests.length - 1) * scaleX;
            const currentY = graphBottom - generationBests[generationBests.length - 1] * scaleY;

            ctxFitness.strokeStyle = '#ff6b6b';
            ctxFitness.lineWidth = 2;
            ctxFitness.beginPath();
            ctxFitness.arc(currentX, currentY, 8, 0, Math.PI * 2);
            ctxFitness.stroke();

            // Current value label
            ctxFitness.fillStyle = '#ff6b6b';
            ctxFitness.font = 'bold 12px Inter';
            ctxFitness.fillText(`Current: ${Math.round(generationBests[generationBests.length - 1])}`,
                              currentX + 12, currentY - 5);
        }

        // Statistics
        ctxFitness.fillStyle = '#2ecc71';
        ctxFitness.font = '12px Inter';
        ctxFitness.fillText(`Max: ${Math.round(maxFitness)}`, graphRight - 80, 35);
        ctxFitness.fillText(`Avg: ${Math.round(generationBests.reduce((a,b) => a+b, 0) / generationBests.length)}`, graphRight - 80, 50);
        ctxFitness.fillText(`Total Gens: ${generationBests.length}`, graphRight - 80, 65);
    }

    drawDiversityGraph() {
        if (!ctxDiversity) return;
        ctxDiversity.clearRect(0, 0, 400, 300);
        
        // Background
        const gradient = ctxDiversity.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        ctxDiversity.fillStyle = gradient;
        ctxDiversity.fillRect(0, 0, 400, 300);

        // Title
        ctxDiversity.fillStyle = '#00d4ff';
        ctxDiversity.font = 'bold 16px Orbitron, monospace';
        ctxDiversity.fillText('Population Diversity', 10, 25);

        if (!this.population || !this.population.birds) {
            ctxDiversity.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctxDiversity.font = '14px Inter';
            ctxDiversity.fillText('No population data', 140, 150);
            return;
        }

        // Calculate diversity metrics
        const aliveCount = this.population.birds.filter(bird => bird.alive).length;
        const totalFitness = this.population.birds.reduce((sum, bird) => sum + bird.fitness, 0);
        const avgFitness = totalFitness / this.population.birds.length;

        // Draw diversity bars
        const barWidth = 60;
        const barSpacing = 80;
        const startX = 50;
        
        // Alive birds bar
        ctxDiversity.fillStyle = '#2ecc71';
        const aliveHeight = (aliveCount / POP_SIZE) * 200;
        ctxDiversity.fillRect(startX, 250 - aliveHeight, barWidth, aliveHeight);
        
        ctxDiversity.fillStyle = '#ffffff';
        ctxDiversity.font = '12px Inter';
        ctxDiversity.fillText('Alive', startX + 10, 270);
        ctxDiversity.fillText(aliveCount.toString(), startX + 15, 250 - aliveHeight - 10);

        // Average fitness bar
        ctxDiversity.fillStyle = '#e74c3c';
        const fitnessHeight = Math.min(avgFitness * 1000, 200);
        ctxDiversity.fillRect(startX + barSpacing, 250 - fitnessHeight, barWidth, fitnessHeight);
        
        ctxDiversity.fillStyle = '#ffffff';
        ctxDiversity.fillText('Avg Fit', startX + barSpacing + 5, 270);
        ctxDiversity.fillText(Math.round(avgFitness * 1000).toString(), startX + barSpacing + 10, 250 - fitnessHeight - 10);
    }

    drawActivityHeatmap() {
        if (!ctxActivity) return;
        ctxActivity.clearRect(0, 0, 400, 300);
        
        // Background
        const gradient = ctxActivity.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        ctxActivity.fillStyle = gradient;
        ctxActivity.fillRect(0, 0, 400, 300);

        // Title
        ctxActivity.fillStyle = '#00d4ff';
        ctxActivity.font = 'bold 16px Orbitron, monospace';
        ctxActivity.fillText('Neural Activity Heatmap', 10, 25);

        if (!this.population || !this.population.birds) {
            ctxActivity.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctxActivity.font = '14px Inter';
            ctxActivity.fillText('No activity data', 140, 150);
            return;
        }

        // Create heatmap grid
        const gridSize = 20;
        const startX = 50;
        const startY = 60;
        
        for (let x = 0; x < 12; x++) {
            for (let y = 0; y < 8; y++) {
                // Calculate average activation for this neuron across population
                let totalActivation = 0;
                let count = 0;
                
                this.population.birds.forEach(bird => {
                    if (bird.hiddenActivations && bird.hiddenActivations[x + y * 4]) {
                        totalActivation += Math.abs(bird.hiddenActivations[x + y * 4][0]);
                        count++;
                    }
                });
                
                const avgActivation = count > 0 ? totalActivation / count : 0;
                const intensity = Math.min(255, avgActivation * 255);
                
                // Color based on activation intensity
                ctxActivity.fillStyle = `rgb(${intensity}, ${100}, ${255 - intensity})`;
                ctxActivity.fillRect(startX + x * gridSize, startY + y * gridSize, gridSize - 2, gridSize - 2);
                
                // Add activation value
                if (avgActivation > 0.1) {
                    ctxActivity.fillStyle = '#ffffff';
                    ctxActivity.font = '8px Inter';
                    ctxActivity.fillText(avgActivation.toFixed(1), startX + x * gridSize + 2, startY + y * gridSize + 12);
                }
            }
        }

        // Add labels
        ctxActivity.fillStyle = '#ffffff';
        ctxActivity.font = '12px Inter';
        ctxActivity.fillText('Hidden Layer Neurons', 120, 40);
    }

    drawGenePoolGraph() {
        if (!ctxGenePool) return;
        ctxGenePool.clearRect(0, 0, 400, 300);
        
        // Background
        const gradient = ctxGenePool.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        ctxGenePool.fillStyle = gradient;
        ctxGenePool.fillRect(0, 0, 400, 300);

        // Title
        ctxGenePool.fillStyle = '#00d4ff';
        ctxGenePool.font = 'bold 16px Orbitron, monospace';
        ctxGenePool.fillText('Gene Pool Analysis', 10, 25);

        if (!this.population || !this.population.birds) {
            ctxGenePool.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctxGenePool.font = '14px Inter';
            ctxGenePool.fillText('No gene data', 150, 150);
            return;
        }

        // Analyze gene diversity
        const geneStats = {
            inputHidden: { total: 0, unique: 0 },
            hiddenOutput: { total: 0, unique: 0 }
        };

        const geneSet = new Set();
        
        this.population.birds.forEach(bird => {
            if (bird.brain) {
                // Count unique weight patterns
                const weightsStr = JSON.stringify(bird.brain.weightsIH) + JSON.stringify(bird.brain.weightsHO);
                geneSet.add(weightsStr);
                
                geneStats.inputHidden.total += bird.brain.weightsIH.flat().length;
                geneStats.hiddenOutput.total += bird.brain.weightsHO.flat().length;
            }
        });

        geneStats.inputHidden.unique = geneSet.size;
        geneStats.hiddenOutput.unique = geneStats.inputHidden.unique; // Same for both layers

        // Draw gene diversity visualization
        const centerX = 200;
        const centerY = 150;
        const radius = 80;

        // Draw circles representing gene diversity
        ctxGenePool.strokeStyle = '#00d4ff';
        ctxGenePool.lineWidth = 3;
        ctxGenePool.beginPath();
        ctxGenePool.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctxGenePool.stroke();

        // Fill based on diversity
        const diversityRatio = geneStats.inputHidden.unique / this.population.birds.length;
        const fillRadius = radius * diversityRatio;
        
        ctxGenePool.fillStyle = `rgba(0, 212, 255, ${diversityRatio * 0.5})`;
        ctxGenePool.beginPath();
        ctxGenePool.arc(centerX, centerY, fillRadius, 0, Math.PI * 2);
        ctxGenePool.fill();

        // Add text
        ctxGenePool.fillStyle = '#ffffff';
        ctxGenePool.font = '14px Inter';
        ctxGenePool.fillText(`Unique Genotypes: ${geneStats.inputHidden.unique}`, 100, 220);
        ctxGenePool.fillText(`Total Population: ${this.population.birds.length}`, 100, 240);
        ctxGenePool.fillText(`Diversity: ${(diversityRatio * 100).toFixed(1)}%`, 100, 260);
    }

    // Additional graph functions for comprehensive AI monitoring
    drawLearningRateGraph() {
        if (!ctxLearning) return;
        ctxLearning.clearRect(0, 0, 400, 300);

        // Background
        const gradient = ctxLearning.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        ctxLearning.fillStyle = gradient;
        ctxLearning.fillRect(0, 0, 400, 300);

        // Title
        ctxLearning.fillStyle = '#00d4ff';
        ctxLearning.font = 'bold 16px Orbitron, monospace';
        ctxLearning.fillText('Learning Rate Evolution', 10, 25);

        if (learningRates.length === 0) {
            ctxLearning.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctxLearning.font = '14px Inter';
            ctxLearning.fillText('No learning data yet...', 120, 150);
            return;
        }

        // Graph area
        const graphLeft = 50, graphTop = 40, graphWidth = 320, graphHeight = 220;

        // Draw learning rate line
        ctxLearning.strokeStyle = '#ff6b6b';
        ctxLearning.lineWidth = 3;
        ctxLearning.beginPath();

        const maxRate = Math.max(...learningRates, MUTATION_RATE);
        const scaleX = graphWidth / Math.max(learningRates.length - 1, 1);
        const scaleY = graphHeight / maxRate;

        learningRates.forEach((rate, i) => {
            const x = graphLeft + i * scaleX;
            const y = graphTop + graphHeight - rate * scaleY;

            if (i === 0) ctxLearning.moveTo(x, y);
            else ctxLearning.lineTo(x, y);
        });

        ctxLearning.stroke();

        // Current value
        ctxLearning.fillStyle = '#ff6b6b';
        ctxLearning.font = '12px Inter';
        ctxLearning.fillText(`Current: ${(learningRates[learningRates.length - 1] || MUTATION_RATE).toFixed(3)}`, 280, 35);
    }

    drawSpeciesGraph() {
        if (!ctxSpecies) return;
        ctxSpecies.clearRect(0, 0, 400, 300);

        // Background
        const gradient = ctxSpecies.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        ctxSpecies.fillStyle = gradient;
        ctxSpecies.fillRect(0, 0, 400, 300);

        // Title
        ctxSpecies.fillStyle = '#00d4ff';
        ctxSpecies.font = 'bold 16px Orbitron, monospace';
        ctxSpecies.fillText('Species Diversity', 10, 25);

        if (!this.population || !this.population.birds) {
            ctxSpecies.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctxSpecies.font = '14px Inter';
            ctxSpecies.fillText('No population data', 140, 150);
            return;
        }

        // Simple species detection based on fitness patterns
        const species = {};
        this.population.birds.forEach((bird, i) => {
            const fitnessRange = Math.floor(bird.fitness / 100);
            species[fitnessRange] = (species[fitnessRange] || 0) + 1;
        });

        // Draw species bars
        const barWidth = 30;
        let x = 50;
        Object.entries(species).forEach(([range, count]) => {
            const height = (count / POP_SIZE) * 200;

            ctxSpecies.fillStyle = `hsl(${parseInt(range) * 60}, 70%, 50%)`;
            ctxSpecies.fillRect(x, 250 - height, barWidth, height);

            ctxSpecies.fillStyle = '#ffffff';
            ctxSpecies.font = '10px Inter';
            ctxSpecies.fillText(`${range * 100}-${(range + 1) * 100}`, x, 270);
            ctxSpecies.fillText(count.toString(), x + 5, 250 - height - 5);

            x += barWidth + 10;
        });

        ctxSpecies.fillStyle = '#00d4ff';
        ctxSpecies.font = '12px Inter';
        ctxSpecies.fillText(`Species: ${Object.keys(species).length}`, 280, 35);
    }

    drawPerformanceHistogram() {
        if (!ctxPerformance) return;
        ctxPerformance.clearRect(0, 0, 400, 300);

        // Background
        const gradient = ctxPerformance.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        ctxPerformance.fillStyle = gradient;
        ctxPerformance.fillRect(0, 0, 400, 300);

        // Title
        ctxPerformance.fillStyle = '#00d4ff';
        ctxPerformance.font = 'bold 16px Orbitron, monospace';
        ctxPerformance.fillText('Performance Distribution', 10, 25);

        if (!this.population || !this.population.birds) {
            ctxPerformance.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctxPerformance.font = '14px Inter';
            ctxPerformance.fillText('No performance data', 130, 150);
            return;
        }

        // Create histogram bins
        const scores = this.population.birds.map(bird => bird.score);
        const maxScore = Math.max(...scores);
        const bins = Array(10).fill(0);

        scores.forEach(score => {
            const binIndex = Math.min(9, Math.floor((score / Math.max(maxScore, 1)) * 10));
            bins[binIndex]++;
        });

        // Draw histogram
        const barWidth = 25;
        const maxCount = Math.max(...bins);

        bins.forEach((count, i) => {
            const x = 50 + i * (barWidth + 5);
            const height = (count / Math.max(maxCount, 1)) * 200;

            ctxPerformance.fillStyle = '#2ecc71';
            ctxPerformance.fillRect(x, 250 - height, barWidth, height);

            ctxPerformance.fillStyle = '#ffffff';
            ctxPerformance.font = '10px Inter';
            ctxPerformance.fillText(Math.round(maxScore * i / 10).toString(), x, 270);
        });

        // Statistics
        ctxPerformance.fillStyle = '#00d4ff';
        ctxPerformance.font = '12px Inter';
        const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        ctxPerformance.fillText(`Avg: ${Math.round(avgScore)}`, 280, 35);
        ctxPerformance.fillText(`Max: ${maxScore}`, 280, 50);
    }

    drawComplexityGraph() {
        if (!ctxComplexity) return;
        ctxComplexity.clearRect(0, 0, 400, 300);

        // Background
        const gradient = ctxComplexity.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        ctxComplexity.fillStyle = gradient;
        ctxComplexity.fillRect(0, 0, 400, 300);

        // Title
        ctxComplexity.fillStyle = '#00d4ff';
        ctxComplexity.font = 'bold 16px Orbitron, monospace';
        ctxComplexity.fillText('Neural Complexity', 10, 25);

        if (complexityScores.length === 0) {
            ctxComplexity.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctxComplexity.font = '14px Inter';
            ctxComplexity.fillText('Analyzing complexity...', 120, 150);
            return;
        }

        // Draw complexity trend
        ctxComplexity.strokeStyle = '#9b59b6';
        ctxComplexity.lineWidth = 3;
        ctxComplexity.beginPath();

        const maxComplexity = Math.max(...complexityScores);
        const scaleX = 320 / Math.max(complexityScores.length - 1, 1);
        const scaleY = 200 / Math.max(maxComplexity, 1);

        complexityScores.forEach((complexity, i) => {
            const x = 50 + i * scaleX;
            const y = 260 - complexity * scaleY;

            if (i === 0) ctxComplexity.moveTo(x, y);
            else ctxComplexity.lineTo(x, y);
        });

        ctxComplexity.stroke();

        // Current complexity
        ctxComplexity.fillStyle = '#9b59b6';
        ctxComplexity.font = '12px Inter';
        const currentComplexity = complexityScores[complexityScores.length - 1] || 0;
        ctxComplexity.fillText(`Complexity: ${currentComplexity.toFixed(2)}`, 250, 35);
    }

    drawSurvivalGraph() {
        if (!ctxSurvival) return;
        ctxSurvival.clearRect(0, 0, 400, 300);

        // Background
        const gradient = ctxSurvival.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        ctxSurvival.fillStyle = gradient;
        ctxSurvival.fillRect(0, 0, 400, 300);

        // Title
        ctxSurvival.fillStyle = '#00d4ff';
        ctxSurvival.font = 'bold 16px Orbitron, monospace';
        ctxSurvival.fillText('Survival Rate', 10, 25);

        if (survivalRates.length === 0) {
            ctxSurvival.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctxSurvival.font = '14px Inter';
            ctxSurvival.fillText('Tracking survival...', 130, 150);
            return;
        }

        // Draw survival rate line
        ctxSurvival.strokeStyle = '#e67e22';
        ctxSurvival.lineWidth = 3;
        ctxSurvival.beginPath();

        const scaleX = 320 / Math.max(survivalRates.length - 1, 1);
        const scaleY = 200;

        survivalRates.forEach((rate, i) => {
            const x = 50 + i * scaleX;
            const y = 260 - rate * scaleY;

            if (i === 0) ctxSurvival.moveTo(x, y);
            else ctxSurvival.lineTo(x, y);
        });

        ctxSurvival.stroke();

        // Current survival rate
        ctxSurvival.fillStyle = '#e67e22';
        ctxSurvival.font = '12px Inter';
        const currentRate = survivalRates[survivalRates.length - 1] || 0;
        ctxSurvival.fillText(`Survival: ${(currentRate * 100).toFixed(1)}%`, 250, 35);
    }

    drawConfidenceGraph() {
        if (!ctxConfidence) return;
        ctxConfidence.clearRect(0, 0, 400, 300);

        // Background
        const gradient = ctxConfidence.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        ctxConfidence.fillStyle = gradient;
        ctxConfidence.fillRect(0, 0, 400, 300);

        // Title
        ctxConfidence.fillStyle = '#00d4ff';
        ctxConfidence.font = 'bold 16px Orbitron, monospace';
        ctxConfidence.fillText('Decision Confidence', 10, 25);

        if (!this.population || !this.population.birds) {
            ctxConfidence.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctxConfidence.font = '14px Inter';
            ctxConfidence.fillText('Analyzing decisions...', 120, 150);
            return;
        }

        // Calculate average confidence from neural network outputs
        let totalConfidence = 0;
        let birdCount = 0;

        this.population.birds.forEach(bird => {
            if (bird.brain && bird.hiddenActivations) {
                try {
                    // Use the same inputs as the think method
                    let inputs = [0.5, 0, 1.0, 0.5, 0.0, 0]; // Default inputs
                    
                    // Try to get real inputs if available
                    if (this.pipes.length > 0) {
                        let closest = null;
                        let closestD = Infinity;
                        for (let pipe of this.pipes) {
                            let d = pipe.x + PIPE_WIDTH - bird.x;
                            if (d > -PIPE_WIDTH && d < closestD) {
                                closest = pipe;
                                closestD = d;
                            }
                        }
                        
                        if (closest) {
                            let distanceToGap = closest.x + PIPE_WIDTH - bird.x;
                            let gapCenter = (closest.top + closest.bottom) / 2;
                            let heightDiff = bird.y - gapCenter;
                            
                            inputs = [
                                bird.y / HEIGHT,
                                bird.velocity / 20,
                                distanceToGap / WIDTH,
                                gapCenter / HEIGHT,
                                heightDiff / HEIGHT,
                                bird.framesSinceFlap / 60
                            ];
                        }
                    }
                    
                    const result = bird.brain.feedforward(inputs);
                    const confidence = Math.abs(result.output[0][0] - 0.5) * 2; // Decision strength (0-1)
                    totalConfidence += confidence;
                    birdCount++;
                } catch (error) {
                    console.warn('⚠️ Error calculating confidence for bird:', error);
                }
            }
        });

        const avgConfidence = birdCount > 0 ? totalConfidence / birdCount : 0;

        // Draw confidence meter
        const centerX = 200;
        const centerY = 180;
        const radius = 80;

        // Background circle
        ctxConfidence.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctxConfidence.lineWidth = 20;
        ctxConfidence.beginPath();
        ctxConfidence.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctxConfidence.stroke();

        // Confidence arc
        ctxConfidence.strokeStyle = '#f39c12';
        ctxConfidence.lineWidth = 20;
        ctxConfidence.beginPath();
        ctxConfidence.arc(centerX, centerY, radius, -Math.PI/2, -Math.PI/2 + (avgConfidence * Math.PI * 2));
        ctxConfidence.stroke();

        // Center text
        ctxConfidence.fillStyle = '#ffffff';
        ctxConfidence.font = 'bold 24px Orbitron';
        ctxConfidence.fillText(`${(avgConfidence * 100).toFixed(0)}%`, centerX - 30, centerY + 8);

        ctxConfidence.font = '14px Inter';
        ctxConfidence.fillText('Confidence', centerX - 35, centerY + 30);
    }

    drawTrainingProgress() {
        if (!ctxTraining) return;
        ctxTraining.clearRect(0, 0, 400, 300);

        // Background
        const gradient = ctxTraining.createLinearGradient(0, 0, 400, 300);
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.9)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
        ctxTraining.fillStyle = gradient;
        ctxTraining.fillRect(0, 0, 400, 300);

        // Title
        ctxTraining.fillStyle = '#00d4ff';
        ctxTraining.font = 'bold 16px Orbitron, monospace';
        ctxTraining.fillText('Training Progress', 10, 25);

        // Progress metrics
        const progress = Math.min(100, (totalGenerations / 100) * 100);
        const improvement = generationBests.length > 1 ?
            ((generationBests[generationBests.length - 1] - generationBests[0]) / Math.max(generationBests[0], 1)) * 100 : 0;

        // Draw progress bar
        ctxTraining.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctxTraining.lineWidth = 20;
        ctxTraining.beginPath();
        ctxTraining.moveTo(50, 100);
        ctxTraining.lineTo(350, 100);
        ctxTraining.stroke();

        ctxTraining.strokeStyle = '#2ecc71';
        ctxTraining.beginPath();
        ctxTraining.moveTo(50, 100);
        ctxTraining.lineTo(50 + (progress * 3), 100);
        ctxTraining.stroke();

        // Progress text
        ctxTraining.fillStyle = '#ffffff';
        ctxTraining.font = 'bold 18px Orbitron';
        ctxTraining.fillText(`${Math.round(progress)}%`, 180, 85);

        // Stats
        ctxTraining.fillStyle = '#00d4ff';
        ctxTraining.font = '14px Inter';
        ctxTraining.fillText(`Generations: ${totalGenerations}`, 50, 140);
        ctxTraining.fillText(`Best Score: ${allTimeBest}`, 50, 160);
        ctxTraining.fillText(`Improvement: ${improvement.toFixed(1)}%`, 50, 180);

        // Training phases
        const phases = ['Initializing', 'Exploring', 'Optimizing', 'Converging'];
        const currentPhase = Math.min(3, Math.floor(progress / 25));

        ctxTraining.fillStyle = '#f39c12';
        ctxTraining.font = '16px Orbitron';
        ctxTraining.fillText(`Phase: ${phases[currentPhase]}`, 50, 220);
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

// Verify game initialization
if (!game) {
    console.error('❌ Failed to initialize game!');
} else {
    console.log('✅ Game initialized successfully');
    console.log('Population size:', game.population ? game.population.birds.length : 'undefined');
}

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
    try {
        // Calculate FPS
        if (currentTime - lastTime >= 1000) {
            fps = frameCount;
            frameCount = 0;
            lastTime = currentTime;
        }
        frameCount++;

        // Run multiple updates per frame for faster learning
        for (let i = 0; i < SPEED_MULTIPLIER; i++) {
            if (game && typeof game.update === 'function') {
                game.update();
            }
        }
        
        if (game && typeof game.show === 'function') {
            game.show();
        }
        
        // Display FPS if stats are enabled
        if (ctx && game && game.showStats) {
            ctx.fillStyle = 'white';
            ctx.font = '12px Arial';
            ctx.fillText(`FPS: ${fps}`, WIDTH - 60, 20);
            ctx.fillText(`Speed: ${SPEED_MULTIPLIER}x`, WIDTH - 80, 35);
        }
        
        requestAnimationFrame(animate);
    } catch (error) {
        console.error('❌ Error in animate function:', error);
        console.log('Stopping animation due to error');
    }
}

// Start the game
console.log('🚀 Starting animation loop...');
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
