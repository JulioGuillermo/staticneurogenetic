# StaticNeuroGenetic

This library use a genetic algorithm to fit a neural network weights.
This is useful when you don't have a dataset to train your neural network,
for example when you need an agent to interact with an environment
or to learn to play some games.

## How to use

Create an evaluation function

```go
// Eval an individual
func eval(agents *staticneurogenetic.SNG, individual int) {
    inputs := [][]float64 {
        []float64 {0, 0},
        []float64 {0, 1},
        []float64 {1, 0},
        []float64 {1, 1},
    }
    targets := []float64 {
        1,
        0,
        0,
        1
    }

    for i, input := range inputs {
        // Get individual output ([]float64)
        output := agents.Output(individual, input)
        // Calculate how wrong is the output
        dif := math.abs(targets[i] - output[0])
        // Added to the fitness
        agents.AddFitness(individual, 1 - dif)
    }
}

// Eval each individual
func evalAll(agents *staticneurogenetic.SNG) {
    for i := range agents.Population {
        eval(agents, i)
    }
}
```

Create a new set of agents

```go
agents := staticneurogenetic.NewSNG(
    []int{2, 3, 1},                     //Neural network's layers [input, hiddens..., output]
    staticneurogenetic.Sigmoid,         //Activation function for the neural network
    300,                                //PopulationSize (number of individual to work with)
    10,                                 //Survivors (number of individual that will not change in next generation and to use as parents)
    0.1,                                //MutRate (probability to mutate a new individual)
    0.1,                                //MutSize (how big the mutation will be)
    staticneurogenetic.OneMutation,     //MutType
    staticneurogenetic.DivPointCross,   //CrossType
)
```

To train the agents we just need to get the next generation

```go
for i := 0; i < 300; i++ {
    agents.ResetFitness() //Set all fitness to 0, for use AddFitness
    evalAll(agents)
    agents.NextGeneration() //Evolve each neural networks
}
```
