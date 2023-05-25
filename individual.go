package staticneurogenetic

import (
	"math/rand"
)

type Individual struct {
	Fitness float64
	Genome  []float64
}

func newIndividual(genome_size int) Individual {
	return Individual{
		Genome: make([]float64, genome_size),
	}
}

func (p *Individual) randomize() {
	for i := range p.Genome {
		p.Genome[i] = rand.Float64()*2 - 1
	}
}

func (p *Individual) output(input []float64, layers []int, activation ActivationFunction) []float64 {
	var (
		offset = 0
		in     = input
		out    []float64
	)

	for l := 1; l < len(layers); l++ {
		out = make([]float64, layers[l])

		for o := 0; o < layers[l]; o++ {
			out[o] = p.Genome[offset] // bias
			offset++

			for i := 0; i < layers[l-1]; i++ {
				out[o] += in[i] * p.Genome[offset]
				offset++
			}

			out[o] = activation(out[o]) // activation
		}

		in = out // this output is the input for the next layer
	}

	return out
}

func (p *Individual) monoParentCross(father *Individual) {
	copy(p.Genome, father.Genome)
}

func (p *Individual) divPointCross(father, mother *Individual) {
	point := rand.Intn(len(p.Genome))
	for i := 0; i < point; i++ {
		p.Genome[i] = father.Genome[i]
	}
	for i := point; i < len(p.Genome); i++ {
		p.Genome[i] = mother.Genome[i]
	}
}

func (p *Individual) randomCross(father, mother *Individual) {
	for i := 0; i < len(p.Genome); i++ {
		if rand.Float32() < 0.5 {
			p.Genome[i] = father.Genome[i]
		} else {
			p.Genome[i] = mother.Genome[i]
		}
	}
}

func (p *Individual) oneMutate(mut_rate float32, mut_size float64) {
	if rand.Float32() < mut_rate {
		p.Genome[rand.Intn(len(p.Genome))] += rand.Float64()*2*mut_size - mut_size
	}
}

func (p *Individual) multiMutate(mut_rate float32, mut_size float64) {
	for i := 0; i < len(p.Genome); i++ {
		if rand.Float32() < mut_rate {
			p.Genome[rand.Intn(len(p.Genome))] += rand.Float64()*2*mut_size - mut_size
		}
	}
}
