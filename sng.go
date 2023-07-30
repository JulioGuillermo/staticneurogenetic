package staticneurogenetic

import (
	"bytes"
	"encoding/gob"
	"errors"
	"io/ioutil"
	"math/rand"
	"sync"
)

type MutationType byte

const (
	OneMutation = MutationType(iota)
	MultiMutation
)

type CrossType byte

const (
	DivPointCross = CrossType(iota)
	RandCross
	MonoParent
	ArithmeticCross
)

type SNG struct {
	last_best_index int

	Layers     []int
	Activation Activation

	Population []Individual
	Survivors  int
	MutRate    float32
	MutSize    float64

	MutationType MutationType
	CrossType    CrossType

	Generation uint64
}

func NewSNG(
	layers []int,
	activation Activation,
	population_size int,
	survivors int,
	mutation_rate float32,
	mutation_size float64,
	mutation_type MutationType,
	cross_type CrossType,
) *SNG {
	genome_size := GetGenomeSize(layers)
	population := make([]Individual, population_size)
	for i := 0; i < population_size; i++ {
		population[i] = NewIndividual(genome_size)
		population[i].Randomize()
	}

	return &SNG{
		Layers:     layers,
		Activation: activation,

		Population:   population,
		Survivors:    survivors,
		MutRate:      mutation_rate,
		MutSize:      mutation_size,
		MutationType: mutation_type,
		CrossType:    cross_type,
	}
}

func (p *SNG) GetGeneration() uint64 {
	return p.Generation
}

// Evolution
func (p *SNG) nextGenerationIndividual(ctl chan int, wg *sync.WaitGroup) {
	var father, mother int
	for i := range ctl {
		// Select father and mother from survivors
		father, mother = p.SelectFatherMother()

		// Create the new individual
		switch p.CrossType {
		case MonoParent:
			p.Population[i].MonoParentCross(&p.Population[father])
		case RandCross:
			p.Population[i].RandomCross(&p.Population[father], &p.Population[mother])
		case ArithmeticCross:
			p.Population[i].AritmeticCross(&p.Population[father], &p.Population[mother])
		default:
			p.Population[i].DivPointCross(&p.Population[father], &p.Population[mother])
		}

		switch p.MutationType {
		case MultiMutation:
			p.Population[i].MultiMutate(p.MutRate, p.MutSize)
		default:
			p.Population[i].OneMutate(p.MutRate, p.MutSize)
		}

		wg.Done()
	}
}

func (p *SNG) NextGenerationConcurrently(bufSize int) {
	p.Generation++
	p.Sort()

	ctl := make(chan int, bufSize)
	var wg sync.WaitGroup
	for i := 0; i < bufSize; i++ {
		go p.nextGenerationIndividual(ctl, &wg)
	}

	for i := p.Survivors; i < len(p.Population); i++ {
		wg.Add(1)
		ctl <- i
	}

	wg.Wait()
	close(ctl)
}

func (p *SNG) NextGeneration() {
	p.Generation++
	p.Sort()

	var father, mother int
	for i := p.Survivors; i < len(p.Population); i++ {
		// Select father and mother from survivors
		father, mother = p.SelectFatherMother()

		// Create the new individual
		switch p.CrossType {
		case MonoParent:
			p.Population[i].MonoParentCross(&p.Population[father])
		case RandCross:
			p.Population[i].RandomCross(&p.Population[father], &p.Population[mother])
		default:
			p.Population[i].DivPointCross(&p.Population[father], &p.Population[mother])
		}

		switch p.MutationType {
		case MultiMutation:
			p.Population[i].MultiMutate(p.MutRate, p.MutSize)
		default:
			p.Population[i].OneMutate(p.MutRate, p.MutSize)
		}
	}
}

// Sort
func (p *SNG) Sort() {
	/*sort.SliceStable(p.Population, func(i, j int) bool {
		return p.Population[i].Fitness >= p.Population[j].Fitness
	})*/
	p.last_best_index = p.getMaxFitnessIndex(0)
	for i := 0; i < p.Survivors; i++ {
		max := p.getMaxFitnessIndex(i + 1)
		if p.Population[i].Fitness < p.Population[max].Fitness {
			p.Population[i], p.Population[max] = p.Population[max], p.Population[i]
		}
	}
}

func (p *SNG) getMaxFitnessIndex(from int) int {
	var (
		index = from
		val   = p.Population[index].Fitness
	)
	for i := from + 1; i < len(p.Population); i++ {
		if val < p.Population[i].Fitness {
			val = p.Population[i].Fitness
			index = i
		}
	}
	return index
}

func (p *SNG) GetLastBestIndex() int {
	return p.last_best_index
}

// Selection
func (p *SNG) SelectFatherMother() (father, mother int) {
	father = rand.Intn(p.Survivors)
	mother = rand.Intn(p.Survivors)
	if father > mother {
		father, mother = mother, father
	}
	return
}

// Population
func (p *SNG) SetPopulationSize(size int, randomize bool) error {
	if size < p.Survivors {
		return errors.New("invalid new population size")
	}
	p.Sort()
	if len(p.Population) < size {
		genome_size := GetGenomeSize(p.Layers)
		for i := len(p.Population); i < size; i++ {
			father, mother := p.SelectFatherMother()
			individual := NewIndividual(genome_size)
			if randomize {
				individual.Randomize()
			} else {
				switch p.CrossType {
				case MonoParent:
					individual.MonoParentCross(&p.Population[father])
				case RandCross:
					individual.RandomCross(&p.Population[father], &p.Population[mother])
				default:
					individual.DivPointCross(&p.Population[father], &p.Population[mother])
				}
			}
			p.Population = append(p.Population, individual)
		}
	} else if len(p.Population) > size {
		p.Population = p.Population[:size]
	}
	return nil
}

func (p *SNG) Randomize(from, to int) {
	for i := from; i < to; i++ {
		p.Population[i].Randomize()
	}
}

// Fitness
func (p *SNG) ResetFitness() {
	for i := 0; i < len(p.Population); i++ {
		p.Population[i].Fitness = 0
	}
}

func (p *SNG) GetFitness(ind int) float64 {
	return p.Population[ind].Fitness
}

func (p *SNG) SetFitness(ind int, fitness float64) {
	p.Population[ind].Fitness = fitness
}

func (p *SNG) AddFitness(ind int, fitness float64) {
	p.Population[ind].Fitness += fitness
}

// Serialization
func (p *SNG) SaveAsBin(path string) error {
	var buff bytes.Buffer
	enc := gob.NewEncoder(&buff)
	err := enc.Encode(*p)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(path, buff.Bytes(), 0777)
	if err != nil {
		return err
	}

	return nil
}
