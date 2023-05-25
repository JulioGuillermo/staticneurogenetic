package staticneurogenetic

// Output
func (p *SNG) Output(ind int, input []float64) []float64 {
	return p.Population[ind].output(input, p.Layers, getActivation(p.Activation))
}

func (p *SNG) MaxOutput(ind int, input []float64) (value float64, index int) {
	output := p.Output(ind, input)
	index = 0
	value = output[0]
	for i, v := range output {
		if value < v {
			index = i
			value = v
		}
	}
	return
}

func (p *SNG) MinOutput(ind int, input []float64) (value float64, index int) {
	output := p.Output(ind, input)
	index = 0
	value = output[0]
	for i, v := range output {
		if value > v {
			index = i
			value = v
		}
	}
	return
}
