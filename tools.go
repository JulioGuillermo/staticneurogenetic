package staticneurogenetic

func GetGenomeSize(layers []int) int {
	genome_size := 0
	for i := 1; i < len(layers); i++ {
		// one genome values for each input and bias for each layer
		// +1 for bias
		genome_size += layers[i] * (layers[i-1] + 1)
	}
	return genome_size
}
