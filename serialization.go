package staticneurogenetic

import (
	"encoding/gob"
	"os"
)

func LoadFromBin(path string) (*SNG, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	decoder := gob.NewDecoder(file)
	if err != nil {
		return nil, err
	}

	var sng SNG
	err = decoder.Decode(&sng)
	if err != nil {
		return nil, err
	}

	return &sng, nil
}
