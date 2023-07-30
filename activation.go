package staticneurogenetic

import "math"

type ActivationFunction func(float64) float64

type Activation int

const (
	Linear = Activation(iota)
	Relu
	LeakyRelu
	Sigmoid
	Tanh
	Sin
	Sign
)

func GetActivation(activation Activation) ActivationFunction {
	switch activation {
	case Relu:
		return ReluFunc
	case LeakyRelu:
		return LeakyReluFunc
	case Sigmoid:
		return SigmoidFunc
	case Tanh:
		return TanhFunc
	case Sin:
		return SinFunc
	case Sign:
		return SignFunc
	default:
		return LinearFunc
	}
}

func LinearFunc(x float64) float64 {
	return x
}

func ReluFunc(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func LeakyReluFunc(x float64) float64 {
	if x < 0 {
		return x / 10
	}
	return x
}

func SigmoidFunc(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func TanhFunc(x float64) float64 {
	return math.Tanh(x)
}

func SinFunc(x float64) float64 {
	return math.Sin(x)
}

func SignFunc(x float64) float64 {
	if x > 0 {
		return 1
	}
	if x < 0 {
		return -1
	}
	return 0
}
