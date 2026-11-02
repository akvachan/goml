package main

import (
	"fmt"
	"math/rand"
)

func Fibonacci(startIndex uint64, count uint64) []float64 {
	var results []float64

	var next uint64
	var current uint64 = 1
	var previous uint64 = 0

	for i := range startIndex + count {
		if i >= startIndex {
			results = append(results, float64(previous))
		}

		next = current + previous
		previous = current
		current = next
	}

	return results
}

func GenerateFibonacciSequences(rng *rand.Rand, count uint64, dim uint64) [][]float64 {
	results := make([][]float64, count)

	for i := range count {
		results[i] = Fibonacci(uint64(rng.Intn(4)), dim)
	}

	return results
}

type SequenceType uint64

const (
	SequenceTypeFibonacci SequenceType = 1
	SequenceTypeOther     SequenceType = 0
)

type Sequence struct {
	Values []float64
	Type   SequenceType
}

func GenerateFibonacciClassification(rng *rand.Rand, count uint64, dim uint64) []Sequence {
	sequences := make([]Sequence, count)

	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	for i, sequence := range GenerateFibonacciSequences(rng, count/2, dim) {
		sequences[i] = Sequence{sequence, SequenceTypeFibonacci}
	}

	for i := count / 2; i < count; i += 1 {
		var sequence []float64
		current := float64(rng.Intn(100))
		for range dim {
			sequence = append(sequence, float64(current))
			current = sequence[len(sequence)-1] + float64(rng.Intn(100))
		}
		sequences[i] = Sequence{sequence, SequenceTypeOther}
	}

	rng.Shuffle(len(sequences), func(i, j int) {
		sequences[i], sequences[j] = sequences[j], sequences[i]
	})

	return sequences
}

func Predict(input []float64, weights []float64, bias float64) (SequenceType, error) {
	var result float64

	if len(input) != len(weights) {
		return SequenceType(3), fmt.Errorf("different dimensions input=%v and weights=%v", len(input), len(weights))
	}

	for i := range len(input) {
		result += input[i] * weights[i]
	}
	result += bias

	if result < 0 {
		return SequenceTypeOther, nil
	}

	return SequenceTypeFibonacci, nil
}

func TrainFibonacciClassification(rng *rand.Rand, data []Sequence, iterations uint64, learningRate float64) ([]float64, float64, error) {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	weights := make([]float64, len(data[0].Values))
	bias := rng.Float64()

	for i := range len(data[0].Values) {
		weights[i] = rng.Float64()
	}

	for range iterations {
		for _, sequence := range data {
			predicted, err := Predict(sequence.Values, weights, bias)
			if err != nil {
				return nil, 0, err
			}

			for i := range len(weights) {
				error := float64(sequence.Type) - float64(predicted)
				input := sequence.Values[i]
				weights[i] = weights[i] + learningRate*error*input
				bias = bias + learningRate*error
			}
		}
	}

	return weights, bias, nil
}

type Metrics struct {
	Precision float64
	Recall    float64
	F1        float64
}

func PrintMetrics(metrics Metrics) {
	fmt.Printf("Precision: %f\nRecall: %f\nF1: %f\n", metrics.Precision, metrics.Recall, metrics.F1)
}

type ConfusionMatrix struct {
	TruePositives  uint64
	FalsePositives uint64
	TrueNegatives  uint64
	FalseNegatives uint64
}

func NewConfusionMatrix(true []SequenceType, predicted []SequenceType) (ConfusionMatrix, error) {
	var matrix ConfusionMatrix

	if len(true) != len(predicted) {
		return matrix, fmt.Errorf("different dimensions true=%v and predicted=%v", len(true), len(predicted))
	}

	for i := range len(true) {
		if true[i] == predicted[i] {
			if true[i] == SequenceTypeFibonacci {
				matrix.TruePositives += 1
			} else {
				matrix.TrueNegatives += 1
			}
		} else {
			if true[i] != SequenceTypeFibonacci {
				matrix.FalsePositives += 1
			} else {
				matrix.FalseNegatives += 1
			}
		}
	}

	return matrix, nil
}

func CalculateMetricsFromConfusionMatrix(matrix ConfusionMatrix) Metrics {
	return Metrics{
		Precision(matrix.TruePositives, matrix.FalsePositives),
		Recall(matrix.TruePositives, matrix.FalseNegatives),
		F1(matrix.TruePositives, matrix.FalsePositives, matrix.FalseNegatives),
	}
}

func Precision(truePositives uint64, falsePositives uint64) float64 {
	bottom := truePositives + falsePositives
	if bottom == 0 {
		return 0
	}
	return float64(truePositives) / float64(bottom)
}

func Recall(truePositives uint64, falseNegatives uint64) float64 {
	bottom := truePositives + falseNegatives
	if bottom == 0 {
		return 0
	}
	return float64(truePositives) / float64(bottom)
}

func F1(truePositives uint64, falsePositives uint64, falseNegatives uint64) float64 {
	bottom := (2*truePositives + falsePositives + falseNegatives)
	if bottom == 0 {
		return 0
	}
	return float64(2*truePositives) / float64(bottom)
}

func EvaluateFibonacciClassification(data []Sequence, weights []float64, bias float64) (Metrics, error) {
	var true []SequenceType
	var predicted []SequenceType
	for i := range len(data) {
		predictedType, err := Predict(data[i].Values, weights, bias)
		if err != nil {
			return Metrics{}, err
		}
		predicted = append(predicted, predictedType)
		true = append(true, data[i].Type)
	}

	matrix, err := NewConfusionMatrix(true, predicted)
	if err != nil {
		return Metrics{}, nil
	}

	return CalculateMetricsFromConfusionMatrix(matrix), nil
}

func main() {
	var dataDim uint64 = 33

	var trainDataCount uint64 = 20
	rng := rand.New(rand.NewSource(250))

	trainData := GenerateFibonacciClassification(rng, trainDataCount, dataDim)

	weights, bias, err := TrainFibonacciClassification(rng, trainData, 50, 0.05)
	if err != nil {
		fmt.Printf("Error occured during training: %v", err)
	}

	var evaluationDataCount uint64 = 20
	evaluationData := GenerateFibonacciClassification(rng, evaluationDataCount, dataDim)

	metrics, err := EvaluateFibonacciClassification(evaluationData, weights, bias)
	if err != nil {
		fmt.Printf("Error occured during evaluation: %v", err)
	}

	PrintMetrics(metrics)
}
