package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
)

var FibonacciSequence = [64]float64{0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141, 267914296, 433494437, 701408733, 1134903170, 1836311903, 2971215073, 4807526976, 7778742049, 12586269025, 20365011074, 32951280099, 53316291173, 86267571272, 139583862445, 225851433717, 365435296162, 591286729879, 956722026041, 1548008755920, 2504730781961, 4052739537881, 6557470319842}

type SequenceType float64

const (
	SequenceTypeFibonacci SequenceType = 1
	SequenceTypeOther     SequenceType = -1
)

type Sequence struct {
	Values []float64
	Type   SequenceType
}

func GenerateFibonacciClassification(rng *rand.Rand, sequenceSize uint8, sequencesCount uint64) ([]Sequence, error) {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	if sequenceSize >= uint8(len(FibonacciSequence)) {
		return nil, fmt.Errorf("sequence size cannot exceed %d", len(FibonacciSequence))
	}

	sequences := make([]Sequence, sequencesCount)
	maxStartIndex := len(FibonacciSequence) - int(sequenceSize)

	for i := uint64(0); i < sequencesCount/2; i += 1 {
		startIndex := uint8(rng.Intn(maxStartIndex))
		sequence := FibonacciSequence[startIndex : startIndex+sequenceSize]
		for i, value := range sequence {
			sequence[i] = math.Log1p(value)
		}
		sequences[i] = Sequence{sequence, SequenceTypeFibonacci}
	}

	for i := sequencesCount / 2; i < sequencesCount; i += 1 {
		sequence := make([]float64, sequenceSize)
		current := float64(rng.Intn(100))
		previous := float64(0)
		for j := range sequenceSize {
			sequence[j] = math.Log1p(current)
			previous = current
			current = previous + float64(rng.Intn(100))
		}
		sequences[i] = Sequence{sequence, SequenceTypeOther}
	}

	rng.Shuffle(len(sequences), func(i, j int) {
		sequences[i], sequences[j] = sequences[j], sequences[i]
	})

	return sequences, nil
}

type BlackJackDecision uint8

const (
	BlackJackDecisionHit   BlackJackDecision = 1
	BlackJackDecisionStand BlackJackDecision = 0
)

type BlackJackPair struct {
	DealerHand uint8
	PlayerHand uint8
	Decision   BlackJackDecision
}

var BlackJackOptimalStrategy = [220]BlackJackDecision{}

func GenerateBlackJackClassification(rng *rand.Rand, pairsCount uint64) {
	panic("GenerateBlackJackClassification not implemented")
}

func PredictPerceptron(input []float64, weights []float64, bias float64) (SequenceType, error) {
	result := float64(0)

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

func GenerateWeights(rng *rand.Rand, weightsCount uint64) []float64 {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	weights := make([]float64, weightsCount)
	for i := range weightsCount {
		weights[i] = rng.Float64()*2 - 1
	}

	return weights
}

func TrainPerceptronFibonacciClassification(weights []float64, bias float64, data []Sequence, iterations uint64, learningRate float64) ([]float64, float64, error) {
	for range iterations {
		for _, sequence := range data {
			predicted, err := PredictPerceptron(sequence.Values, weights, bias)
			if err != nil {
				return nil, 0, err
			}

			target := -1.0
			if sequence.Type == SequenceTypeFibonacci {
				target = 1.0
			}

			predictedValue := -1.0
			if predicted == SequenceTypeFibonacci {
				predictedValue = 1.0
			}

			error := target - predictedValue
			for i := range len(weights) {
				weights[i] = weights[i] + learningRate*error*sequence.Values[i]
			}

			bias = bias + learningRate*error
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

func EvaluatePerceptronFibonacciClassification(data []Sequence, weights []float64, bias float64) (Metrics, error) {
	var true []SequenceType
	var predicted []SequenceType
	for i := range len(data) {
		predictedType, err := PredictPerceptron(data[i].Values, weights, bias)
		if err != nil {
			return Metrics{}, err
		}
		predicted = append(predicted, predictedType)
		true = append(true, data[i].Type)
	}

	matrix, err := NewConfusionMatrix(true, predicted)
	if err != nil {
		return Metrics{}, err
	}

	return CalculateMetricsFromConfusionMatrix(matrix), nil
}

func main() {
	cpuprofile := flag.String("cpuprofile", "", "write cpu profile to file")
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			panic(err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			panic(err)
		}
		defer pprof.StopCPUProfile()
	}

	dataDim := uint8(12)
	trainDataCount := uint64(10)
	iterations := uint64(10)
	learningRate := 0.025

	rng := rand.New(rand.NewSource(42))

	trainData, err := GenerateFibonacciClassification(rng, dataDim, trainDataCount)
	if err != nil {
		fmt.Printf("Error occured during generation: %v\n", err)
		os.Exit(0)
	}

	weights := GenerateWeights(rng, uint64(dataDim+1))

	outWeights, outBias, err := TrainPerceptronFibonacciClassification(weights[:dataDim], weights[dataDim], trainData, iterations, learningRate)
	if err != nil {
		fmt.Printf("Error occured during training: %v\n", err)
		os.Exit(0)
	}

	evaluationDataCount := uint64(200)
	evaluationData, err := GenerateFibonacciClassification(rng, dataDim, evaluationDataCount)
	if err != nil {
		fmt.Printf("Error occured during generation: %v\n", err)
		os.Exit(0)
	}

	metrics, err := EvaluatePerceptronFibonacciClassification(evaluationData, outWeights, outBias)
	if err != nil {
		fmt.Printf("Error occured during evaluation: %v\n", err)
		os.Exit(0)
	}

	PrintMetrics(metrics)
}
