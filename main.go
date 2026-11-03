package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
)

var fibonacciSequence = [64]float64{0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141, 267914296, 433494437, 701408733, 1134903170, 1836311903, 2971215073, 4807526976, 7778742049, 12586269025, 20365011074, 32951280099, 53316291173, 86267571272, 139583862445, 225851433717, 365435296162, 591286729879, 956722026041, 1548008755920, 2504730781961, 4052739537881, 6557470319842}

type GenerateFibonacciClassificationConfig struct {
	Rng            *rand.Rand
	NormalizeFunc  func(x []float64) error
	SequencesCount uint64
	SequenceSize   uint64
	Normalize      bool
	Shuffle        bool
}

func GenerateFibonacciClassification(c GenerateFibonacciClassificationConfig) (x []float64, y []float64, err error) {
	rng := c.Rng
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	if c.SequenceSize == 0 {
		return nil, nil, fmt.Errorf("sequence size cannot be zero")
	}

	if c.SequencesCount == 0 {
		return nil, nil, fmt.Errorf("sequences count cannot be zero")
	}

	if c.SequenceSize >= uint64(len(fibonacciSequence)) {
		return nil, nil, fmt.Errorf("sequence size cannot exceed %d", len(fibonacciSequence))
	}

	y = make([]float64, c.SequencesCount)
	x = make([]float64, c.SequencesCount*c.SequenceSize)
	maxStartIndex := len(fibonacciSequence) - int(c.SequenceSize)

	for i := uint64(0); i < c.SequencesCount/2; i += 1 {
		startIndex := uint64(rng.Intn(maxStartIndex + 1))
		offset := i * c.SequenceSize
		copy(x[offset:offset+c.SequenceSize], fibonacciSequence[startIndex:startIndex+uint64(c.SequenceSize)])
		y[i] = 1
	}

	for i := c.SequencesCount / 2; i < c.SequencesCount; i++ {
		current := float64(rng.Intn(100))
		offset := i * c.SequenceSize
		for j := uint64(0); j < c.SequenceSize; j++ {
			random := float64(rng.Intn(200))
			if j == 0 {
				x[offset+j] = current + random
			} else {
				x[offset+j] = x[offset+j-1] + random
			}
		}
		y[i] = -1
	}

	if c.Normalize {
		normalizeFunc := c.NormalizeFunc
		if normalizeFunc == nil {
			normalizeFunc = NormalizeViaLog1p
		}
		if err := normalizeFunc(x); err != nil {
			return nil, nil, fmt.Errorf("failed to normalize sequence: %w", err)
		}
	}

	if c.Shuffle {
		rng.Shuffle(int(c.SequencesCount), func(i, j int) {
			for k := uint64(0); k < c.SequenceSize; k++ {
				x[uint64(i)*c.SequenceSize+k], x[uint64(j)*c.SequenceSize+k] = x[uint64(j)*c.SequenceSize+k], x[uint64(i)*c.SequenceSize+k]
			}

			y[i], y[j] = y[j], y[i]
		})
	}

	return x, y, nil
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

func GenerateBlackJackClassification(rng *rand.Rand, pairsCount uint64) {
	panic("GenerateBlackJackClassification not implemented")
}

func NormalizeViaLog1p(x []float64) error {
	for i := range len(x) {
		if x[i] <= -1 || math.IsNaN(x[i]) || math.IsInf(x[i], 0) {
			return fmt.Errorf("invalid value at index %d: %v", i, x[i])
		}

		x[i] = math.Log1p(x[i])
	}
	return nil
}

func NormalizeViaMinMax(x []float64) error {
	if len(x) < 1 {
		return fmt.Errorf("empty list")
	}

	minimum := x[0]
	maximum := x[0]
	for i := 1; i < len(x); i++ {
		minimum = min(minimum, x[i])
		maximum = max(maximum, x[i])
	}

	bottom := maximum - minimum
	if bottom == 0 {
		return fmt.Errorf("cannot use MinMax normalization when maximum (%v) == minimum (%v)", maximum, minimum)
	}

	for i := range len(x) {
		x[i] = (x[i] - minimum) / bottom
	}

	return nil
}

func NormalizeViaMean(x []float64) error {
	if len(x) == 0 {
		return fmt.Errorf("empty list")
	}

	minimum := x[0]
	maximum := x[0]
	sum := x[0]

	for i := 1; i < len(x); i++ {
		minimum = min(minimum, x[i])
		maximum = max(maximum, x[i])
		sum += x[i]
	}

	mean := sum / float64(len(x))
	bottom := maximum - minimum

	if bottom == 0 {
		return fmt.Errorf(
			"cannot use mean normalization when maximum (%v) == minimum (%v)",
			maximum,
			minimum,
		)
	}

	for i := range x {
		x[i] = (x[i] - mean) / bottom
	}

	return nil
}

func NormalizeViaZScore(x []float64) error {
	if len(x) == 0 {
		return fmt.Errorf("empty list")
	}

	var sum float64
	for _, value := range x {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("invalid value: %v", value)
		}
		sum += value
	}

	mean := sum / float64(len(x))
	var sumOfSquares float64

	for _, value := range x {
		difference := value - mean
		sumOfSquares += difference * difference
	}

	variance := sumOfSquares / float64(len(x))
	if variance == 0 {
		return fmt.Errorf("standard deviation is zero")
	}

	deviation := math.Sqrt(variance)
	for i := range x {
		x[i] = (x[i] - mean) / deviation
	}

	return nil
}

func PredictPerceptron(x []float64, weights []float64) (float64, error) {
	if len(weights) != len(x)+1 {
		return 0, fmt.Errorf("different dimensions input=%v and weights=%v", len(x), len(weights))
	}

	result := weights[len(weights)-1]
	for i := range x {
		result += x[i] * weights[i]
	}

	if result < 0 {
		return -1, nil
	}

	return 1, nil
}

func GenerateWeightsRandomNormal(rng *rand.Rand, weightsCount uint64, standardDeviation float64, mean float64) ([]float64, error) {
	if math.IsNaN(standardDeviation) || math.IsInf(standardDeviation, 0) || standardDeviation < 0 {
		return nil, fmt.Errorf("invalid standard deviation")
	}

	if math.IsNaN(mean) || math.IsInf(mean, 0) {
		return nil, fmt.Errorf("invalid mean")
	}

	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	weights := make([]float64, int(weightsCount))

	for i := range weights {
		weights[i] = rng.NormFloat64()*standardDeviation + mean
	}

	return weights, nil
}

func GenerateWeightsRandomUniform(rng *rand.Rand, weightsCount uint64, minimum float64, maximum float64) ([]float64, error) {
	if math.IsNaN(minimum) || math.IsInf(minimum, 0) {
		return nil, fmt.Errorf("invalid minimum")
	}

	if math.IsNaN(maximum) || math.IsInf(maximum, 0) {
		return nil, fmt.Errorf("invalid maximum")
	}

	if maximum <= minimum {
		return nil, fmt.Errorf("minimum cannot exceed maximum")
	}

	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	weights := make([]float64, int(weightsCount))

	for i := range weights {
		weights[i] = minimum + rng.Float64()*(maximum-minimum)
	}

	return weights, nil
}

type PerceptronHyperparams struct {
	LearningRate float64
	Iterations   uint64
}

func TrainPerceptron(x []float64, featuresCount uint64, y []float64, weights []float64, hp PerceptronHyperparams) ([]float64, error) {
	if featuresCount == 0 {
		return nil, fmt.Errorf("cannot have zero features")
	}

	if len(x) == 0 || len(y) == 0 {
		return nil, fmt.Errorf("training data cannot be empty")
	}

	if uint64(len(x))%featuresCount != 0 {
		return nil, fmt.Errorf(
			"input dimensions are not divisible by feature count",
		)
	}

	samplesCount := uint64(len(x)) / featuresCount

	if samplesCount != uint64(len(y)) {
		return nil, fmt.Errorf(
			"number of samples does not correspond to outputs: %d != %d",
			samplesCount,
			len(y),
		)
	}

	if math.IsNaN(hp.LearningRate) ||
		math.IsInf(hp.LearningRate, 0) ||
		hp.LearningRate <= 0 {
		return nil, fmt.Errorf("invalid learning rate")
	}

	if weights == nil {
		var err error
		if weights, err = GenerateWeightsRandomUniform(nil, featuresCount+1, -1, 1); err != nil {
			return nil, fmt.Errorf("failed to generate weights: %w", err)
		}
	}

	if uint64(len(weights)) != featuresCount+1 {
		return nil, fmt.Errorf(
			"number of weights must equal features count + 1",
		)
	}

	for i, label := range y {
		if label != -1 && label != 1 {
			return nil, fmt.Errorf(
				"invalid label at index %d: %v",
				i,
				label,
			)
		}
	}

	for range hp.Iterations {
		for i := uint64(0); i < uint64(len(x)); i += featuresCount {
			predicted, err := PredictPerceptron(x[i:i+featuresCount], weights)
			if err != nil {
				return nil, fmt.Errorf("failed to compute predictions: %w", err)
			}

			error := float64(y[i/featuresCount] - predicted)
			for j := range uint64(len(weights) - 1) {
				weights[j] = weights[j] + hp.LearningRate*float64(error)*x[i+j]
			}

			weights[len(weights)-1] = weights[len(weights)-1] + hp.LearningRate*error
		}
	}

	return weights, nil
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

func NewConfusionMatrix(true []float64, predicted []float64, trueValue float64) (ConfusionMatrix, error) {
	var matrix ConfusionMatrix

	if len(true) != len(predicted) {
		return matrix, fmt.Errorf("different dimensions true=%v and predicted=%v", len(true), len(predicted))
	}

	for i := range len(true) {
		if true[i] == predicted[i] {
			if true[i] == trueValue {
				matrix.TruePositives += 1
			} else {
				matrix.TrueNegatives += 1
			}
		} else {
			if true[i] != trueValue {
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

func EvaluatePerceptron(x []float64, featuresCount uint64, y []float64, weights []float64) (Metrics, error) {
	if len(x) == 0 || len(y) == 0 {
		return Metrics{}, fmt.Errorf("evaluation data cannot be empty")
	}

	if featuresCount == 0 {
		return Metrics{}, fmt.Errorf("cannot have zero features")
	}

	if uint64(len(x))%featuresCount != 0 {
		return Metrics{}, fmt.Errorf("input dimensions are not divisible by feature count")
	}

	if len(weights) == 0 {
		return Metrics{}, fmt.Errorf("weights cannot be empty")
	}

	if uint64(len(weights)) != featuresCount+1 {
		return Metrics{}, fmt.Errorf("number of weights must equal features count + 1")
	}

	samplesCount := uint64(len(x)) / featuresCount

	if samplesCount != uint64(len(y)) {
		return Metrics{}, fmt.Errorf("number of samples does not correspond to outputs")
	}

	var predicted, true []float64
	for i := uint64(0); i < uint64(len(x)); i += featuresCount {
		predictedType, err := PredictPerceptron(x[i:i+featuresCount], weights)
		if err != nil {
			return Metrics{}, fmt.Errorf("failed to compute predictions: %w", err)
		}
		predicted = append(predicted, predictedType)
		true = append(true, y[i/featuresCount])
	}

	matrix, err := NewConfusionMatrix(true, predicted, 1)
	if err != nil {
		return Metrics{}, err
	}

	return CalculateMetricsFromConfusionMatrix(matrix), nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	cpuprofile := flag.String("cpuprofile", "", "write cpu profile to file")
	flag.Parse()
	if *cpuprofile != "" {
		file, err := os.Create(*cpuprofile)
		if err != nil {
			return fmt.Errorf("failed to profile: %v", err)
		}
		if err := pprof.StartCPUProfile(file); err != nil {
			return fmt.Errorf("failed to profile: %v", err)
		}
		defer pprof.StopCPUProfile()
		defer file.Close()
	}

	rng := rand.New(rand.NewSource(42))
	featuresCount := uint64(12)

	x, y, err := GenerateFibonacciClassification(GenerateFibonacciClassificationConfig{
		Rng:            rng,
		SequenceSize:   featuresCount,
		SequencesCount: 64,
		Normalize:      true,
		Shuffle:        true,
	})
	if err != nil {
		return fmt.Errorf("failed to generate training fibonacci dataset: %v", err)
	}

	weights, err := GenerateWeightsRandomUniform(rng, uint64(featuresCount+1), -1, 1)
	if err != nil {
		return fmt.Errorf("failed to generate weights: %v", err)
	}

	weights, err = TrainPerceptron(x, featuresCount, y, weights, PerceptronHyperparams{
		LearningRate: 0.05,
		Iterations:   10,
	})
	if err != nil {
		return fmt.Errorf("failed to train perceptron: %v", err)
	}

	evalDataCount := uint64(200)
	xEval, yEval, err := GenerateFibonacciClassification(GenerateFibonacciClassificationConfig{
		Rng:            rng,
		SequenceSize:   featuresCount,
		SequencesCount: evalDataCount,
		Normalize:      true,
		Shuffle:        true,
	})
	if err != nil {
		return fmt.Errorf("failed to generate evaluation fibonacci dataset: %v", err)
	}

	metrics, err := EvaluatePerceptron(xEval, featuresCount, yEval, weights)
	if err != nil {
		return fmt.Errorf("failed to evaluate perceptron: %v", err)
	}

	PrintMetrics(metrics)
	return nil
}
