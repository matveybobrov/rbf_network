package main

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"neural-network-rbf/matrix"

	"gonum.org/v1/gonum/mat"
)

type RBFNetwork struct {
	// Центры активационных функций
	centers mat.Matrix
	// Веса выходного слоя
	weights mat.Matrix
	// Ширина активационных окон
	widths []float64
}

func NewRBFNetwork(templates mat.Matrix) *RBFNetwork {
	// Центра равны входным шаблонам (c[i] = X[i])
	templatesCount, _ := templates.Dims()
	centers := templates

	// Начальные веса генерируются случайно в интервале [0.0;1.0)
	weightsData := make([]float64, templatesCount)
	for i := range templatesCount {
		weightsData[i] = rand.Float64()
	}
	// rows: templatesCount(centersCount); cols: 1
	weights := mat.NewDense(templatesCount, 1, weightsData)

	// Ширина активационных окон генерируются случайно в интервале [0.0;1.0)
	widths := make([]float64, templatesCount)
	for i := range templatesCount {
		widths[i] = rand.Float64()
	}

	return &RBFNetwork{centers, weights, widths}
}

// Активационная функция
func Gaussian(input []float64, center []float64, width float64) float64 {
	sum := 0.0
	if len(input) != len(center) {
		log.Fatal("Input and center dimensions (N) don't match")
	}

	for i := range len(input) {
		sum += math.Pow(input[i]-center[i], 2)
	}
	return math.Exp(-sum / math.Pow(width, 2))
}

func (net *RBFNetwork) Train(templates mat.Matrix, desired mat.Matrix) mat.Matrix {
	//
	//	Forward pass
	//
	numCenters, _ := net.centers.Dims()
	_, numInputs := templates.Dims()

	// Calculate activation matrix (F)
	activations := mat.NewDense(numCenters, numCenters, nil)
	for i := range numCenters {
		// same input for every column
		input := make([]float64, numInputs)
		mat.Row(input, i, templates)
		for j := range numCenters {
			// different center for every column
			center := make([]float64, numInputs)
			mat.Row(center, j, net.centers)

			activations.Set(i, j, Gaussian(input, center, net.widths[i]))
		}
	}

	result := matrix.Dot(activations, net.weights)
	//
	// Training
	//
	inversedActivations := mat.NewDense(numCenters, numCenters, nil)
	inversedActivations.Inverse(activations)

	net.weights = matrix.Dot(inversedActivations, desired)

	// Returns template prediction result matrix
	return result
}

func calculateAverageTemplateAccuracy(output []float64, desired []float64) float64 {
	fmt.Println("result")
	fmt.Println(output)
	fmt.Println("desired result")
	fmt.Println(desired)

	if len(output) != len(desired) {
		return 0.0 // Return 0 if lengths do not match
	}

	totalAccuracy := 0.0

	// Calculate the accuracy for each prediction
	for i := range output {
		// Calculate accuracy as the absolute difference, normalized to the desired value
		accuracy := 1 - (math.Abs(output[i]-desired[i]) / desired[i])
		if accuracy < 0 {
			accuracy = 0 // Ensure accuracy is not negative
		}
		totalAccuracy += accuracy
	}

	// Calculate average accuracy
	averageAccuracy := (totalAccuracy / float64(len(desired))) * 100 // Return as percentage
	return averageAccuracy
}

func main() {
	// []float64 slice of time series values
	//trainingData := helper.ReadCSVData("data/training.csv")
	//testingData := helper.ReadCSVData("data/testing.csv")

	templatesCount := 2
	inputsPerTemplate := 2

	templatesData := make([]float64, templatesCount*inputsPerTemplate)
	for i := range len(templatesData) {
		templatesData[i] = rand.Float64()
	}
	templates := mat.NewDense(templatesCount, inputsPerTemplate, templatesData)
	desiredData := make([]float64, templatesCount)
	for i := range len(desiredData) {
		desiredData[i] = rand.Float64()
	}
	desired := mat.NewDense(templatesCount, 1, desiredData)
	net := NewRBFNetwork(templates)
	/*
		fmt.Println("templates")
		matrix.PrintMatrix(templates)
		fmt.Println("desired")
		matrix.PrintMatrix(desired)
		fmt.Println("Network structure")
		fmt.Println("centers")
		matrix.PrintMatrix(net.centers)
		fmt.Println("weights")
		matrix.PrintMatrix(net.weights)
		fmt.Println("widths")
		fmt.Println(net.widths)
	*/

	// training
	// IT FUCKING WORKS (perfectly for random data)
	fmt.Println("\nTraining")
	for i := 0; i < 2; i++ {
		fmt.Printf("\niteration %v\n", i)
		result := net.Train(templates, desired)

		// Calculate accuracy
		resultVector := make([]float64, templatesCount)
		mat.Col(resultVector, 0, result)
		accuracy := calculateAverageTemplateAccuracy(resultVector, desiredData)
		fmt.Println("accuracy")
		fmt.Printf("%.2f%%\n", accuracy)
	}
}
