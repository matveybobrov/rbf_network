package main

import (
	"fmt"
	"log"
	"neural-network-rbf/helper"
	"neural-network-rbf/rbf"
)

func main() {
	const WINDOW_SIZE = 50
	const STEP_SIZE = 9
	const VALUES_TO_PREDICT = 75
	// НЕ ИЗМЕНЯТЬ: сейчас сеть может работать только с одним выходным значением.
	// Для предсказания нескольких значений - использовать net.PredictRecursively()
	const OUTPUT_SIZE = 1
	const INPUT_SIZE = WINDOW_SIZE - OUTPUT_SIZE

	// Готовим тренировочные шаблоны (матрица входных значений и вектор желаемых)
	trainingData := helper.ReadCSVData("data/DailyDelhiClimateTrainNormalized.csv")
	inputs, desired := helper.PrepareData(trainingData, WINDOW_SIZE, STEP_SIZE, OUTPUT_SIZE)

	// Создаём сеть и тренируем её, обновляя веса
	net := rbf.NewRBFNetwork(inputs)
	net.Log()
	net.Train(inputs, desired)

	// Рекурсивно предсказываем <VALUES_TO_PREDICT> значений
	firstInput := trainingData[len(trainingData)-INPUT_SIZE:]
	predictedValues := net.PredictRecursively(firstInput, VALUES_TO_PREDICT)

	// Выводим точность работы сети
	fmt.Println("\n---Result---")
	testingData := helper.ReadCSVData("data/DailyDelhiClimateTestNormalized.csv")
	desiredData := testingData[:VALUES_TO_PREDICT]
	fmt.Printf("Desired values: \n%v", desiredData)
	fmt.Printf("\nPredicted values: \n%.6f", predictedValues)
	accuracy := helper.CalculateAverageAccuracy(predictedValues, desiredData)
	fmt.Println("\nAverage accuracy: ")
	fmt.Printf("%.5f%%\n", accuracy)

	// Денормализуем полученные значения
	allData := helper.ReadCSVData("data/DailyDelhiClimateTotal.csv")
	predictedValues = helper.UnnormalizeData(predictedValues, allData)

	// Сохраняем результат
	helper.SaveResult("result/result.csv", "data/DailyDelhiClimateTest.csv", predictedValues)

	// Рисуем график
	if err := helper.PlotCSVFiles("data/DailyDelhiClimateTest.csv", "result/result.csv", "result/plot.png"); err != nil {
		log.Fatalf("Error plotting CSV files: %v", err)
	}
}
