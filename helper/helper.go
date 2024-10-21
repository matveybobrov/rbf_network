// Данный пакет содержит функции для работы с данными
package helper

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"neural-network-rbf/matrix"
	"os"
	"slices"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// Разбивает вектор тренировочных данных на матрицу inputs(X) и вектор desired(Y)
func PrepareData(data []float64, windowSize int, stepSize int, outputSize int) (mat.Matrix, mat.Matrix) {
	fmt.Println("---Preparing training data---")
	fmt.Printf("Training data length: %v\n", len(data))
	fmt.Printf("Window size: %v\n", windowSize)
	fmt.Printf("Step size: %v\n", stepSize)

	if len(data)%windowSize != 0 {
		log.Fatal("Training data length must be divisible withour remainder by window size")
	}

	// Разбиваем тренировочные данные на шаблоны, последнее значение шаблона
	// будет отражать желаемый результат
	templatesCount := (len(data) - windowSize + stepSize) / stepSize
	inputsPerTemplate := windowSize - outputSize

	fmt.Printf("Total templates: %v\n", templatesCount)
	fmt.Printf("Inputs per template: %v\n", inputsPerTemplate)
	fmt.Printf("Outputs per template: %v\n", outputSize)

	// Составляем матрицу входных значений(X) и вектор результатов(Y)
	inputsData := []float64{}
	desiredData := []float64{}
	for i := range templatesCount {
		start := i * stepSize
		end := start + windowSize
		if end > len(data) {
			break
		}

		window := data[start:end]
		input := window[:inputsPerTemplate]
		inputsData = append(inputsData, input...)
		desired := window[inputsPerTemplate:]
		desiredData = append(desiredData, desired...)
	}
	// Матрица входных данных X для каждого шаблона
	inputs := matrix.CreateMatrix(templatesCount, inputsPerTemplate, inputsData)
	// Вектор результатов Y для каджого шаблона
	desired := matrix.CreateMatrix(templatesCount, outputSize, desiredData)

	return inputs, desired
}

// Денормализует data на основе base(в котором находит min и max)
func UnnormalizeData(data []float64, base []float64) []float64 {
	max := slices.Max(base)
	min := slices.Min(base)

	for i := range data {
		data[i] = data[i]*(max-min) + min
	}

	return data
}
func NormalizeData(data []float64) []float64 {
	if len(data) == 0 {
		return data
	}

	min, max := data[0], data[0]
	for _, value := range data {
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
	}

	normalized := make([]float64, len(data))
	for i, value := range data {
		normalized[i] = (value - min) / (max - min)
	}

	return normalized
}

// Возвращает вторую колонку CSV файла без заголовка, приводя всё к float
func ReadCSVData(filePath string) []float64 {
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, _ := reader.ReadAll()

	data := []float64{}
	for i := 1; i < len(records); i++ {
		record := records[i]
		value, _ := strconv.ParseFloat(record[1], 64)
		data = append(data, value)
	}

	return data
}
func NormalizeCSVFile(normalizedFileDest string, initFile string) error {
	data := ReadCSVData(initFile)
	data = NormalizeData(data)
	err := SaveResult(normalizedFileDest, initFile, data)
	return err
}

// Принимает массив полученных результатов по нескольким шаблонам и
// массив желаемых результатов. Возвращает средний процент точности
func CalculateAverageAccuracy(output []float64, desired []float64) float64 {
	if len(output) != len(desired) {
		return 0.0
	}

	totalAccuracy := 0.0
	// Считаем точность для каджого предсказания
	for i := range output {
		accuracy := 1 - (math.Abs(output[i]-desired[i]) / desired[i])
		if accuracy < 0 {
			accuracy = 0
		}
		totalAccuracy += accuracy
	}

	// Считаем среднюю точность и переводим в проценты
	averageAccuracy := (totalAccuracy / float64(len(desired))) * 100
	return averageAccuracy
}

// Сохраняет результат работы сети в файл. Принимает расположение файла для
// сохранения, расположение файла с тестовыми данными для сопоставления
// значений других колонок и заголовков, а также сами данные для сохранения
func SaveResult(outputDest string, testFile string, columnData []float64) error {
	// Создаём или открываем CSV файл
	resultFile, err := os.Create(outputDest)
	if err != nil {
		return err
	}
	defer resultFile.Close()

	// Начинаем запись
	writer := csv.NewWriter(resultFile)
	defer writer.Flush()

	// Открываем файл с тестовыми данными
	file, err := os.Open(testFile)
	if err != nil {
		log.Fatal("Unable to read test file "+testFile, err)
	}
	defer file.Close()

	// Получаем данные из тестового файла
	reader := csv.NewReader(file)
	records, _ := reader.ReadAll()

	// Записываем заголовок из тестового файла в результирующий файл
	if err := writer.Write(records[0]); err != nil {
		return err
	}

	// Записываем данные из columnData во второй столбец, сохраняя 1 столбец
	// (дату) из testFile
	for i, value := range columnData {
		record := []string{records[i+1][0], fmt.Sprintf("%f", value)}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}
