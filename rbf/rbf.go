package rbf

import (
	"fmt"
	"log"
	"math"
	"neural-network-rbf/matrix"

	"gonum.org/v1/gonum/mat"
)

type RBFNetwork struct {
	// Центры активационных функций
	Centers mat.Matrix
	// Веса выходного слоя (строки соответствуют центрам, а колонки - выходным
	// нейронам)
	Weights mat.Matrix
	// Ширина активационных окон (для каждого центра)
	Widths []float64
}

// Создаёт RBF сеть с
// центрами, равными входным значениям шаблонов (c[i]=X[i])
// (количество центров также равно количеству шаблонов);
// случайными весами в интервале [0.0; 1.0);
// с шириной окон, равной 1
func NewRBFNetwork(templatesInputs mat.Matrix) *RBFNetwork {
	templatesCount, _ := templatesInputs.Dims()

	// Центры равны входным шаблонам (c[i] = X[i])
	centers := templatesInputs
	centersAmount := templatesCount

	// Начальные веса генерируются случайно в интервале [0.0;1.0)
	weights := matrix.GetRandomizedMatrix(centersAmount, 1)

	// Ширина каждого окна равна 1
	widths := make([]float64, centersAmount)
	for i := range centersAmount {
		widths[i] = 1
	}

	return &RBFNetwork{centers, weights, widths}
}

// Метод получает результат на основе входных данных, а затем обновляет
// веса сети так, чтобы результат совпадал с желаемым.
// Возвращает вектор результатов для каждого входного шаблона
func (net *RBFNetwork) Train(templates mat.Matrix, desired mat.Matrix) mat.Matrix {
	numCenters, _ := net.Centers.Dims()

	// Считаем матрицу активации F
	activations := matrix.CreateMatrix(numCenters, numCenters, nil)
	for i := range numCenters {
		// Набор входных значений X меняется в строках, но не в колонках
		input := matrix.Row(templates, i)
		for j := range numCenters {
			// Центр меняется в колонках, но не в строках
			center := matrix.Row(net.Centers, j)

			activations.Set(i, j, gaussian(input, center, net.Widths[i]))
		}
	}

	// Сохраняем результат: Y=F*W
	result := matrix.Dot(activations, net.Weights)

	// Обновляем веса: W=F^(-1)*Y
	inversedActivations := matrix.Inverse(activations)
	net.Weights = matrix.Dot(inversedActivations, desired)

	return result
}

// Функция берёт последние inputDataSize значений из тренировочных данных как начальные.
// Затем предсказывает 1 следующее число, добавляет его в inputData и убирает оттуда
// первое значение. Таким образом количество входных данных всегда равно
// одному и тому же числу, а предсказанные значения используются для
// прогнозирования следующих
func (net *RBFNetwork) PredictRecursively(input []float64, valuesToPredict int) []float64 {
	fmt.Printf("Testing input length: %v\n", len(input))
	fmt.Printf("Values to predict: %v\n", valuesToPredict)

	result := []float64{}
	for i := 0; i < valuesToPredict; i++ {
		predicted := net.Predict(input)
		result = append(result, predicted)

		// Смещаем входные данные вправо на 1
		input = append(input, predicted)
		input = input[1:]
	}
	return result
}

// Возвращает 1 предсказанное значение на основе входных данных.
// Количество входных значений должно совпадать с размерностью центров сети для
// вычисления функции Гаусса
func (net *RBFNetwork) Predict(input []float64) float64 {
	numCenters, _ := net.Centers.Dims()

	output := 0.0
	for i := range numCenters {
		center := matrix.Row(net.Centers, i)
		output += gaussian(input, center, net.Widths[i]) * net.Weights.At(i, 0)
	}
	return output
}

// Выводит настройки сети
func (net *RBFNetwork) Log() {
	centersCount, centersDimension := net.Centers.Dims()
	weightsCount, _ := net.Weights.Dims()

	fmt.Printf("\n--Settings--\n")
	fmt.Printf("Number of hidden neurons: %v\n", centersCount)
	fmt.Printf("Centers dimension (equals number of inputs): %v\n", centersDimension)
	fmt.Printf("Number of weights: %v\n", weightsCount)
	fmt.Println("Activation window width: ", net.Widths[0])
}

// Активационная функция нейронов скрытого слоя. Принимает X(вектор входных значений)
// и C(конкретный центр), а также ширину активационного окна для него.
// Возвращает значение функции активации
func gaussian(input []float64, center []float64, width float64) float64 {
	if len(input) != len(center) {
		log.Fatal("Input and center dimensions don't match")
	}

	distance := 0.0
	for i := range input {
		distance += math.Pow(input[i]-center[i], 2)
	}

	power := -distance / math.Pow(width, 2)
	result := math.Exp(power)
	return result
}
