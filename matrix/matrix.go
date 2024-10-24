// Данный пакет содержит методы работы с матрицами
package matrix

import (
	"fmt"
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

// Создаёт матрицу с указанным числом строк и колонок, заполняя её данными.
// Длина вектора данных должна быть равна rows * cols
func CreateMatrix(rows int, cols int, data []float64) *mat.Dense {
	matrix := mat.NewDense(rows, cols, data)
	return matrix
}

// Возвращает указанную строку матрицы
func Row(m mat.Matrix, id int) []float64 {
	_, rowLength := m.Dims()
	result := make([]float64, rowLength)
	mat.Row(result, id, m)
	return result
}

func Inverse(m mat.Matrix) mat.Matrix {
	rows, cols := m.Dims()
	result := mat.NewDense(rows, cols, nil)
	result.Inverse(m)
	return result
}

func ReplaceRow(m mat.Matrix, rowIndex int, newRow []float64) {
	// Convert mat.Matrix to mat.Dense to modify it
	denseMatrix, ok := m.(*mat.Dense)
	if !ok {
		fmt.Println("Provided matrix is not of type *mat.Dense")
		return
	}

	rows, cols := denseMatrix.Dims()
	if rowIndex < 0 || rowIndex >= rows {
		fmt.Println("Row index out of bounds")
		return
	}
	if len(newRow) != cols {
		fmt.Println("New row length must match the number of columns in the matrix")
		return
	}

	for j := 0; j < cols; j++ {
		denseMatrix.Set(rowIndex, j, newRow[j])
	}
}

func GetRandomizedMatrix(row, col int) mat.Matrix {
	data := make([]float64, row*col)
	for i := range data {
		data[i] = rand.NormFloat64()
	}

	result := mat.NewDense(row, col, data)
	return result
}

func GetRandomizedVector(row int) []float64 {
	result := make([]float64, row)
	for i := 0; i < row; i++ {
		result[i] = rand.NormFloat64()
	}
	return result
}

func Print(matrix mat.Matrix) {
	rows, cols := matrix.Dims()
	for row := range rows {
		fmt.Printf("%v.\t", row)
		for col := range cols {
			fmt.Printf("%v\t", matrix.At(row, col))
		}
		fmt.Println()
	}
}

func Dot(m, n mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	_, columns := n.Dims()
	o := mat.NewDense(rows, columns, nil)
	o.Product(m, n)
	return o
}
