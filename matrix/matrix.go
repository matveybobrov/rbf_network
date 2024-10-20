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

// Dot product of 2 matrices
func Dot(m, n mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	_, columns := n.Dims()
	o := mat.NewDense(rows, columns, nil)
	o.Product(m, n)
	return o
}

// Apply a function to the matrix
func Apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

// Scale the matrix (multiply by scalar)
func Scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

// Перемножение двух матриц
func Multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

// Складывание и вычитание матриц
func Add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}
func Subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

// Add a scalar to each value of the matrix
func AddScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return Add(m, n)
}
