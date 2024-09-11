package helper

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
)

// Returns 1 column of csv file
func ReadCSVData(filePath string) []float64 {
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, _ := reader.ReadAll()

	data := []float64{}
	for _, record := range records {
		value, _ := strconv.ParseFloat(record[0], 64)
		data = append(data, value)
	}

	return data
}

// WriteFloat64ColumnToCSV writes a slice of float64 to a CSV file as a column.
func WriteFloat64ColumnToCSV(filePath string, columnData []float64) error {
	// Create or open the CSV file
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	// Create a new CSV writer
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Loop through the float64 slice and write each value as a new row in the CSV
	for _, value := range columnData {
		// Convert float64 to string
		record := []string{fmt.Sprintf("%f", value)}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}
