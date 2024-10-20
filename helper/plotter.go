package helper

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func parseDate(dateStr string) (time.Time, error) {
	t, err := time.Parse("2006-01-02", dateStr)
	if err != nil {
		return time.Time{}, fmt.Errorf("failed to parse date: %v", err)
	}
	return t, nil
}

func readCSV(filePath string) (plotter.XYs, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV file: %v", err)
	}

	var xys plotter.XYs
	for i, record := range records {
		if i == 0 {
			continue // Skip header
		}
		if len(record) < 2 {
			continue // Skip rows without enough columns
		}

		x, err := parseDate(record[0])
		if err != nil {
			return nil, fmt.Errorf("failed to parse x value: %v", err)
		}

		y, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse y value: %v", err)
		}

		xys = append(xys, struct{ X, Y float64 }{float64(x.Unix()), y})
	}

	return xys, nil
}

// Визуализирует графики двух CSV файлов, длиной наименьшего из них
func PlotCSVFiles(file1, file2, outputFile string) error {
	// Read the first CSV file
	xys1, err := readCSV(file1)
	if err != nil {
		return fmt.Errorf("failed to read %s: %v", file1, err)
	}

	// Read the second CSV file
	xys2, err := readCSV(file2)
	if err != nil {
		return fmt.Errorf("failed to read %s: %v", file2, err)
	}

	// Determine the length of the smallest dataset
	minLength := len(xys1)
	if len(xys2) < minLength {
		minLength = len(xys2)
	}

	// Truncate both datasets to the length of the smallest dataset
	xys1 = xys1[:minLength]
	xys2 = xys2[:minLength]

	// Create a new plot
	p := plot.New()
	p.Title.Text = "Comparison"
	p.X.Label.Text = "Date"
	p.Y.Label.Text = "Mean temperature"
	p.X.Tick.Marker = plot.TimeTicks{Format: "2006-01-02"}

	// Create a time series plotter for the first CSV file
	timeSeries1 := make(plotter.XYs, len(xys1))
	for i, xy := range xys1 {
		timeSeries1[i].X = xy.X
		timeSeries1[i].Y = xy.Y
	}
	line1, err := plotter.NewLine(timeSeries1)
	if err != nil {
		return fmt.Errorf("failed to create line plotter for %s: %v", file1, err)
	}
	line1.Color = plotutil.Color(0)
	p.Add(line1)
	p.Legend.Add("Real data", line1)

	// Create a time series plotter for the second CSV file
	timeSeries2 := make(plotter.XYs, len(xys2))
	for i, xy := range xys2 {
		timeSeries2[i].X = xy.X
		timeSeries2[i].Y = xy.Y
	}
	line2, err := plotter.NewLine(timeSeries2)
	if err != nil {
		return fmt.Errorf("failed to create line plotter for %s: %v", file2, err)
	}
	line2.Color = plotutil.Color(1)
	p.Add(line2)
	p.Legend.Add("Predicted data", line2)

	// Save the plot to a file
	if err := p.Save(8*vg.Inch, 8*vg.Inch, outputFile); err != nil {
		return fmt.Errorf("failed to save plot: %v", err)
	}

	return nil
}
