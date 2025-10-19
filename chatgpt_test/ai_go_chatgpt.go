package main

import (
	"fmt"
	"log"
	"time"

	"github.com/montanaflynn/stats"
)

type RegressionResult struct {
	Name      string
	Slope     float64
	Intercept float64
	Elapsed   time.Duration
}

func main() {
	// Anscombe's Quartet datasets
	datasets := map[string][]stats.Coordinate{
		"I": {
			{10.0, 8.04}, {8.0, 6.95}, {13.0, 7.58}, {9.0, 8.81}, {11.0, 8.33},
			{14.0, 9.96}, {6.0, 7.24}, {4.0, 4.26}, {12.0, 10.84}, {7.0, 4.82}, {5.0, 5.68},
		},
		"II": {
			{10.0, 9.14}, {8.0, 8.14}, {13.0, 8.74}, {9.0, 8.77}, {11.0, 9.26},
			{14.0, 8.10}, {6.0, 6.13}, {4.0, 3.10}, {12.0, 9.13}, {7.0, 7.26}, {5.0, 4.74},
		},
		"III": {
			{10.0, 7.46}, {8.0, 6.77}, {13.0, 12.74}, {9.0, 7.11}, {11.0, 7.81},
			{14.0, 8.84}, {6.0, 6.08}, {4.0, 5.39}, {12.0, 8.15}, {7.0, 6.42}, {5.0, 5.73},
		},
		"IV": {
			{8.0, 6.58}, {8.0, 5.76}, {8.0, 7.71}, {8.0, 8.84}, {8.0, 8.47},
			{8.0, 7.04}, {8.0, 5.25}, {19.0, 12.50}, {8.0, 5.56}, {8.0, 7.91}, {8.0, 6.89},
		},
	}

	fmt.Println("Performing linear regression using montanaflynn/stats...\n")

	var results []RegressionResult

	for name, data := range datasets {
		start := time.Now()

		// Separate X and Y values
		var xVals, yVals stats.Float64Data
		for _, pt := range data {
			xVals = append(xVals, pt.X)
			yVals = append(yVals, pt.Y)
		}

		// Perform regression using the stats package
		_, err := stats.LinearRegression(data)
		if err != nil {
			log.Fatalf("Error computing regression for dataset %s: %v", name, err)
		}

		// Compute slope and intercept using stats-based means
		slope, intercept := computeSlopeIntercept(xVals, yVals)
		elapsed := time.Since(start)

		results = append(results, RegressionResult{
			Name:      name,
			Slope:     slope,
			Intercept: intercept,
			Elapsed:   elapsed,
		})

		fmt.Printf("Dataset %s:\n", name)
		fmt.Printf("  Slope:     %.4f\n", slope)
		fmt.Printf("  Intercept: %.4f\n", intercept)
		fmt.Printf("  Time Elapsed: %s\n\n", elapsed)
	}

	// Print formatted summary table
	fmt.Println("📊 Summary Table — Anscombe’s Quartet Linear Regression Results")
	fmt.Println("--------------------------------------------------------------")
	fmt.Printf("%-10s %-12s %-12s %-15s\n", "Dataset", "Slope", "Intercept", "Elapsed Time")
	fmt.Println("--------------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-10s %-12.4f %-12.4f %-15s\n", r.Name, r.Slope, r.Intercept, r.Elapsed)
	}
	fmt.Println("--------------------------------------------------------------")
	fmt.Println("All regressions complete ✅")
}

// computeSlopeIntercept calculates slope and intercept using montanaflynn/stats
func computeSlopeIntercept(xVals, yVals stats.Float64Data) (slope, intercept float64) {
	meanX, err := stats.Mean(xVals)
	if err != nil {
		log.Fatal(err)
	}
	meanY, err := stats.Mean(yVals)
	if err != nil {
		log.Fatal(err)
	}

	var numerator, denominator float64
	for i := range xVals {
		numerator += (xVals[i] - meanX) * (yVals[i] - meanY)
		denominator += (xVals[i] - meanX) * (xVals[i] - meanX)
	}

	slope = numerator / denominator
	intercept = meanY - slope*meanX
	return
}
