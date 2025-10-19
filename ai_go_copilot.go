package main

import (
	"fmt"
	"log"
	"math"
	"time"

	"github.com/montanaflynn/stats"
)

// Dataset represents x and y values for regression
type Dataset struct {
	X []float64
	Y []float64
}

// RegressionResult holds regression analysis results
type RegressionResult struct {
	Dataset   string
	Slope     float64
	Intercept float64
	RSquared  float64
	Duration  time.Duration
}

// LoadAnscombeDatasets returns the four Anscombe Quartet datasets
func LoadAnscombeDatasets() map[string]Dataset {
	return map[string]Dataset{
		"I": {
			X: []float64{10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0},
			Y: []float64{8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68},
		},
		"II": {
			X: []float64{10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0},
			Y: []float64{9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74},
		},
		"III": {
			X: []float64{10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0},
			Y: []float64{7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73},
		},
		"IV": {
			X: []float64{8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0},
			Y: []float64{6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89},
		},
	}
}

// Copilot: improve the PerformLinearRegression function to handle NaN and Inf values and return the ManualRegression function calculation as a fallback if montanaflynn/stats fails.
// Note: This function may return results from either the montanaflynn/stats package or the manual fallback implementation.
// This can affect reproducibility and debugging, as results may differ slightly depending on which method is used.

func PerformLinearRegression(x, y []float64) (slope, intercept, rSquared float64, err error) {
	// Basic validation
	if len(x) != len(y) {
		return 0, 0, 0, fmt.Errorf("x and y length mismatch: %d vs %d", len(x), len(y))
	}
	if len(x) < 2 {
		return 0, 0, 0, fmt.Errorf("need at least two data points")
	}

	// Helper function to check for NaN or Inf
	isInvalid := func(v float64) bool {
		return math.IsNaN(v) || math.IsInf(v, 0)
	}

	// Clean NaN/Inf values
	cleanX := make([]float64, 0, len(x))
	cleanY := make([]float64, 0, len(y))
	coords := make([]stats.Coordinate, 0, len(x))
	for i := range x {
		xi, yi := x[i], y[i]
		if isInvalid(xi) || isInvalid(yi) {
			// skip invalid pair
			continue
		}
		cleanX = append(cleanX, xi)
		cleanY = append(cleanY, yi)
		coords = append(coords, stats.Coordinate{X: xi, Y: yi})
	}

	if len(cleanX) < 2 {
		return 0, 0, 0, fmt.Errorf("not enough valid points after removing NaN/Inf (have %d)", len(cleanX))
	}

	// Try using the library's linear regression first
	regressionLine, lrErr := stats.LinearRegression(coords)
	if lrErr != nil || len(regressionLine) < 2 {
		// Fallback: use manual least-squares calculation
		slope, intercept, rSquared = ManualRegression(cleanX, cleanY)
		fmt.Printf("\nWarning: falling back to manual regression due to error: %v", lrErr)
		return slope, intercept, rSquared, nil
	}

	// Compute slope and intercept from regressionLine endpoints
	first := regressionLine[0]
	last := regressionLine[len(regressionLine)-1]

	// Validate endpoints
	if isInvalid(first.X) || isInvalid(first.Y) || isInvalid(last.X) || isInvalid(last.Y) {
		// fallback to manual method if regression endpoints are invalid
		slope, intercept, rSquared = ManualRegression(cleanX, cleanY)
		fmt.Printf("\nWarning: falling back to manual regression due to invalid regression line endpoints")
		return slope, intercept, rSquared, nil
	}

	// Protect against division by zero if Xs are identical
	if math.Abs(last.X-first.X) < 1e-12 {
		slope, intercept, rSquared = ManualRegression(cleanX, cleanY)
		fmt.Printf("\nWarning: falling back to manual regression due to vertical line (identical X values)")
		return slope, intercept, rSquared, nil
	}

	// Use library line
	slope = (last.Y - first.Y) / (last.X - first.X)
	intercept = first.Y - slope*first.X

	// Compute R-squared via correlation; fallback to manual R² if correlation fails or corr is NaN
	corr, corrErr := stats.Correlation(cleanX, cleanY)
	if corrErr != nil || math.IsNaN(corr) {
		_, _, rSquared = ManualRegression(cleanX, cleanY)
		fmt.Printf("\nWarning: falling back to manual R² calculation due to error: %v", corrErr)
	} else {
		rSquared = corr * corr
	}

	return slope, intercept, rSquared, nil
}

// ManualRegression alternative implementation using basic formulas
// Ensuring match R/Python results exactly if needed
func ManualRegression(x, y []float64) (slope, intercept, rSquared float64) {
	n := float64(len(x))

	var sumX, sumY, sumXY, sumXX, sumYY float64
	for i := range x {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumXX += x[i] * x[i]
		sumYY += y[i] * y[i]
	}

	// Least squares formulas
	den := n*sumXX - sumX*sumX
	if den == 0 {
		// degenerate case: treat slope as 0 to avoid division by zero
		slope = 0
		intercept = sumY / n
	} else {
		slope = (n*sumXY - sumX*sumY) / den
		intercept = (sumY - slope*sumX) / n
	}

	// Calculate R-squared manually
	ssTotal := sumYY - (sumY*sumY)/n
	ssResidual := 0.0
	for i := range x {
		predicted := intercept + slope*x[i]
		residual := y[i] - predicted
		ssResidual += residual * residual
	}

	if ssTotal > 0 {
		rSquared = 1 - (ssResidual / ssTotal)
	} else {
		// If total variance is zero, define R² as 1 when residual is zero, else 0
		if ssResidual == 0 {
			rSquared = 1
		} else {
			rSquared = 0
		}
	}

	return slope, intercept, rSquared
}

func main() {
	fmt.Println("=== Anscombe Quartet Regression Analysis ===")
	fmt.Println("Loading datasets and performing linear regression...")

	datasets := LoadAnscombeDatasets()
	results := make([]RegressionResult, 0, 4)

	// Perform regression on all datasets
	overallStart := time.Now()

	for name, data := range datasets {
		start := time.Now()

		slope, intercept, rSquared, err := PerformLinearRegression(data.X, data.Y)
		if err != nil {
			log.Printf("Regression failed for dataset %s: %v", name, err)
			continue
		}

		elapsed := time.Since(start)

		results = append(results, RegressionResult{
			Dataset:   name,
			Slope:     slope,
			Intercept: intercept,
			RSquared:  rSquared,
			Duration:  elapsed,
		})

		fmt.Printf("\nDataset %s:\n", name)
		fmt.Printf("  Slope:     %.6f\n", slope)
		fmt.Printf("  Intercept: %.6f\n", intercept)
		fmt.Printf("  R-squared: %.6f\n", rSquared)
		fmt.Printf("  Time:      %v\n", elapsed)
	}

	totalTime := time.Since(overallStart)

	fmt.Printf("\n=== Summary ===\n")
	fmt.Printf("Total execution time: %v\n", totalTime)
	if len(datasets) > 0 {
		avgSeconds := totalTime.Seconds() / float64(len(datasets))
		fmt.Printf("Average per dataset:  %.6fs\n", avgSeconds)
	} else {
		fmt.Printf("Average per dataset:  N/A (no datasets)\n")
	}
	// Reference values below are based on standard linear regression results for the original Anscombe Quartet datasets.
	// These values were obtained using R's lm() function and Python's statsmodels. See:
	// https://en.wikipedia.org/wiki/Anscombe%27s_quartet
	// For reproducibility, recalculate these if the dataset changes.
	fmt.Printf("\n=== Expected Results (R/Python Reference) ===\n")
	fmt.Printf("All datasets should have approximately:\n")
	fmt.Printf("  Slope:     0.500091\n")
	fmt.Printf("  Intercept: 3.000091\n")
	fmt.Printf("  R-squared: 0.666542\n")
}
