package main

import (
	"math"
	"testing"
	"time"
)

// ✅ Test 1: Coefficient accuracy
func TestRegressionCoefficients(t *testing.T) {
	datasets := LoadAnscombeDatasets()
	tolerance := 0.01
	expectedSlope := 0.500
	expectedIntercept := 3.000

	for name, data := range datasets {
		slope, intercept, _, err := PerformLinearRegression(data.X, data.Y)
		if err != nil {
			t.Errorf("Dataset %s: regression failed: %v", name, err)
			continue
		}
		if math.Abs(slope-expectedSlope) > tolerance {
			t.Errorf("%s slope mismatch: got %.4f, expected ~%.4f", name, slope, expectedSlope)
		}
		if math.Abs(intercept-expectedIntercept) > tolerance {
			t.Errorf("%s intercept mismatch: got %.4f, expected ~%.4f", name, intercept, expectedIntercept)
		}
	}
}

// ✅ Test 2: Consistency across datasets
func TestDatasetConsistency(t *testing.T) {
	datasets := LoadAnscombeDatasets()
	tolerance := 0.01
	refSlope, refIntercept, _, _ := PerformLinearRegression(datasets["I"].X, datasets["I"].Y)

	for name, data := range datasets {
		if name == "I" {
			continue
		}
		slope, intercept, _, err := PerformLinearRegression(data.X, data.Y)
		if err != nil {
			t.Errorf("Dataset %s failed: %v", name, err)
			continue
		}
		if math.Abs(slope-refSlope) > tolerance {
			t.Errorf("%s slope inconsistent: got %.4f, expected ~%.4f", name, slope, refSlope)
		}
		if math.Abs(intercept-refIntercept) > tolerance {
			t.Errorf("%s intercept inconsistent: got %.4f, expected ~%.4f", name, intercept, refIntercept)
		}
	}
}

// ✅ Test 3: Performance
func TestExecutionTime(t *testing.T) {
	datasets := LoadAnscombeDatasets()
	maxTime := 100 * time.Millisecond

	start := time.Now()
	for _, data := range datasets {
		_, _, _, err := PerformLinearRegression(data.X, data.Y)
		if err != nil {
			t.Errorf("Regression failed: %v", err)
		}
	}
	elapsed := time.Since(start)
	if elapsed > maxTime {
		t.Errorf("Regression took too long: %v (max %v)", elapsed, maxTime)
	}
}

// ✅ Benchmark 1: All datasets
func BenchmarkRegression(b *testing.B) {
	datasets := LoadAnscombeDatasets()
	for i := 0; i < b.N; i++ {
		for _, data := range datasets {
			_, _, _, _ = PerformLinearRegression(data.X, data.Y)
		}
	}
}

// ✅ Benchmark 2: Individual datasets
func BenchmarkIndividualDatasets(b *testing.B) {
	datasets := LoadAnscombeDatasets()
	for name, data := range datasets {
		b.Run("Dataset_"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _, _, _ = PerformLinearRegression(data.X, data.Y)
			}
		})
	}
}
