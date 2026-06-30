package bench

import (
	"context"
	"io"
	"time"

	"machboost/internal/profile"
	"machboost/internal/runner"
)

type CommandResult struct {
	SchemaVersion string   `json:"schema_version"`
	Backend       string   `json:"backend"`
	Command       []string `json:"command"`
	ExitCode      int      `json:"exit_code"`
	DurationMS    int64    `json:"duration_ms"`
}

type CompareResult struct {
	SchemaVersion string      `json:"schema_version"`
	Backend       string      `json:"backend"`
	Command       []string    `json:"command"`
	Profile       string      `json:"profile"`
	Workload      string      `json:"workload"`
	Repeats       int         `json:"repeats"`
	Baseline      CompareSide `json:"baseline"`
	Boosted       CompareSide `json:"boosted"`
	DeltaMS       int64       `json:"delta_ms"`
	Speedup       float64     `json:"speedup"`
	ChangePercent float64     `json:"change_percent"`
	Verdict       string      `json:"verdict"`
	Warnings      []string    `json:"warnings,omitempty"`
}

type CompareSide struct {
	Label       string  `json:"label"`
	ExitCodes   []int   `json:"exit_codes"`
	DurationsMS []int64 `json:"durations_ms"`
	AverageMS   int64   `json:"average_ms"`
}

func Command(ctx context.Context, args []string, stdout, stderr io.Writer) (CommandResult, error) {
	start := time.Now()
	result, err := runner.Run(ctx, args, nil, false, stdout, stderr)
	duration := time.Since(start)
	return CommandResult{
		SchemaVersion: "machboost.bench.v1",
		Backend:       "command",
		Command:       append([]string{}, args...),
		ExitCode:      result.ExitCode,
		DurationMS:    duration.Milliseconds(),
	}, err
}

func Compare(ctx context.Context, args []string, prof profile.Profile, repeats int, stdout, stderr io.Writer) (CompareResult, error) {
	if repeats < 1 {
		repeats = 1
	}

	result := CompareResult{
		SchemaVersion: "machboost.bench.compare.v1",
		Backend:       "compare",
		Command:       append([]string{}, args...),
		Profile:       prof.Name,
		Workload:      prof.Workload,
		Repeats:       repeats,
		Baseline:      CompareSide{Label: "without machboost"},
		Boosted:       CompareSide{Label: "with machboost"},
		Warnings: []string{
			"Benchmark results can be affected by cache warmth, background load, thermal state, and workload variance.",
		},
	}

	for i := 0; i < repeats; i++ {
		baselineDuration, baselineRun, err := timedRun(ctx, args, nil, false, stdout, stderr)
		result.Baseline.DurationsMS = append(result.Baseline.DurationsMS, baselineDuration.Milliseconds())
		result.Baseline.ExitCodes = append(result.Baseline.ExitCodes, baselineRun.ExitCode)
		if err != nil {
			return result, err
		}

		boostedDuration, boostedRun, err := timedRun(ctx, args, prof.Env, prof.KeepAwake, stdout, stderr)
		result.Boosted.DurationsMS = append(result.Boosted.DurationsMS, boostedDuration.Milliseconds())
		result.Boosted.ExitCodes = append(result.Boosted.ExitCodes, boostedRun.ExitCode)
		result.Warnings = append(result.Warnings, boostedRun.Warnings...)
		if err != nil {
			return result, err
		}
	}

	result.Baseline.AverageMS = average(result.Baseline.DurationsMS)
	result.Boosted.AverageMS = average(result.Boosted.DurationsMS)
	result.DeltaMS = result.Baseline.AverageMS - result.Boosted.AverageMS
	if result.Boosted.AverageMS > 0 {
		result.Speedup = float64(result.Baseline.AverageMS) / float64(result.Boosted.AverageMS)
	}
	if result.Baseline.AverageMS > 0 {
		result.ChangePercent = (float64(result.DeltaMS) / float64(result.Baseline.AverageMS)) * 100
	}
	result.Verdict = verdict(result.Speedup)
	if !sameExitCodes(result.Baseline.ExitCodes, result.Boosted.ExitCodes) {
		result.Warnings = append(result.Warnings, "baseline and boosted runs produced different exit codes; compare the workload output before trusting timing.")
		result.Verdict = "invalid"
	}

	return result, nil
}

func timedRun(ctx context.Context, args []string, env map[string]string, keepAwake bool, stdout, stderr io.Writer) (time.Duration, runner.Result, error) {
	start := time.Now()
	result, err := runner.Run(ctx, args, env, keepAwake, stdout, stderr)
	return time.Since(start), result, err
}

func average(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}
	var total int64
	for _, value := range values {
		total += value
	}
	return total / int64(len(values))
}

func sameExitCodes(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func verdict(speedup float64) string {
	if speedup >= 1.05 {
		return "faster"
	}
	if speedup <= 0.95 {
		return "slower"
	}
	return "no_clear_change"
}
