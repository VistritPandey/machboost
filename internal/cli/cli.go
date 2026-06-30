package cli

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"runtime"
	"strconv"
	"strings"
	"time"

	"machboost/internal/backend/ollama"
	"machboost/internal/bench"
	"machboost/internal/config"
	"machboost/internal/doctor"
	"machboost/internal/overlap"
	"machboost/internal/profile"
	"machboost/internal/runner"
)

func Execute(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		printUsage(stderr)
		return 2
	}

	switch args[0] {
	case "-h", "--help", "help":
		printUsage(stdout)
		return 0
	case "doctor":
		return runDoctor(args[1:], stdout, stderr)
	case "run":
		return runCommand(args[1:], stdout, stderr)
	case "bench":
		return runBench(args[1:], stdout, stderr)
	case "overlap":
		return runOverlap(args[1:], stdout, stderr)
	case "draft":
		return runDraft(args[1:], stdout, stderr)
	case "simulate-draft":
		return runSimulateDraft(args[1:], stdout, stderr)
	case "profile":
		return runProfile(args[1:], stdout, stderr)
	default:
		fmt.Fprintf(stderr, "unknown command %q\n\n", args[0])
		printUsage(stderr)
		return 2
	}
}

func runDoctor(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	report := doctor.Generate(context.Background())
	if *jsonOut {
		data, err := doctor.MarshalJSON(report)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintln(stdout, string(data))
		return 0
	}

	fmt.Fprintf(stdout, "machboost doctor\n")
	fmt.Fprintf(stdout, "OS: %s %s (%s) %s\n", report.OS.Name, report.OS.Version, report.OS.Build, report.OS.Arch)
	fmt.Fprintf(stdout, "CPU: %s, %d logical cores\n", emptyAs(report.Machine.CPUBrand, "unknown CPU"), report.Machine.CPUCount)
	fmt.Fprintf(stdout, "Memory: %s\n", humanBytes(report.Machine.MemoryBytes))
	if report.Machine.SwapUsage != "" {
		fmt.Fprintf(stdout, "Swap: %s\n", report.Machine.SwapUsage)
	}
	fmt.Fprintln(stdout, "\nRuntimes:")
	for _, rt := range report.Runtimes {
		status := "missing"
		if rt.Available {
			status = "available"
		}
		fmt.Fprintf(stdout, "- %s: %s", rt.Name, status)
		if rt.Path != "" {
			fmt.Fprintf(stdout, " (%s)", rt.Path)
		}
		if rt.Version != "" {
			fmt.Fprintf(stdout, " - %s", rt.Version)
		}
		fmt.Fprintln(stdout)
		for _, detail := range rt.Details {
			fmt.Fprintf(stdout, "  - %s\n", detail)
		}
	}
	if len(report.Warnings) > 0 {
		fmt.Fprintln(stdout, "\nWarnings:")
		for _, warning := range report.Warnings {
			fmt.Fprintf(stdout, "- %s\n", warning)
		}
	}
	return 0
}

func runCommand(args []string, stdout, stderr io.Writer) int {
	profileName := profile.DefaultProfile
	workload := profile.DefaultWorkload

	commandIndex := indexOfDoubleDash(args)
	flagArgs := args
	commandArgs := []string{}
	if commandIndex >= 0 {
		flagArgs = args[:commandIndex]
		commandArgs = args[commandIndex+1:]
	}

	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&profileName, "profile", profile.DefaultProfile, "profile: sustained, balanced, quiet, or config-defined")
	fs.StringVar(&workload, "workload", profile.DefaultWorkload, "workload: generic, llm, build, render")
	if err := fs.Parse(flagArgs); err != nil {
		return 2
	}
	if commandIndex < 0 {
		commandArgs = fs.Args()
	}
	if len(commandArgs) == 0 {
		fmt.Fprintln(stderr, "missing command; use: machboost run [flags] -- <command...>")
		return 2
	}

	cfg, _, err := config.Load(config.DefaultPath)
	if err != nil {
		fmt.Fprintf(stderr, "config error: %v\n", err)
		return 1
	}
	prof, err := profile.Resolve(profileName, workload, cfg, runtime.NumCPU())
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	for _, warning := range prof.Warnings {
		fmt.Fprintf(stderr, "warning: %s\n", warning)
	}

	result, err := runner.Run(context.Background(), commandArgs, prof.Env, prof.KeepAwake, stdout, stderr)
	for _, warning := range result.Warnings {
		fmt.Fprintf(stderr, "warning: %s\n", warning)
	}
	if err != nil {
		fmt.Fprintln(stderr, err)
		return result.ExitCode
	}
	return result.ExitCode
}

func runBench(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "missing bench subcommand: command or ollama")
		return 2
	}
	switch args[0] {
	case "command":
		return benchCommand(args[1:], stdout, stderr)
	case "compare":
		return benchCompare(args[1:], stdout, stderr)
	case "ollama":
		return benchOllama(args[1:], stdout, stderr)
	default:
		fmt.Fprintf(stderr, "unknown bench subcommand %q\n", args[0])
		return 2
	}
}

func benchCommand(args []string, stdout, stderr io.Writer) int {
	commandIndex := indexOfDoubleDash(args)
	commandArgs := args
	if commandIndex >= 0 {
		commandArgs = args[commandIndex+1:]
	}
	if len(commandArgs) == 0 {
		fmt.Fprintln(stderr, "missing command; use: machboost bench command -- <command...>")
		return 2
	}
	result, err := bench.Command(context.Background(), commandArgs, stdout, stderr)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return result.ExitCode
	}
	fmt.Fprintf(stdout, "\nbench command: duration=%dms exit_code=%d\n", result.DurationMS, result.ExitCode)
	return result.ExitCode
}

func benchCompare(args []string, stdout, stderr io.Writer) int {
	profileName := profile.DefaultProfile
	workload := profile.DefaultWorkload
	repeats := 3
	jsonOut := false

	commandIndex := indexOfDoubleDash(args)
	flagArgs := args
	commandArgs := []string{}
	if commandIndex >= 0 {
		flagArgs = args[:commandIndex]
		commandArgs = args[commandIndex+1:]
	}

	fs := flag.NewFlagSet("bench compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&profileName, "profile", profile.DefaultProfile, "profile: sustained, balanced, quiet, or config-defined")
	fs.StringVar(&workload, "workload", profile.DefaultWorkload, "workload: generic, llm, build, render")
	fs.IntVar(&repeats, "repeat", repeats, "number of baseline/boosted pairs to run")
	fs.BoolVar(&jsonOut, "json", false, "print JSON")
	if err := fs.Parse(flagArgs); err != nil {
		return 2
	}
	if commandIndex < 0 {
		commandArgs = fs.Args()
	}
	if len(commandArgs) == 0 {
		fmt.Fprintln(stderr, "missing command; use: machboost bench compare [flags] -- <command...>")
		return 2
	}
	if repeats < 1 {
		fmt.Fprintln(stderr, "repeat must be at least 1")
		return 2
	}

	cfg, _, err := config.Load(config.DefaultPath)
	if err != nil {
		fmt.Fprintf(stderr, "config error: %v\n", err)
		return 1
	}
	prof, err := profile.Resolve(profileName, workload, cfg, runtime.NumCPU())
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}

	result, err := bench.Compare(context.Background(), commandArgs, prof, repeats, stdout, stderr)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return firstNonZero(result)
	}
	if jsonOut {
		data, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintln(stdout, string(data))
		return firstNonZero(result)
	}

	fmt.Fprintf(stdout, "\nbench compare: profile=%s workload=%s repeats=%d\n", result.Profile, result.Workload, result.Repeats)
	fmt.Fprintf(stdout, "without machboost: avg=%dms runs=%s exit_codes=%s\n",
		result.Baseline.AverageMS, formatInt64s(result.Baseline.DurationsMS), formatInts(result.Baseline.ExitCodes))
	fmt.Fprintf(stdout, "with machboost:    avg=%dms runs=%s exit_codes=%s\n",
		result.Boosted.AverageMS, formatInt64s(result.Boosted.DurationsMS), formatInts(result.Boosted.ExitCodes))
	fmt.Fprintf(stdout, "delta: %dms, speedup: %.2fx\n", result.DeltaMS, result.Speedup)
	fmt.Fprintf(stdout, "verdict: %s (%.1f%%)\n", result.Verdict, result.ChangePercent)
	for _, warning := range result.Warnings {
		fmt.Fprintf(stdout, "warning: %s\n", warning)
	}
	return firstNonZero(result)
}

func benchOllama(args []string, stdout, stderr io.Writer) int {
	model := "qwen3:8b"
	prompt := ""
	tokens := 64
	ctxLen := 4096
	jsonOut := false

	fs := flag.NewFlagSet("bench ollama", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&model, "model", model, "Ollama model name")
	fs.StringVar(&prompt, "prompt", prompt, "benchmark prompt")
	fs.IntVar(&tokens, "tokens", tokens, "num_predict value")
	fs.IntVar(&ctxLen, "ctx", ctxLen, "num_ctx value")
	fs.BoolVar(&jsonOut, "json", false, "print JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	result, err := ollama.NewClient("").Bench(ctx, model, prompt, tokens, ctxLen)
	if err != nil {
		fmt.Fprintf(stderr, "ollama bench failed: %v\n", err)
		return 1
	}
	if jsonOut {
		data, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintln(stdout, string(data))
		return 0
	}
	fmt.Fprintf(stdout, "ollama bench: model=%s eval_tokens=%d tokens_per_sec=%.2f duration=%dms load=%dms total=%dms\n",
		result.Model, result.EvalCount, result.TokensPerSec, result.DurationMS, result.LoadMS, result.TotalMS)
	for _, warning := range result.Warnings {
		fmt.Fprintf(stdout, "warning: %s\n", warning)
	}
	return 0
}

func runOverlap(args []string, stdout, stderr io.Writer) int {
	promptPath := ""
	outputPath := ""
	contextPath := ""
	ngram := 4
	jsonOut := false

	fs := flag.NewFlagSet("overlap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&promptPath, "prompt", "", "prompt/input transcript file")
	fs.StringVar(&outputPath, "output", "", "generated output transcript file")
	fs.StringVar(&contextPath, "context", "", "optional context file or directory")
	fs.IntVar(&ngram, "ngram", ngram, "minimum copied n-gram size")
	fs.BoolVar(&jsonOut, "json", false, "print JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if promptPath == "" || outputPath == "" {
		fmt.Fprintln(stderr, "missing required flags; use: machboost overlap --prompt prompt.txt --output output.txt [--context dir]")
		return 2
	}

	report, err := overlap.Analyze(overlap.Options{
		PromptPath:  promptPath,
		OutputPath:  outputPath,
		ContextPath: contextPath,
		NGram:       ngram,
	})
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if jsonOut {
		data, err := overlap.MarshalJSON(report)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintln(stdout, string(data))
		return 0
	}

	fmt.Fprintf(stdout, "machboost overlap: verdict=%s\n", report.Verdict)
	fmt.Fprintf(stdout, "tokens: prompt=%d context=%d output=%d source=%d\n",
		report.PromptTokens, report.ContextTokens, report.OutputTokens, report.SourceTokens)
	fmt.Fprintf(stdout, "coverage: %.1f%% (%d/%d output tokens), longest copied span=%d tokens\n",
		report.CoveragePercent, report.CoveredTokens, report.OutputTokens, report.LongestCopiedSpan)
	fmt.Fprintf(stdout, "estimated serial decode step save: %.1f%%\n", report.EstimatedStepSave)
	if report.ContextFilesScanned > 0 {
		fmt.Fprintf(stdout, "context files scanned: %d\n", report.ContextFilesScanned)
	}
	return 0
}

func runDraft(args []string, stdout, stderr io.Writer) int {
	prefixPath := ""
	contextPath := ""
	ngram := 4
	maxDraftTokens := 32
	limit := 5
	jsonOut := false

	fs := flag.NewFlagSet("draft", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&prefixPath, "prefix", "", "prefix transcript file")
	fs.StringVar(&contextPath, "context", "", "context file or directory")
	fs.IntVar(&ngram, "ngram", ngram, "minimum matched suffix size")
	fs.IntVar(&maxDraftTokens, "max-tokens", maxDraftTokens, "maximum draft tokens per candidate")
	fs.IntVar(&limit, "limit", limit, "maximum candidates")
	fs.BoolVar(&jsonOut, "json", false, "print JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if prefixPath == "" || contextPath == "" {
		fmt.Fprintln(stderr, "missing required flags; use: machboost draft --prefix prefix.txt --context path")
		return 2
	}

	report, err := overlap.DraftCandidates(overlap.CandidateOptions{
		PrefixPath:     prefixPath,
		ContextPath:    contextPath,
		NGram:          ngram,
		MaxDraftTokens: maxDraftTokens,
		Limit:          limit,
	})
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if jsonOut {
		data, err := overlap.MarshalCandidateReport(report)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintln(stdout, string(data))
		return 0
	}

	fmt.Fprintf(stdout, "machboost draft: verdict=%s candidates=%d\n", report.Verdict, len(report.Candidates))
	fmt.Fprintf(stdout, "tokens: prefix=%d context=%d, context files=%d\n",
		report.PrefixTokens, report.ContextTokens, report.ContextFilesScanned)
	for i, candidate := range report.Candidates {
		fmt.Fprintf(stdout, "\ncandidate %d: score=%.2f matched_suffix_tokens=%d draft_tokens=%d\n",
			i+1, candidate.Score, candidate.MatchedSuffixTokens, candidate.TokenCount)
		fmt.Fprintf(stdout, "matched suffix: %s\n", candidate.MatchedSuffix)
		fmt.Fprintf(stdout, "draft: %s\n", candidate.DraftText)
	}
	return 0
}

func runSimulateDraft(args []string, stdout, stderr io.Writer) int {
	promptPath := ""
	outputPath := ""
	contextPath := ""
	ngram := 4
	maxDraftTokens := 32
	jsonOut := false

	fs := flag.NewFlagSet("simulate-draft", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&promptPath, "prompt", "", "prompt/input transcript file")
	fs.StringVar(&outputPath, "output", "", "known generated output file")
	fs.StringVar(&contextPath, "context", "", "optional context file or directory")
	fs.IntVar(&ngram, "ngram", ngram, "minimum copied n-gram size")
	fs.IntVar(&maxDraftTokens, "max-tokens", maxDraftTokens, "maximum draft tokens per verification pass")
	fs.BoolVar(&jsonOut, "json", false, "print JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if promptPath == "" || outputPath == "" {
		fmt.Fprintln(stderr, "missing required flags; use: machboost simulate-draft --prompt prompt.txt --output output.txt [--context dir]")
		return 2
	}

	sim, err := overlap.SimulateDraft(overlap.DraftOptions{
		PromptPath:     promptPath,
		OutputPath:     outputPath,
		ContextPath:    contextPath,
		NGram:          ngram,
		MaxDraftTokens: maxDraftTokens,
	})
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if jsonOut {
		data, err := overlap.MarshalDraftSimulation(sim)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintln(stdout, string(data))
		return 0
	}

	fmt.Fprintf(stdout, "machboost simulate-draft: verdict=%s\n", sim.Verdict)
	fmt.Fprintf(stdout, "baseline decode steps: %d\n", sim.BaselineDecodeSteps)
	fmt.Fprintf(stdout, "simulated verify passes: %d\n", sim.SimulatedVerifyPasses)
	fmt.Fprintf(stdout, "accepted draft tokens: %d in %d span(s)\n", sim.AcceptedDraftTokens, sim.AcceptedDraftSpans)
	fmt.Fprintf(stdout, "normal decode tokens: %d\n", sim.NormalDecodeTokens)
	fmt.Fprintf(stdout, "longest accepted draft: %d tokens\n", sim.LongestAcceptedDraft)
	fmt.Fprintf(stdout, "step reduction: %.1f%%, estimated speedup: %.2fx\n", sim.StepReductionPercent, sim.EstimatedSpeedup)
	if sim.ContextFilesScanned > 0 {
		fmt.Fprintf(stdout, "context files scanned: %d\n", sim.ContextFilesScanned)
	}
	fmt.Fprintln(stdout, "note: simulation is idealized; real acceleration requires decoder verification integration.")
	return 0
}

func runProfile(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "missing profile subcommand: init")
		return 2
	}
	switch args[0] {
	case "init":
		if err := config.Init(config.DefaultPath); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintf(stdout, "created %s\n", config.DefaultPath)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown profile subcommand %q\n", args[0])
		return 2
	}
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, `machboost runs heavy local workloads under safe performance profiles.

Usage:
  machboost doctor [--json]
  machboost run [--profile sustained|balanced|quiet] [--workload generic|llm|build|render] -- <command...>
  machboost bench command -- <command...>
  machboost bench compare [--profile name] [--workload type] [--repeat n] [--json] -- <command...>
  machboost bench ollama [--model name] [--tokens n] [--ctx n] [--json]
  machboost overlap --prompt prompt.txt --output output.txt [--context path] [--json]
  machboost draft --prefix prefix.txt --context path [--json]
  machboost simulate-draft --prompt prompt.txt --output output.txt [--context path] [--json]
  machboost profile init`)
}

func indexOfDoubleDash(args []string) int {
	for i, arg := range args {
		if arg == "--" {
			return i
		}
	}
	return -1
}

func humanBytes(bytes int64) string {
	if bytes <= 0 {
		return "unknown"
	}
	const unit = 1024
	if bytes < unit {
		return strconv.FormatInt(bytes, 10) + " B"
	}
	div := int64(unit)
	exp := 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

func emptyAs(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func formatInt64s(values []int64) string {
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, strconv.FormatInt(value, 10)+"ms")
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func formatInts(values []int) string {
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, strconv.Itoa(value))
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func firstNonZero(result bench.CompareResult) int {
	for _, code := range result.Baseline.ExitCodes {
		if code != 0 {
			return code
		}
	}
	for _, code := range result.Boosted.ExitCodes {
		if code != 0 {
			return code
		}
	}
	return 0
}
