package runner

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
)

type Result struct {
	ExitCode int
	Warnings []string
}

func Run(ctx context.Context, args []string, env map[string]string, keepAwake bool, stdout, stderr io.Writer) (Result, error) {
	if len(args) == 0 {
		return Result{ExitCode: 2}, fmt.Errorf("missing command")
	}

	commandName := args[0]
	commandArgs := args[1:]
	var warnings []string

	if keepAwake && runtime.GOOS == "darwin" {
		if _, err := exec.LookPath("caffeinate"); err == nil {
			commandName = "caffeinate"
			commandArgs = append([]string{"-dimsu"}, args...)
		} else {
			warnings = append(warnings, "caffeinate was not found; running without keep-awake wrapping")
		}
	}

	cmd := exec.CommandContext(ctx, commandName, commandArgs...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.Stdin = os.Stdin
	cmd.Env = mergeEnv(os.Environ(), env)

	err := cmd.Run()
	if err == nil {
		return Result{ExitCode: 0, Warnings: warnings}, nil
	}
	if exitErr, ok := err.(*exec.ExitError); ok {
		return Result{ExitCode: exitErr.ExitCode(), Warnings: warnings}, nil
	}
	return Result{ExitCode: 1, Warnings: warnings}, err
}

func mergeEnv(base []string, overrides map[string]string) []string {
	if len(overrides) == 0 {
		return base
	}
	indexByKey := map[string]int{}
	for i, item := range base {
		for j, ch := range item {
			if ch == '=' {
				indexByKey[item[:j]] = i
				break
			}
		}
	}
	merged := append([]string{}, base...)
	for key, value := range overrides {
		entry := key + "=" + value
		if index, ok := indexByKey[key]; ok {
			merged[index] = entry
			continue
		}
		indexByKey[key] = len(merged)
		merged = append(merged, entry)
	}
	return merged
}
