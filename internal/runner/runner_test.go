package runner

import (
	"bytes"
	"context"
	"testing"
)

func TestRunPreservesExitCode(t *testing.T) {
	result, err := Run(context.Background(), []string{"sh", "-c", "exit 7"}, nil, false, &bytes.Buffer{}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if result.ExitCode != 7 {
		t.Fatalf("exit code = %d, want 7", result.ExitCode)
	}
}

func TestRunAppliesEnvironmentOverrides(t *testing.T) {
	var stdout bytes.Buffer
	result, err := Run(context.Background(), []string{"sh", "-c", "printf %s \"$MACHBOOST_TEST\""}, map[string]string{"MACHBOOST_TEST": "yes"}, false, &stdout, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if result.ExitCode != 0 {
		t.Fatalf("exit code = %d, want 0", result.ExitCode)
	}
	if got := stdout.String(); got != "yes" {
		t.Fatalf("stdout = %q, want yes", got)
	}
}
