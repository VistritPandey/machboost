package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExecuteHelp(t *testing.T) {
	var stdout bytes.Buffer
	code := Execute([]string{"--help"}, &stdout, &bytes.Buffer{})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if !strings.Contains(stdout.String(), "machboost doctor") {
		t.Fatalf("help output missing doctor command: %s", stdout.String())
	}
}

func TestExecuteRunPreservesExitCode(t *testing.T) {
	code := Execute([]string{"run", "--profile", "balanced", "--workload", "generic", "--", "sh", "-c", "exit 3"}, &bytes.Buffer{}, &bytes.Buffer{})
	if code != 3 {
		t.Fatalf("exit code = %d, want 3", code)
	}
}

func TestExecuteBenchCommand(t *testing.T) {
	var stdout bytes.Buffer
	code := Execute([]string{"bench", "command", "--", "sh", "-c", "printf ok"}, &stdout, &bytes.Buffer{})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	output := stdout.String()
	if !strings.Contains(output, "ok") || !strings.Contains(output, "bench command") {
		t.Fatalf("unexpected output: %q", output)
	}
}

func TestExecuteBenchCompare(t *testing.T) {
	var stdout bytes.Buffer
	code := Execute([]string{"bench", "compare", "--profile", "balanced", "--workload", "generic", "--repeat", "1", "--", "sh", "-c", "printf ok"}, &stdout, &bytes.Buffer{})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	output := stdout.String()
	if !strings.Contains(output, "bench compare") || !strings.Contains(output, "without machboost") || !strings.Contains(output, "with machboost") {
		t.Fatalf("unexpected output: %q", output)
	}
}

func TestExecuteOverlap(t *testing.T) {
	dir := t.TempDir()
	prompt := filepath.Join(dir, "prompt.txt")
	output := filepath.Join(dir, "output.txt")
	if err := os.WriteFile(prompt, []byte("alpha beta gamma delta"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(output, []byte("beta gamma delta"), 0644); err != nil {
		t.Fatal(err)
	}

	var stdout bytes.Buffer
	code := Execute([]string{"overlap", "--prompt", prompt, "--output", output, "--ngram", "2"}, &stdout, &bytes.Buffer{})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if !strings.Contains(stdout.String(), "strong_candidate") {
		t.Fatalf("unexpected output: %q", stdout.String())
	}
}
