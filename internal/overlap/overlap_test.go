package overlap

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTokenize(t *testing.T) {
	tokens := Tokenize("Hello, MachBoost_v2!")
	want := []string{"hello", ",", "machboost_v2", "!"}
	if len(tokens) != len(want) {
		t.Fatalf("tokens = %#v", tokens)
	}
	for i := range want {
		if tokens[i] != want[i] {
			t.Fatalf("token %d = %q, want %q", i, tokens[i], want[i])
		}
	}
}

func TestAnalyzeFindsCopiedSpan(t *testing.T) {
	dir := t.TempDir()
	prompt := filepath.Join(dir, "prompt.txt")
	output := filepath.Join(dir, "output.txt")
	if err := os.WriteFile(prompt, []byte("Alpha beta gamma delta epsilon zeta."), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(output, []byte("beta gamma delta epsilon"), 0644); err != nil {
		t.Fatal(err)
	}

	report, err := Analyze(Options{PromptPath: prompt, OutputPath: output, NGram: 2})
	if err != nil {
		t.Fatal(err)
	}
	if report.CoveragePercent != 100 {
		t.Fatalf("coverage = %.2f, want 100", report.CoveragePercent)
	}
	if report.LongestCopiedSpan != 4 {
		t.Fatalf("longest span = %d, want 4", report.LongestCopiedSpan)
	}
	if report.Verdict != "strong_candidate" {
		t.Fatalf("verdict = %q", report.Verdict)
	}
}

func TestAnalyzeIncludesContext(t *testing.T) {
	dir := t.TempDir()
	prompt := filepath.Join(dir, "prompt.txt")
	output := filepath.Join(dir, "output.txt")
	contextDir := filepath.Join(dir, "context")
	if err := os.Mkdir(contextDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(prompt, []byte("Short prompt."), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(output, []byte("rare context phrase"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(contextDir, "file.txt"), []byte("rare context phrase lives here"), 0644); err != nil {
		t.Fatal(err)
	}

	report, err := Analyze(Options{PromptPath: prompt, OutputPath: output, ContextPath: contextDir, NGram: 2})
	if err != nil {
		t.Fatal(err)
	}
	if report.ContextFilesScanned != 1 {
		t.Fatalf("context files = %d, want 1", report.ContextFilesScanned)
	}
	if report.CoveragePercent != 100 {
		t.Fatalf("coverage = %.2f, want 100", report.CoveragePercent)
	}
}
