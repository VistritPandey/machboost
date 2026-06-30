package overlap

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSimulateDraftFindsStepReduction(t *testing.T) {
	dir := t.TempDir()
	prompt := filepath.Join(dir, "prompt.txt")
	output := filepath.Join(dir, "output.txt")
	if err := os.WriteFile(prompt, []byte("alpha beta gamma delta epsilon zeta"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(output, []byte("beta gamma delta epsilon zeta"), 0644); err != nil {
		t.Fatal(err)
	}

	sim, err := SimulateDraft(DraftOptions{PromptPath: prompt, OutputPath: output, NGram: 2, MaxDraftTokens: 8})
	if err != nil {
		t.Fatal(err)
	}
	if sim.BaselineDecodeSteps != 5 {
		t.Fatalf("baseline steps = %d, want 5", sim.BaselineDecodeSteps)
	}
	if sim.SimulatedVerifyPasses != 1 {
		t.Fatalf("verify passes = %d, want 1", sim.SimulatedVerifyPasses)
	}
	if sim.StepReductionPercent != 80 {
		t.Fatalf("step reduction = %.2f, want 80", sim.StepReductionPercent)
	}
	if sim.Verdict != "viable_50_plus" {
		t.Fatalf("verdict = %q", sim.Verdict)
	}
}

func TestSimulateDraftHandlesNoOverlap(t *testing.T) {
	dir := t.TempDir()
	prompt := filepath.Join(dir, "prompt.txt")
	output := filepath.Join(dir, "output.txt")
	if err := os.WriteFile(prompt, []byte("alpha beta gamma"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(output, []byte("one two three"), 0644); err != nil {
		t.Fatal(err)
	}

	sim, err := SimulateDraft(DraftOptions{PromptPath: prompt, OutputPath: output, NGram: 2, MaxDraftTokens: 8})
	if err != nil {
		t.Fatal(err)
	}
	if sim.SimulatedVerifyPasses != sim.BaselineDecodeSteps {
		t.Fatalf("verify passes = %d, baseline = %d", sim.SimulatedVerifyPasses, sim.BaselineDecodeSteps)
	}
	if sim.Verdict != "not_viable" {
		t.Fatalf("verdict = %q", sim.Verdict)
	}
}
