package overlap

import (
	"encoding/json"
	"os"
)

type DraftSimulation struct {
	SchemaVersion         string  `json:"schema_version"`
	NGram                 int     `json:"ngram"`
	MaxDraftTokens        int     `json:"max_draft_tokens"`
	PromptTokens          int     `json:"prompt_tokens"`
	ContextTokens         int     `json:"context_tokens"`
	SourceTokens          int     `json:"source_tokens"`
	OutputTokens          int     `json:"output_tokens"`
	BaselineDecodeSteps   int     `json:"baseline_decode_steps"`
	SimulatedVerifyPasses int     `json:"simulated_verify_passes"`
	AcceptedDraftTokens   int     `json:"accepted_draft_tokens"`
	AcceptedDraftSpans    int     `json:"accepted_draft_spans"`
	NormalDecodeTokens    int     `json:"normal_decode_tokens"`
	LongestAcceptedDraft  int     `json:"longest_accepted_draft"`
	StepReductionPercent  float64 `json:"step_reduction_percent"`
	EstimatedSpeedup      float64 `json:"estimated_speedup"`
	Verdict               string  `json:"verdict"`
	ContextFilesScanned   int     `json:"context_files_scanned,omitempty"`
	Idealized             bool    `json:"idealized"`
}

type DraftOptions struct {
	PromptPath     string
	OutputPath     string
	ContextPath    string
	NGram          int
	MaxDraftTokens int
}

func SimulateDraft(opts DraftOptions) (DraftSimulation, error) {
	if opts.NGram <= 0 {
		opts.NGram = 4
	}
	if opts.MaxDraftTokens <= 0 {
		opts.MaxDraftTokens = 32
	}

	promptBytes, err := os.ReadFile(opts.PromptPath)
	if err != nil {
		return DraftSimulation{}, err
	}
	outputBytes, err := os.ReadFile(opts.OutputPath)
	if err != nil {
		return DraftSimulation{}, err
	}

	promptTokens := Tokenize(string(promptBytes))
	outputTokens := Tokenize(string(outputBytes))
	contextTokens := []string{}
	contextFiles := 0
	if opts.ContextPath != "" {
		contextTokens, contextFiles, err = readContextTokens(opts.ContextPath)
		if err != nil {
			return DraftSimulation{}, err
		}
	}

	sourceTokens := append(append([]string{}, promptTokens...), contextTokens...)
	sim := simulateFromTokens(sourceTokens, outputTokens, opts.NGram, opts.MaxDraftTokens)
	sim.SchemaVersion = "machboost.draft_sim.v1"
	sim.NGram = opts.NGram
	sim.MaxDraftTokens = opts.MaxDraftTokens
	sim.PromptTokens = len(promptTokens)
	sim.ContextTokens = len(contextTokens)
	sim.SourceTokens = len(sourceTokens)
	sim.OutputTokens = len(outputTokens)
	sim.BaselineDecodeSteps = len(outputTokens)
	sim.ContextFilesScanned = contextFiles
	sim.Idealized = true
	return sim, nil
}

func MarshalDraftSimulation(sim DraftSimulation) ([]byte, error) {
	return json.MarshalIndent(sim, "", "  ")
}

func simulateFromTokens(source, output []string, ngram, maxDraft int) DraftSimulation {
	sim := DraftSimulation{}
	if len(output) == 0 {
		sim.Verdict = "no_output"
		return sim
	}
	if len(source) == 0 || ngram <= 0 {
		sim.NormalDecodeTokens = len(output)
		sim.SimulatedVerifyPasses = len(output)
		sim.EstimatedSpeedup = 1
		sim.Verdict = "not_viable"
		return sim
	}

	index := map[string][]int{}
	for i := 0; i+ngram <= len(source); i++ {
		index[key(source[i:i+ngram])] = append(index[key(source[i:i+ngram])], i)
	}

	for i := 0; i < len(output); {
		best := 0
		if i+ngram <= len(output) {
			for _, sourcePos := range index[key(output[i:i+ngram])] {
				length := 0
				for sourcePos+length < len(source) && i+length < len(output) && source[sourcePos+length] == output[i+length] {
					length++
				}
				if length > best {
					best = length
				}
			}
		}

		if best >= ngram {
			if best > maxDraft {
				best = maxDraft
			}
			sim.AcceptedDraftTokens += best
			sim.AcceptedDraftSpans++
			sim.SimulatedVerifyPasses++
			if best > sim.LongestAcceptedDraft {
				sim.LongestAcceptedDraft = best
			}
			i += best
			continue
		}

		sim.NormalDecodeTokens++
		sim.SimulatedVerifyPasses++
		i++
	}

	if len(output) > 0 {
		sim.StepReductionPercent = percent(len(output)-sim.SimulatedVerifyPasses, len(output))
	}
	if sim.SimulatedVerifyPasses > 0 {
		sim.EstimatedSpeedup = float64(len(output)) / float64(sim.SimulatedVerifyPasses)
	}
	sim.Verdict = draftVerdict(sim.StepReductionPercent)
	return sim
}

func draftVerdict(reduction float64) string {
	if reduction >= 50 {
		return "viable_50_plus"
	}
	if reduction >= 20 {
		return "viable_20_50"
	}
	if reduction >= 10 {
		return "weak_signal"
	}
	return "not_viable"
}
