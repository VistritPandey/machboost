package overlap

import (
	"encoding/json"
	"os"
	"sort"
	"strings"
)

type CandidateOptions struct {
	PrefixPath     string
	ContextPath    string
	NGram          int
	MaxDraftTokens int
	Limit          int
}

type CandidateReport struct {
	SchemaVersion       string      `json:"schema_version"`
	NGram               int         `json:"ngram"`
	MaxDraftTokens      int         `json:"max_draft_tokens"`
	PrefixTokens        int         `json:"prefix_tokens"`
	ContextTokens       int         `json:"context_tokens"`
	ContextFilesScanned int         `json:"context_files_scanned,omitempty"`
	Candidates          []Candidate `json:"candidates"`
	Verdict             string      `json:"verdict"`
}

type Candidate struct {
	DraftText           string  `json:"draft_text"`
	TokenCount          int     `json:"token_count"`
	MatchedSuffix       string  `json:"matched_suffix"`
	MatchedSuffixTokens int     `json:"matched_suffix_tokens"`
	ContextStartToken   int     `json:"context_start_token"`
	Score               float64 `json:"score"`
}

func DraftCandidates(opts CandidateOptions) (CandidateReport, error) {
	if opts.NGram <= 0 {
		opts.NGram = 4
	}
	if opts.MaxDraftTokens <= 0 {
		opts.MaxDraftTokens = 32
	}
	if opts.Limit <= 0 {
		opts.Limit = 5
	}

	prefixBytes, err := os.ReadFile(opts.PrefixPath)
	if err != nil {
		return CandidateReport{}, err
	}
	contextTokens, contextFiles, err := readContextTokens(opts.ContextPath)
	if err != nil {
		return CandidateReport{}, err
	}
	prefixTokens := Tokenize(string(prefixBytes))

	candidates := candidatesFromTokens(prefixTokens, contextTokens, opts.NGram, opts.MaxDraftTokens, opts.Limit)
	report := CandidateReport{
		SchemaVersion:       "machboost.draft.v1",
		NGram:               opts.NGram,
		MaxDraftTokens:      opts.MaxDraftTokens,
		PrefixTokens:        len(prefixTokens),
		ContextTokens:       len(contextTokens),
		ContextFilesScanned: contextFiles,
		Candidates:          candidates,
		Verdict:             candidateVerdict(candidates),
	}
	return report, nil
}

func MarshalCandidateReport(report CandidateReport) ([]byte, error) {
	return json.MarshalIndent(report, "", "  ")
}

func candidatesFromTokens(prefixTokens, contextTokens []string, ngram, maxDraft, limit int) []Candidate {
	if len(prefixTokens) == 0 || len(contextTokens) == 0 || ngram <= 0 || maxDraft <= 0 {
		return nil
	}

	longest := minInt(16, len(prefixTokens))
	candidateByKey := map[string]Candidate{}
	for suffixLen := longest; suffixLen >= ngram; suffixLen-- {
		suffix := prefixTokens[len(prefixTokens)-suffixLen:]
		for i := 0; i+suffixLen < len(contextTokens); i++ {
			if !sameTokens(suffix, contextTokens[i:i+suffixLen]) {
				continue
			}
			start := i + suffixLen
			end := minInt(start+maxDraft, len(contextTokens))
			if start >= end {
				continue
			}
			draftTokens := contextTokens[start:end]
			draftText := strings.Join(draftTokens, " ")
			key := draftText + "|" + strings.Join(suffix, "\x00")
			score := float64(suffixLen) + float64(len(draftTokens))/100
			existing, ok := candidateByKey[key]
			if ok && existing.Score >= score {
				continue
			}
			candidateByKey[key] = Candidate{
				DraftText:           draftText,
				TokenCount:          len(draftTokens),
				MatchedSuffix:       strings.Join(suffix, " "),
				MatchedSuffixTokens: suffixLen,
				ContextStartToken:   start,
				Score:               score,
			}
		}
		if len(candidateByKey) >= limit {
			break
		}
	}

	candidates := make([]Candidate, 0, len(candidateByKey))
	for _, candidate := range candidateByKey {
		candidates = append(candidates, candidate)
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Score == candidates[j].Score {
			return candidates[i].TokenCount > candidates[j].TokenCount
		}
		return candidates[i].Score > candidates[j].Score
	})
	if len(candidates) > limit {
		candidates = candidates[:limit]
	}
	return candidates
}

func candidateVerdict(candidates []Candidate) string {
	if len(candidates) == 0 {
		return "no_candidate"
	}
	if candidates[0].MatchedSuffixTokens >= 8 {
		return "strong_candidate"
	}
	if candidates[0].MatchedSuffixTokens >= 4 {
		return "candidate"
	}
	return "weak_candidate"
}

func sameTokens(left, right []string) bool {
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

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
