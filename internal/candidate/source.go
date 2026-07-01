package candidate

import (
	"encoding/binary"
	"sort"
)

const (
	DefaultNGram           = 4
	DefaultMaxSuffixTokens = 32
	DefaultMaxDraftTokens  = 32
)

type Options struct {
	NGram           int
	MaxSuffixTokens int
	MaxDraftTokens  int
}

type Candidate struct {
	Tokens              []int32 `json:"tokens"`
	MatchedSuffixTokens int     `json:"matched_suffix_tokens"`
	SourceMatchStart    int     `json:"source_match_start"`
	SourceStart         int     `json:"source_start"`
	Score               float64 `json:"score"`
}

type Source interface {
	Reset(prompt []int32)
	Propose(current []int32, maxTokens int) []int32
	Observe(committed []int32)
}

type CorpusSource struct {
	opts    Options
	corpus  []int32
	index   map[string][]int
	history []int32
}

func NewCorpusSource(corpus []int32, opts Options) *CorpusSource {
	opts = normalizeOptions(opts)
	copied := clone(corpus)
	return &CorpusSource{
		opts:   opts,
		corpus: copied,
		index:  buildIndex(copied, opts.NGram),
	}
}

func (s *CorpusSource) Reset(prompt []int32) {
	s.history = clone(prompt)
}

func (s *CorpusSource) Observe(committed []int32) {
	s.history = append(s.history, committed...)
}

func (s *CorpusSource) Propose(current []int32, maxTokens int) []int32 {
	best, ok := s.Best(current, maxTokens)
	if !ok {
		return nil
	}
	return clone(best.Tokens)
}

func (s *CorpusSource) Best(current []int32, maxTokens int) (Candidate, bool) {
	candidates := s.Candidates(current, maxTokens, 1)
	if len(candidates) == 0 {
		return Candidate{}, false
	}
	return candidates[0], true
}

func (s *CorpusSource) Candidates(current []int32, maxTokens, limit int) []Candidate {
	if s == nil || limit <= 0 {
		return nil
	}

	maxDraft := s.opts.MaxDraftTokens
	if maxTokens > 0 && maxTokens < maxDraft {
		maxDraft = maxTokens
	}
	if len(s.corpus) == 0 || len(s.index) == 0 || maxDraft <= 0 {
		return nil
	}

	prefix := s.prefix(current)
	if len(prefix) < s.opts.NGram {
		return nil
	}

	longest := minInt(s.opts.MaxSuffixTokens, len(prefix))
	byDraft := map[string]Candidate{}
	for suffixLen := longest; suffixLen >= s.opts.NGram; suffixLen-- {
		suffix := prefix[len(prefix)-suffixLen:]
		needle := tokenKey(suffix[len(suffix)-s.opts.NGram:])
		for _, ngramPos := range s.index[needle] {
			matchStart := ngramPos - (suffixLen - s.opts.NGram)
			if matchStart < 0 || matchStart+suffixLen >= len(s.corpus) {
				continue
			}
			if !sameTokens(suffix, s.corpus[matchStart:matchStart+suffixLen]) {
				continue
			}

			sourceStart := matchStart + suffixLen
			sourceEnd := minInt(sourceStart+maxDraft, len(s.corpus))
			if sourceStart >= sourceEnd {
				continue
			}
			tokens := s.corpus[sourceStart:sourceEnd]
			candidate := Candidate{
				Tokens:              clone(tokens),
				MatchedSuffixTokens: suffixLen,
				SourceMatchStart:    matchStart,
				SourceStart:         sourceStart,
				Score:               float64(suffixLen) + float64(len(tokens))/100,
			}

			key := tokenKey(tokens)
			if existing, ok := byDraft[key]; ok && !better(candidate, existing) {
				continue
			}
			byDraft[key] = candidate
		}
		if len(byDraft) >= limit {
			break
		}
	}

	candidates := make([]Candidate, 0, len(byDraft))
	for _, candidate := range byDraft {
		candidates = append(candidates, candidate)
	}
	sort.Slice(candidates, func(i, j int) bool {
		return better(candidates[i], candidates[j])
	})
	if len(candidates) > limit {
		candidates = candidates[:limit]
	}
	return candidates
}

func normalizeOptions(opts Options) Options {
	if opts.NGram <= 0 {
		opts.NGram = DefaultNGram
	}
	if opts.MaxSuffixTokens <= 0 {
		opts.MaxSuffixTokens = DefaultMaxSuffixTokens
	}
	if opts.MaxDraftTokens <= 0 {
		opts.MaxDraftTokens = DefaultMaxDraftTokens
	}
	return opts
}

func buildIndex(tokens []int32, ngram int) map[string][]int {
	index := map[string][]int{}
	if ngram <= 0 {
		return index
	}
	for i := 0; i+ngram <= len(tokens); i++ {
		key := tokenKey(tokens[i : i+ngram])
		index[key] = append(index[key], i)
	}
	return index
}

func (s *CorpusSource) prefix(current []int32) []int32 {
	if len(current) == 0 {
		return s.history
	}
	prefix := make([]int32, 0, len(s.history)+len(current))
	prefix = append(prefix, s.history...)
	prefix = append(prefix, current...)
	return prefix
}

func better(left, right Candidate) bool {
	if left.Score != right.Score {
		return left.Score > right.Score
	}
	if left.MatchedSuffixTokens != right.MatchedSuffixTokens {
		return left.MatchedSuffixTokens > right.MatchedSuffixTokens
	}
	if len(left.Tokens) != len(right.Tokens) {
		return len(left.Tokens) > len(right.Tokens)
	}
	return left.SourceStart < right.SourceStart
}

func tokenKey(tokens []int32) string {
	buf := make([]byte, len(tokens)*4)
	for i, token := range tokens {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], uint32(token))
	}
	return string(buf)
}

func sameTokens(left, right []int32) bool {
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

func clone(tokens []int32) []int32 {
	if len(tokens) == 0 {
		return nil
	}
	copied := make([]int32, len(tokens))
	copy(copied, tokens)
	return copied
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
