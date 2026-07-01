package candidate

import (
	"reflect"
	"testing"
)

var _ Source = (*CorpusSource)(nil)

func TestCorpusSourceProposeUsesPendingCurrent(t *testing.T) {
	source := NewCorpusSource([]int32{10, 11, 12, 13, 14, 15}, Options{NGram: 3, MaxDraftTokens: 8})
	source.Reset([]int32{10, 11, 12})

	got := source.Propose([]int32{13}, 8)
	assertTokens(t, got, []int32{14, 15})
}

func TestCorpusSourceObserveCommittedTokens(t *testing.T) {
	source := NewCorpusSource([]int32{1, 2, 3, 4, 5, 6}, Options{NGram: 2, MaxDraftTokens: 8})
	source.Reset([]int32{1, 2})
	source.Observe([]int32{3, 4})

	got := source.Propose(nil, 2)
	assertTokens(t, got, []int32{5, 6})
}

func TestCorpusSourceCandidatesRankLongestSuffix(t *testing.T) {
	source := NewCorpusSource([]int32{
		7, 8, 3, 4, 99,
		1, 2, 3, 4, 5, 6,
	}, Options{NGram: 2, MaxDraftTokens: 2})
	source.Reset([]int32{1, 2, 3, 4})

	candidates := source.Candidates(nil, 2, 2)
	if len(candidates) != 2 {
		t.Fatalf("candidates = %d, want 2", len(candidates))
	}
	assertTokens(t, candidates[0].Tokens, []int32{5, 6})
	if candidates[0].MatchedSuffixTokens != 4 {
		t.Fatalf("matched suffix = %d, want 4", candidates[0].MatchedSuffixTokens)
	}
	assertTokens(t, candidates[1].Tokens, []int32{99, 1})
	if candidates[1].MatchedSuffixTokens != 2 {
		t.Fatalf("second matched suffix = %d, want 2", candidates[1].MatchedSuffixTokens)
	}
}

func TestCorpusSourceRespectsMaxTokensAndCopies(t *testing.T) {
	corpus := []int32{1, 2, 3, 4, 5, 6}
	source := NewCorpusSource(corpus, Options{NGram: 2, MaxDraftTokens: 8})
	corpus[4] = 99
	source.Reset([]int32{1, 2, 3})

	got := source.Propose([]int32{4}, 1)
	assertTokens(t, got, []int32{5})

	got[0] = 42
	got = source.Propose([]int32{4}, 2)
	assertTokens(t, got, []int32{5, 6})
}

func TestCorpusSourceNoCandidateAtCorpusEnd(t *testing.T) {
	source := NewCorpusSource([]int32{1, 2, 3}, Options{NGram: 2})
	source.Reset([]int32{1, 2, 3})

	if got := source.Propose(nil, 8); got != nil {
		t.Fatalf("proposal = %v, want nil", got)
	}
}

func assertTokens(t *testing.T, got, want []int32) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens = %v, want %v", got, want)
	}
}
