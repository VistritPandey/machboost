package overlap

import "testing"

func TestCandidatesFromTokens(t *testing.T) {
	prefix := Tokenize("func target ( ) string { return")
	context := Tokenize("func target ( ) string { return helper ( ) }")

	candidates := candidatesFromTokens(prefix, context, 3, 8, 3)
	if len(candidates) == 0 {
		t.Fatal("expected at least one candidate")
	}
	if candidates[0].DraftText != "helper ( ) }" {
		t.Fatalf("draft = %q", candidates[0].DraftText)
	}
	if candidates[0].MatchedSuffixTokens < 3 {
		t.Fatalf("matched suffix = %d, want >= 3", candidates[0].MatchedSuffixTokens)
	}
}

func TestCandidatesFromTokensNoMatch(t *testing.T) {
	candidates := candidatesFromTokens(Tokenize("alpha beta"), Tokenize("gamma delta"), 2, 8, 3)
	if len(candidates) != 0 {
		t.Fatalf("candidates = %#v, want none", candidates)
	}
}
