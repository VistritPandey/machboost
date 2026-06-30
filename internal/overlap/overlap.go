package overlap

import (
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

type Report struct {
	SchemaVersion       string  `json:"schema_version"`
	NGram               int     `json:"ngram"`
	PromptTokens        int     `json:"prompt_tokens"`
	ContextTokens       int     `json:"context_tokens"`
	SourceTokens        int     `json:"source_tokens"`
	OutputTokens        int     `json:"output_tokens"`
	CoveredTokens       int     `json:"covered_tokens"`
	CoveragePercent     float64 `json:"coverage_percent"`
	LongestCopiedSpan   int     `json:"longest_copied_span"`
	EstimatedStepSave   float64 `json:"estimated_step_save_percent"`
	Verdict             string  `json:"verdict"`
	ContextFilesScanned int     `json:"context_files_scanned,omitempty"`
}

type Options struct {
	PromptPath  string
	OutputPath  string
	ContextPath string
	NGram       int
}

func Analyze(opts Options) (Report, error) {
	if opts.NGram <= 0 {
		opts.NGram = 4
	}

	promptBytes, err := os.ReadFile(opts.PromptPath)
	if err != nil {
		return Report{}, err
	}
	outputBytes, err := os.ReadFile(opts.OutputPath)
	if err != nil {
		return Report{}, err
	}

	promptTokens := Tokenize(string(promptBytes))
	outputTokens := Tokenize(string(outputBytes))
	contextTokens := []string{}
	contextFiles := 0
	if opts.ContextPath != "" {
		contextTokens, contextFiles, err = readContextTokens(opts.ContextPath)
		if err != nil {
			return Report{}, err
		}
	}

	sourceTokens := append(append([]string{}, promptTokens...), contextTokens...)
	covered, longest := greedyCoverage(sourceTokens, outputTokens, opts.NGram)
	coverage := percent(covered, len(outputTokens))

	return Report{
		SchemaVersion:       "machboost.overlap.v1",
		NGram:               opts.NGram,
		PromptTokens:        len(promptTokens),
		ContextTokens:       len(contextTokens),
		SourceTokens:        len(sourceTokens),
		OutputTokens:        len(outputTokens),
		CoveredTokens:       covered,
		CoveragePercent:     coverage,
		LongestCopiedSpan:   longest,
		EstimatedStepSave:   coverage,
		Verdict:             verdict(coverage),
		ContextFilesScanned: contextFiles,
	}, nil
}

func MarshalJSON(report Report) ([]byte, error) {
	return json.MarshalIndent(report, "", "  ")
}

func Tokenize(input string) []string {
	var tokens []string
	var current strings.Builder
	flush := func() {
		if current.Len() == 0 {
			return
		}
		tokens = append(tokens, strings.ToLower(current.String()))
		current.Reset()
	}

	for _, r := range input {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			current.WriteRune(unicode.ToLower(r))
			continue
		}
		flush()
		if !unicode.IsSpace(r) {
			tokens = append(tokens, string(r))
		}
	}
	flush()
	return tokens
}

func greedyCoverage(source, output []string, ngram int) (int, int) {
	if len(source) == 0 || len(output) == 0 || ngram <= 0 {
		return 0, 0
	}

	index := map[string][]int{}
	for i := 0; i+ngram <= len(source); i++ {
		key := key(source[i : i+ngram])
		index[key] = append(index[key], i)
	}

	covered := 0
	longest := 0
	for i := 0; i < len(output); {
		if i+ngram > len(output) {
			i++
			continue
		}
		candidates := index[key(output[i:i+ngram])]
		best := 0
		for _, sourcePos := range candidates {
			length := 0
			for sourcePos+length < len(source) && i+length < len(output) && source[sourcePos+length] == output[i+length] {
				length++
			}
			if length > best {
				best = length
			}
		}
		if best >= ngram {
			covered += best
			if best > longest {
				longest = best
			}
			i += best
			continue
		}
		i++
	}
	return covered, longest
}

func readContextTokens(path string) ([]string, int, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, 0, err
	}
	if !info.IsDir() {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, 0, err
		}
		return Tokenize(string(data)), 1, nil
	}

	tokens := []string{}
	files := 0
	err = filepath.WalkDir(path, func(filePath string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		name := d.Name()
		if d.IsDir() {
			if shouldSkipDir(name) {
				return filepath.SkipDir
			}
			return nil
		}
		if shouldSkipFile(name) {
			return nil
		}
		data, err := os.ReadFile(filePath)
		if err != nil {
			return nil
		}
		tokens = append(tokens, Tokenize(string(data))...)
		files++
		return nil
	})
	return tokens, files, err
}

func shouldSkipDir(name string) bool {
	switch name {
	case ".git", "node_modules", "vendor", "target", "dist", "build", ".cache":
		return true
	default:
		return false
	}
}

func shouldSkipFile(name string) bool {
	lower := strings.ToLower(name)
	if strings.HasPrefix(lower, ".") {
		return true
	}
	skipped := []string{".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tar", ".mp4", ".mov", ".bin", ".dylib", ".so", ".a"}
	for _, suffix := range skipped {
		if strings.HasSuffix(lower, suffix) {
			return true
		}
	}
	return false
}

func key(tokens []string) string {
	return strings.Join(tokens, "\x00")
}

func percent(part, total int) float64 {
	if total == 0 {
		return 0
	}
	return (float64(part) / float64(total)) * 100
}

func verdict(coverage float64) string {
	if coverage >= 50 {
		return "strong_candidate"
	}
	if coverage >= 30 {
		return "candidate"
	}
	if coverage >= 15 {
		return "weak_candidate"
	}
	return "poor_candidate"
}
