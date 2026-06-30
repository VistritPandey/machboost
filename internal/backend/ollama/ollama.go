package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const DefaultEndpoint = "http://127.0.0.1:11434"

type Client struct {
	Endpoint string
	HTTP     *http.Client
}

type TagsResponse struct {
	Models []Model `json:"models"`
}

type Model struct {
	Name       string       `json:"name"`
	Model      string       `json:"model"`
	Size       int64        `json:"size"`
	Details    ModelDetails `json:"details"`
	ModifiedAt string       `json:"modified_at"`
}

type ModelDetails struct {
	Format            string `json:"format"`
	Family            string `json:"family"`
	ParameterSize     string `json:"parameter_size"`
	QuantizationLevel string `json:"quantization_level"`
	ContextLength     int64  `json:"context_length"`
}

type GenerateRequest struct {
	Model     string                 `json:"model"`
	Prompt    string                 `json:"prompt"`
	Stream    bool                   `json:"stream"`
	KeepAlive interface{}            `json:"keep_alive,omitempty"`
	Options   map[string]interface{} `json:"options,omitempty"`
}

type GenerateResponse struct {
	Model              string `json:"model"`
	Response           string `json:"response"`
	Done               bool   `json:"done"`
	TotalDuration      int64  `json:"total_duration"`
	LoadDuration       int64  `json:"load_duration"`
	PromptEvalCount    int64  `json:"prompt_eval_count"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalCount          int64  `json:"eval_count"`
	EvalDuration       int64  `json:"eval_duration"`
}

type BenchResult struct {
	SchemaVersion string   `json:"schema_version"`
	Backend       string   `json:"backend"`
	Model         string   `json:"model"`
	Prompt        string   `json:"prompt,omitempty"`
	EvalCount     int64    `json:"eval_count"`
	DurationMS    int64    `json:"duration_ms"`
	LoadMS        int64    `json:"load_ms"`
	TotalMS       int64    `json:"total_ms"`
	TokensPerSec  float64  `json:"tokens_per_second"`
	Warnings      []string `json:"warnings,omitempty"`
}

func NewClient(endpoint string) Client {
	if endpoint == "" {
		endpoint = EndpointFromEnv()
	}
	return Client{
		Endpoint: strings.TrimRight(endpoint, "/"),
		HTTP:     &http.Client{Timeout: 120 * time.Second},
	}
}

func EndpointFromEnv() string {
	if host := os.Getenv("OLLAMA_HOST"); host != "" {
		if strings.HasPrefix(host, "http://") || strings.HasPrefix(host, "https://") {
			return host
		}
		return "http://" + host
	}
	return DefaultEndpoint
}

func DecodeTags(data []byte) (TagsResponse, error) {
	var tags TagsResponse
	err := json.Unmarshal(data, &tags)
	return tags, err
}

func DecodeGenerate(data []byte) (GenerateResponse, error) {
	var response GenerateResponse
	err := json.Unmarshal(data, &response)
	return response, err
}

func TokensPerSecond(response GenerateResponse) float64 {
	if response.EvalCount <= 0 || response.EvalDuration <= 0 {
		return 0
	}
	return float64(response.EvalCount) / (float64(response.EvalDuration) / float64(time.Second))
}

func (c Client) Tags(ctx context.Context) (TagsResponse, error) {
	body, err := c.get(ctx, "/api/tags")
	if err != nil {
		return TagsResponse{}, err
	}
	return DecodeTags(body)
}

func (c Client) Generate(ctx context.Context, req GenerateRequest) (GenerateResponse, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return GenerateResponse{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint+"/api/generate", bytes.NewReader(data))
	if err != nil {
		return GenerateResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpClient := c.HTTP
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 120 * time.Second}
	}
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return GenerateResponse{}, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return GenerateResponse{}, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return GenerateResponse{}, fmt.Errorf("ollama returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	return DecodeGenerate(body)
}

func (c Client) Bench(ctx context.Context, model, prompt string, tokens, ctxLen int) (BenchResult, error) {
	if model == "" {
		model = "qwen3:8b"
	}
	if prompt == "" {
		prompt = "Write one concise sentence about local workload optimization."
	}
	if tokens <= 0 {
		tokens = 64
	}
	if ctxLen <= 0 {
		ctxLen = 4096
	}

	start := time.Now()
	response, err := c.Generate(ctx, GenerateRequest{
		Model:     model,
		Prompt:    prompt,
		Stream:    false,
		KeepAlive: -1,
		Options: map[string]interface{}{
			"num_predict": tokens,
			"num_ctx":     ctxLen,
		},
	})
	duration := time.Since(start)
	if err != nil {
		return BenchResult{}, err
	}

	warnings := []string{}
	tps := TokensPerSecond(response)
	if tps == 0 {
		warnings = append(warnings, "Ollama did not return eval timing fields; tokens/sec is unavailable.")
	}

	return BenchResult{
		SchemaVersion: "machboost.bench.v1",
		Backend:       "ollama",
		Model:         model,
		Prompt:        prompt,
		EvalCount:     response.EvalCount,
		DurationMS:    duration.Milliseconds(),
		LoadMS:        response.LoadDuration / int64(time.Millisecond),
		TotalMS:       response.TotalDuration / int64(time.Millisecond),
		TokensPerSec:  tps,
		Warnings:      warnings,
	}, nil
}

func (c Client) get(ctx context.Context, path string) ([]byte, error) {
	httpClient := c.HTTP
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 5 * time.Second}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.Endpoint+path, nil)
	if err != nil {
		return nil, err
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("ollama returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	return body, nil
}
