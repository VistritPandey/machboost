package ollama

import (
	"encoding/json"
	"testing"
)

func TestDecodeTags(t *testing.T) {
	tags, err := DecodeTags([]byte(`{"models":[{"name":"qwen3:8b","size":5225388164,"details":{"format":"gguf","parameter_size":"8.2B","quantization_level":"Q4_K_M","context_length":40960}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(tags.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(tags.Models))
	}
	if got := tags.Models[0].Details.QuantizationLevel; got != "Q4_K_M" {
		t.Fatalf("quantization = %q, want Q4_K_M", got)
	}
}

func TestTokensPerSecond(t *testing.T) {
	got := TokensPerSecond(GenerateResponse{EvalCount: 20, EvalDuration: 2_000_000_000})
	if got != 10 {
		t.Fatalf("tokens/sec = %f, want 10", got)
	}
}

func TestDecodeGenerate(t *testing.T) {
	resp, err := DecodeGenerate([]byte(`{"model":"qwen3:8b","done":true,"eval_count":32,"eval_duration":4000000000,"load_duration":1000000,"total_duration":5000000000}`))
	if err != nil {
		t.Fatal(err)
	}
	if resp.EvalCount != 32 {
		t.Fatalf("eval count = %d, want 32", resp.EvalCount)
	}
	if got := TokensPerSecond(resp); got != 8 {
		t.Fatalf("tokens/sec = %f, want 8", got)
	}
}

func TestGenerateRequestKeepAliveCanBeNumeric(t *testing.T) {
	data, err := json.Marshal(GenerateRequest{
		Model:     "qwen3:8b",
		Prompt:    "hi",
		Stream:    false,
		KeepAlive: -1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"model":"qwen3:8b","prompt":"hi","stream":false,"keep_alive":-1}` {
		t.Fatalf("json = %s", data)
	}
}
