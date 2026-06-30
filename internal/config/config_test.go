package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadMissingConfig(t *testing.T) {
	cfg, found, err := Load(filepath.Join(t.TempDir(), ".machboost.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if found {
		t.Fatal("expected missing config to report found=false")
	}
	if cfg != nil {
		t.Fatal("expected nil config for missing file")
	}
}

func TestLoadConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".machboost.yaml")
	data := []byte(`profiles:
  custom:
    keep_awake: true
    env:
      FOO: bar
`)
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}

	cfg, found, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatal("expected config to be found")
	}
	override := cfg.Profiles["custom"]
	if override.KeepAwake == nil || !*override.KeepAwake {
		t.Fatal("expected keep_awake=true")
	}
	if got := override.Env["FOO"]; got != "bar" {
		t.Fatalf("FOO = %q, want bar", got)
	}
}

func TestInitRefusesExistingConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".machboost.yaml")
	if err := os.WriteFile(path, []byte("profiles: {}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := Init(path); err == nil {
		t.Fatal("expected Init to refuse existing config")
	}
}
