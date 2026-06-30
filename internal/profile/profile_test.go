package profile

import "testing"

func TestResolveSustainedBuild(t *testing.T) {
	prof, err := Resolve("sustained", "build", nil, 10)
	if err != nil {
		t.Fatal(err)
	}
	if !prof.KeepAwake {
		t.Fatal("expected sustained profile to keep the machine awake")
	}
	if got := prof.Env["MAKEFLAGS"]; got != "-j10" {
		t.Fatalf("MAKEFLAGS = %q, want -j10", got)
	}
	if got := prof.Env["GOMAXPROCS"]; got != "10" {
		t.Fatalf("GOMAXPROCS = %q, want 10", got)
	}
}

func TestResolveQuietRenderUsesHalfThreads(t *testing.T) {
	prof, err := Resolve("quiet", "render", nil, 10)
	if err != nil {
		t.Fatal(err)
	}
	if prof.KeepAwake {
		t.Fatal("quiet profile should not keep awake")
	}
	if got := prof.Env["OMP_NUM_THREADS"]; got != "5" {
		t.Fatalf("OMP_NUM_THREADS = %q, want 5", got)
	}
}

func TestResolveAppliesConfigOverride(t *testing.T) {
	keepAwake := false
	cfg := &Config{Profiles: map[string]Override{
		"sustained": {
			KeepAwake: &keepAwake,
			Env:       map[string]string{"CUSTOM": "yes"},
		},
	}}

	prof, err := Resolve("sustained", "generic", cfg, 8)
	if err != nil {
		t.Fatal(err)
	}
	if prof.KeepAwake {
		t.Fatal("expected config to disable keep_awake")
	}
	if got := prof.Env["CUSTOM"]; got != "yes" {
		t.Fatalf("CUSTOM = %q, want yes", got)
	}
}

func TestResolveRejectsUnknownWorkload(t *testing.T) {
	if _, err := Resolve("sustained", "gaming", nil, 8); err == nil {
		t.Fatal("expected unknown workload error")
	}
}
