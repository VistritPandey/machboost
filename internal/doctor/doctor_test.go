package doctor

import (
	"context"
	"errors"
	"testing"
)

type fakeCommander struct {
	outputs map[string]string
	paths   map[string]string
}

func (f fakeCommander) Run(ctx context.Context, name string, args ...string) (string, error) {
	key := name
	for _, arg := range args {
		key += " " + arg
	}
	if out, ok := f.outputs[key]; ok {
		return out, nil
	}
	return "", errors.New("missing command output: " + key)
}

func (f fakeCommander) LookPath(file string) (string, error) {
	if path, ok := f.paths[file]; ok {
		return path, nil
	}
	return "", errors.New("not found")
}

func TestParseSwVers(t *testing.T) {
	info := parseSwVers("ProductName:\tmacOS\nProductVersion:\t26.5.1\nBuildVersion:\t25F80\n", OSInfo{Arch: "arm64"})
	if info.Name != "macOS" || info.Version != "26.5.1" || info.Build != "25F80" {
		t.Fatalf("unexpected OS info: %+v", info)
	}
	if info.Arch != "arm64" {
		t.Fatalf("arch = %q, want arm64", info.Arch)
	}
}

func TestGenerateWithCommanderDetectsRuntimes(t *testing.T) {
	report := GenerateWithCommander(context.Background(), fakeCommander{
		outputs: map[string]string{
			"sw_vers":                            "ProductName:\tmacOS\nProductVersion:\t26.5.1\nBuildVersion:\t25F80\n",
			"sysctl -n hw.ncpu":                  "10\n",
			"sysctl -n hw.memsize":               "34359738368\n",
			"sysctl -n machdep.cpu.brand_string": "Apple M1 Max\n",
			"sysctl vm.swapusage":                "vm.swapusage: total = 2048.00M  used = 0.00M  free = 2048.00M  (encrypted)\n",
			"ollama --version":                   "ollama version is 0.30.11\n",
		},
		paths: map[string]string{
			"ollama": "/usr/local/bin/ollama",
			"docker": "/usr/local/bin/docker",
		},
	})
	if report.Machine.CPUCount != 10 {
		t.Fatalf("cpu count = %d, want 10", report.Machine.CPUCount)
	}
	if report.Machine.MemoryBytes != 34359738368 {
		t.Fatalf("memory bytes = %d", report.Machine.MemoryBytes)
	}
	if len(report.Runtimes) != 3 {
		t.Fatalf("runtimes = %d, want 3", len(report.Runtimes))
	}
	if !report.Runtimes[0].Available {
		t.Fatal("expected ollama to be available")
	}
}
