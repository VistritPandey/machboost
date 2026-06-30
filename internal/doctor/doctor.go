package doctor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	goruntime "runtime"
	"strconv"
	"strings"
	"time"

	"machboost/internal/backend/ollama"
)

type Report struct {
	SchemaVersion string    `json:"schema_version"`
	GeneratedAt   time.Time `json:"generated_at"`
	OS            OSInfo    `json:"os"`
	Machine       Machine   `json:"machine"`
	Runtimes      []Runtime `json:"runtimes"`
	Warnings      []string  `json:"warnings,omitempty"`
}

type OSInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Build   string `json:"build"`
	Arch    string `json:"arch"`
}

type Machine struct {
	CPUCount    int    `json:"cpu_count"`
	CPUBrand    string `json:"cpu_brand,omitempty"`
	MemoryBytes int64  `json:"memory_bytes"`
	SwapUsage   string `json:"swap_usage,omitempty"`
}

type Runtime struct {
	Name      string   `json:"name"`
	Available bool     `json:"available"`
	Path      string   `json:"path,omitempty"`
	Version   string   `json:"version,omitempty"`
	Details   []string `json:"details,omitempty"`
}

type Commander interface {
	Run(ctx context.Context, name string, args ...string) (string, error)
	LookPath(file string) (string, error)
}

type RealCommander struct{}

func (RealCommander) Run(ctx context.Context, name string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err != nil && stderr.Len() > 0 {
		return stdout.String(), fmt.Errorf("%w: %s", err, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), err
}

func (RealCommander) LookPath(file string) (string, error) {
	return exec.LookPath(file)
}

func Generate(ctx context.Context) Report {
	return GenerateWithCommander(ctx, RealCommander{})
}

func GenerateWithCommander(ctx context.Context, commander Commander) Report {
	report := Report{
		SchemaVersion: "machboost.doctor.v1",
		GeneratedAt:   time.Now().UTC(),
		OS: OSInfo{
			Name: "macOS",
			Arch: goruntime.GOARCH,
		},
		Machine: Machine{CPUCount: goruntime.NumCPU()},
	}

	if goruntime.GOOS != "darwin" {
		report.OS.Name = goruntime.GOOS
		report.Warnings = append(report.Warnings, "machboost v1 is Mac-first; non-macOS diagnostics are limited.")
	}

	if out, err := commander.Run(ctx, "sw_vers"); err == nil {
		report.OS = parseSwVers(out, report.OS)
	}
	if out, err := commander.Run(ctx, "sysctl", "-n", "hw.ncpu"); err == nil {
		if value, parseErr := strconv.Atoi(strings.TrimSpace(out)); parseErr == nil {
			report.Machine.CPUCount = value
		}
	}
	if out, err := commander.Run(ctx, "sysctl", "-n", "hw.memsize"); err == nil {
		if value, parseErr := strconv.ParseInt(strings.TrimSpace(out), 10, 64); parseErr == nil {
			report.Machine.MemoryBytes = value
		}
	}
	if out, err := commander.Run(ctx, "sysctl", "-n", "machdep.cpu.brand_string"); err == nil {
		report.Machine.CPUBrand = strings.TrimSpace(out)
	}
	if out, err := commander.Run(ctx, "sysctl", "vm.swapusage"); err == nil {
		report.Machine.SwapUsage = strings.TrimSpace(out)
		if strings.Contains(out, "used = 0.00M") {
			// All clear; keep report compact.
		} else if strings.Contains(out, "used =") {
			report.Warnings = append(report.Warnings, "swap appears to be in use; heavy workloads may slow down when memory pressure is high.")
		}
	}

	report.Runtimes = append(report.Runtimes, detectOllama(ctx, commander))
	report.Runtimes = append(report.Runtimes, detectCommandRuntime(commander, "llama.cpp", []string{"llama-bench", "llama-cli", "llama-server"}))
	report.Runtimes = append(report.Runtimes, detectCommandRuntime(commander, "docker", []string{"docker"}))

	return report
}

func MarshalJSON(report Report) ([]byte, error) {
	return json.MarshalIndent(report, "", "  ")
}

func parseSwVers(out string, base OSInfo) OSInfo {
	for _, line := range strings.Split(out, "\n") {
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		switch key {
		case "ProductName":
			base.Name = value
		case "ProductVersion":
			base.Version = value
		case "BuildVersion":
			base.Build = value
		}
	}
	return base
}

func detectOllama(ctx context.Context, commander Commander) Runtime {
	runtime := Runtime{Name: "ollama"}
	path, err := commander.LookPath("ollama")
	if err != nil {
		return runtime
	}
	runtime.Available = true
	runtime.Path = path
	if out, err := commander.Run(ctx, "ollama", "--version"); err == nil {
		runtime.Version = strings.TrimSpace(out)
	}

	httpClient := &http.Client{Timeout: 2 * time.Second}
	client := ollama.Client{Endpoint: ollama.EndpointFromEnv(), HTTP: httpClient}
	tags, err := client.Tags(ctx)
	if err != nil {
		runtime.Details = append(runtime.Details, "API unavailable: "+err.Error())
		return runtime
	}
	runtime.Details = append(runtime.Details, fmt.Sprintf("%d installed model(s)", len(tags.Models)))
	for i, model := range tags.Models {
		if i >= 5 {
			runtime.Details = append(runtime.Details, fmt.Sprintf("+%d more model(s)", len(tags.Models)-i))
			break
		}
		detail := model.Name
		if model.Details.ParameterSize != "" || model.Details.QuantizationLevel != "" {
			detail += " (" + strings.TrimSpace(model.Details.ParameterSize+" "+model.Details.QuantizationLevel) + ")"
		}
		runtime.Details = append(runtime.Details, detail)
	}
	return runtime
}

func detectCommandRuntime(commander Commander, name string, binaries []string) Runtime {
	runtime := Runtime{Name: name}
	for _, binary := range binaries {
		path, err := commander.LookPath(binary)
		if err != nil {
			continue
		}
		runtime.Available = true
		runtime.Path = path
		if binary != name {
			runtime.Details = append(runtime.Details, "detected binary: "+binary)
		}
		return runtime
	}
	return runtime
}
