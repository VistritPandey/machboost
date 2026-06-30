package profile

import (
	"fmt"
	"strconv"
)

const (
	DefaultProfile  = "sustained"
	DefaultWorkload = "generic"
)

type Config struct {
	Profiles map[string]Override `yaml:"profiles"`
}

type Override struct {
	KeepAwake *bool             `yaml:"keep_awake"`
	Env       map[string]string `yaml:"env"`
}

type Profile struct {
	Name      string            `json:"name"`
	Workload  string            `json:"workload"`
	KeepAwake bool              `json:"keep_awake"`
	Env       map[string]string `json:"env"`
	Warnings  []string          `json:"warnings,omitempty"`
}

func Resolve(name, workload string, cfg *Config, cpuCount int) (Profile, error) {
	if name == "" {
		name = DefaultProfile
	}
	if workload == "" {
		workload = DefaultWorkload
	}
	if cpuCount < 1 {
		cpuCount = 1
	}

	prof, ok := builtInProfile(name)
	if !ok {
		prof = Profile{Name: name, Env: map[string]string{}}
	}
	prof.Workload = workload
	prof.Env = cloneEnv(prof.Env)

	if err := applyWorkload(&prof, workload, cpuCount); err != nil {
		return Profile{}, err
	}

	if cfg != nil && cfg.Profiles != nil {
		if override, ok := cfg.Profiles[name]; ok {
			if override.KeepAwake != nil {
				prof.KeepAwake = *override.KeepAwake
			}
			for key, value := range override.Env {
				prof.Env[key] = value
			}
		}
	}

	return prof, nil
}

func builtInProfile(name string) (Profile, bool) {
	switch name {
	case "sustained":
		return Profile{
			Name:      "sustained",
			KeepAwake: true,
			Env:       map[string]string{"MACHBOOST_PROFILE": "sustained"},
		}, true
	case "balanced":
		return Profile{
			Name:      "balanced",
			KeepAwake: false,
			Env:       map[string]string{"MACHBOOST_PROFILE": "balanced"},
		}, true
	case "quiet":
		return Profile{
			Name:      "quiet",
			KeepAwake: false,
			Env:       map[string]string{"MACHBOOST_PROFILE": "quiet"},
		}, true
	default:
		return Profile{}, false
	}
}

func applyWorkload(prof *Profile, workload string, cpuCount int) error {
	switch workload {
	case "generic":
		return nil
	case "llm":
		if prof.Name == "sustained" {
			prof.Env["OLLAMA_KEEP_ALIVE"] = "-1"
			prof.Env["OLLAMA_FLASH_ATTENTION"] = "1"
			prof.Warnings = append(prof.Warnings, "Ollama environment hints only affect Ollama processes launched by machboost.")
		}
		return nil
	case "build":
		jobs := jobsFor(prof.Name, cpuCount)
		prof.Env["MAKEFLAGS"] = "-j" + strconv.Itoa(jobs)
		prof.Env["CMAKE_BUILD_PARALLEL_LEVEL"] = strconv.Itoa(jobs)
		prof.Env["GOMAXPROCS"] = strconv.Itoa(jobs)
		prof.Env["CARGO_BUILD_JOBS"] = strconv.Itoa(jobs)
		return nil
	case "render":
		jobs := jobsFor(prof.Name, cpuCount)
		prof.Env["OMP_NUM_THREADS"] = strconv.Itoa(jobs)
		prof.Env["VECLIB_MAXIMUM_THREADS"] = strconv.Itoa(jobs)
		prof.Env["OPENBLAS_NUM_THREADS"] = strconv.Itoa(jobs)
		return nil
	default:
		return fmt.Errorf("unknown workload %q", workload)
	}
}

func jobsFor(profileName string, cpuCount int) int {
	switch profileName {
	case "quiet":
		return maxInt(1, cpuCount/2)
	case "balanced":
		return maxInt(1, (cpuCount*3)/4)
	default:
		return cpuCount
	}
}

func cloneEnv(src map[string]string) map[string]string {
	dst := map[string]string{}
	for key, value := range src {
		dst[key] = value
	}
	return dst
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
