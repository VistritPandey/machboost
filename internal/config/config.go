package config

import (
	"errors"
	"io/ioutil"
	"os"

	"gopkg.in/yaml.v3"

	"machboost/internal/profile"
)

const DefaultPath = ".machboost.yaml"

func Load(path string) (*profile.Config, bool, error) {
	if path == "" {
		path = DefaultPath
	}
	data, err := ioutil.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, false, nil
		}
		return nil, false, err
	}

	var cfg profile.Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, true, err
	}
	if cfg.Profiles == nil {
		cfg.Profiles = map[string]profile.Override{}
	}
	return &cfg, true, nil
}

func Init(path string) error {
	if path == "" {
		path = DefaultPath
	}
	if _, err := os.Stat(path); err == nil {
		return errors.New(path + " already exists")
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return ioutil.WriteFile(path, []byte(Sample()), 0644)
}

func Sample() string {
	return `# machboost profiles are safe, local-only command launch hints.
profiles:
  sustained:
    keep_awake: true
    env:
      MACHBOOST_PROFILE: sustained

  balanced:
    keep_awake: false
    env:
      MACHBOOST_PROFILE: balanced

  quiet:
    keep_awake: false
    env:
      MACHBOOST_PROFILE: quiet

  local-build:
    keep_awake: true
    env:
      MAKEFLAGS: -j8
      CMAKE_BUILD_PARALLEL_LEVEL: "8"
      GOMAXPROCS: "8"
      CARGO_BUILD_JOBS: "8"
`
}
