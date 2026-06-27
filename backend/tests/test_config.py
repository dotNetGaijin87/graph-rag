"""Unit tests for environment-sourced configuration."""

import importlib

import app.config as config_module
from app.config import Config


def test_from_env_returns_a_config_instance():
    cfg = Config.from_env()

    assert isinstance(cfg, Config)
    assert cfg.chunk_overlap < cfg.chunk_size


def test_get_bool_parses_truthy_and_falsy_strings(monkeypatch):
    monkeypatch.setenv("FEATURE_FLAG", "YES")
    assert config_module._get_bool("FEATURE_FLAG", False) is True

    monkeypatch.setenv("FEATURE_FLAG", "off")
    assert config_module._get_bool("FEATURE_FLAG", True) is False

    monkeypatch.delenv("FEATURE_FLAG", raising=False)
    assert config_module._get_bool("FEATURE_FLAG", True) is True
