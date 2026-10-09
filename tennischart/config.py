"""Paths and defaults. Everything can be overridden with environment variables."""

from __future__ import annotations

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PACKAGE_DIR)
DATA_DIR = os.environ.get("TENNISCHART_DATA", os.path.join(REPO_DIR, "data"))
RESOURCES_DIR = os.path.join(PACKAGE_DIR, "resources")
CACHE_DIR = os.environ.get("TENNISCHART_CACHE", os.path.join(REPO_DIR, ".cache", "tennischart"))
RUNS_DIR = os.environ.get("TENNISCHART_RUNS", os.path.join(REPO_DIR, "runs"))
MODELS_DIR = os.environ.get("TENNISCHART_MODELS", os.path.join(REPO_DIR, ".cache", "models"))

# Decisions API: hard ceiling on cumulative spend recorded in the ledger, in USD.
DECISIONS_BUDGET_USD = float(os.environ.get("TENNISCHART_DECISIONS_BUDGET", "15"))
DECISIONS_MODEL = os.environ.get("TENNISCHART_DECISIONS_MODEL", "gpt-6-luna")
DECISIONS_URL = os.environ.get("TENNISCHART_DECISIONS_URL", "https://api.openai.com/v1/decisions")
DECISIONS_USD_PER_MTOK = float(os.environ.get("TENNISCHART_DECISIONS_USD_PER_MTOK", "0.10"))
