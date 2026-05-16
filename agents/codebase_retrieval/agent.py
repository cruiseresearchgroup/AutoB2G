import json
import ast
from typing import List, Dict
from pathlib import Path
from agents.base_agent import BaseAgent
from .prompt_generation import (
    load_codebase,
    pipeline_selection
)

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "codebase.py"
DEFAULT_REGISTRY_FILE = BASE_DIR / "codebase.json"

def load_source_text(path=SOURCE_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ==========================================
# Extract function source
# ==========================================
def extract_function_source(fn_name: str, source_text: str) -> str:
    tree = ast.parse(source_text)
    lines = source_text.splitlines()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == fn_name:
            start = node.lineno - 1
            end = node.end_lineno
            return "\n".join(lines[start:end])

    raise RuntimeError(f"Function or class {fn_name} not found in codebase.py")


# ==========================================
# Registry Helpers
# ==========================================
def registry_map(registry):
    return {item["function_name"]: item for item in registry}


def get_required_functions(registry):
    return [r["function_name"] for r in registry if r.get("required", False)]


def get_dependencies(fn, regmap):
    return regmap[fn].get("depends_on", []) or []


# ==========================================
# Pipeline closure and validation
# ==========================================
def close_pipeline(functions, registry):
    regmap = registry_map(registry)
    plan = list(functions)

    changed = True
    while changed:
        changed = False

        for fn in list(plan):
            deps = get_dependencies(fn, regmap)
            for d in deps:
                if d in regmap and d not in plan:
                    plan.insert(0, d)
                    changed = True

    return list(dict.fromkeys(plan))


def validate_pipeline(plan, registry):
    regmap = registry_map(registry)
    missing = []

    for fn in plan:
        deps = get_dependencies(fn, regmap)
        for d in deps:
            if d not in plan:
                missing.append((fn, d))

    return missing


# ==========================================
# MAIN: Agent + Closure + Mapping to Code
# ==========================================
def workflow_generation(
        user_goal: str,
        path="codebase.json",
        max_attempts=3
):
    """
    Returns:
        str: a complete runnable Python script
    """

    # 1) load
    registry = load_codebase(path)
    full_source = load_source_text()

    attempts = 0
    candidate = None

    while attempts < max_attempts:
        attempts += 1

        # 2) LLM pipeline selection
        pipeline = pipeline_selection(user_goal, registry)

        # 3) inject required stages
        required = get_required_functions(registry)

        candidate = close_pipeline(required + pipeline, registry)

        # 4) verify dependencies
        missing = validate_pipeline(candidate, registry)

        if not missing:
            break

    # 5) final closure
    final_pipeline = close_pipeline(candidate, registry)

    # 6) extract ordered function code
    selected_code_blocks = [
        extract_function_source(fn, full_source)
        for fn in final_pipeline
    ]

    # 7) compose final runnable script
    header = """
import os
import json
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pandapower.shortcircuit as sc
import opendssdirect as dss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
while not (root / "citylearn").exists():
    root = root.parent
sys.path.insert(0, str(root))
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import BasicRBC as Agent_RBC
from citylearn.agents.base import BaselineAgent as Agent_Baseline
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import StableBaselines3Wrapper

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if PROJECT_ROOT is None:
    PROJECT_ROOT = str(root)
DATA_PATH = os.environ.get("DATA_PATH")
if DATA_PATH is None:
    DATA_PATH = "results"
DATA_DIR = str((Path(PROJECT_ROOT) / DATA_PATH).resolve())
os.makedirs(DATA_DIR, exist_ok=True)

picture_path_voltages = os.path.join(DATA_DIR, "voltages.png")
picture_path_lines = os.path.join(DATA_DIR, "line_loadings.png")
picture_path_n1 = os.path.join(DATA_DIR, "n1_violations.png")
picture_path_sc = os.path.join(DATA_DIR, "short_circuit_ikss.png")

"""

    body = "\n\n\n".join(selected_code_blocks)

    # ensure entrypoint only uses functions that actually exist
    footer = """
if __name__ == "__main__":
    run()

"""

    # combine
    script = header + body + footer

    return script


class CodebaseRetrievalAgent(BaseAgent):
    """
    Retrieves the workflow functions needed for CityLearn-grid generation.
    """

    def _load_prompt_template(self) -> str:
        return ""

    def process(
        self,
        user_goal: str,
        path=None,
        max_attempts=3
    ):
        self.logger.info("Retrieving codebase workflow")

        registry_path = Path(path) if path else Path(self.config.get("codebase_path", DEFAULT_REGISTRY_FILE))
        code = workflow_generation(
            user_goal=user_goal,
            path=registry_path,
            max_attempts=max_attempts
        )

        result = {
            "code": code,
            "metadata": {
                "registry_path": str(registry_path),
                "source_file": str(SOURCE_FILE),
                "max_attempts": max_attempts,
                "mode": "codebase_retrieval"
            }
        }
        self.logger.info("Codebase workflow retrieval completed")
        return result



