import json
from typing import Dict, List, Optional

from utils.llm_utils import get_llm_provider


def load_codebase(path="codebase.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_api_key(key_name: str) -> Optional[str]:
    """
    Load API key from keys.py file.
    """
    try:
        import keys

        return getattr(keys, key_name, None)
    except ImportError:
        return None


def pipeline_selection(
    user_instruction: str,
    registry: List[Dict],
    model="gpt-4o",
) -> List[str]:
    system_prompt = """
You are an AI pipeline planner for the runnable CityLearn-grid script.

BACKGROUND
----------
The codebase couples CityLearn building-control simulation with distribution-grid
analysis.

1) CityLearn
   - create_citylearn_env builds the CityLearn environment.
   - create_citylearn_agent creates or trains the control policy.
   - run_citylearn runs one episode and outputs building_kw, a time-series matrix
     of building electricity demand.
   - evaluate_citylearn_kpis evaluates the CityLearn controller.
   - build_neighborhood_schema is optional and is used only when the user asks to
     generate a new EnergyPlus/CityLearn neighborhood schema.

2) Distribution grid
   - build_network initializes either a pandapower network or an OpenDSS network.
   - build_building_bus_map maps CityLearn buildings to grid buses.
   - run_grid runs time-series power flow using building_kw and building_bus_map.
   - evaluate_grid_kpis calculates grid KPI tables from run_grid results.
   - analyze_short_circuit runs short-circuit/fault-current analysis.
   - analyze_n1 runs N-1 contingency analysis.


TYPICAL EXECUTION ORDER
-----------------------
Full CityLearn plus grid workflow:

build_network -> create_citylearn_env -> build_building_bus_map ->
create_citylearn_agent -> run_citylearn -> run_grid

Optional evaluation and analysis functions are appended only when requested:

evaluate_citylearn_kpis, evaluate_grid_kpis, analyze_short_circuit, analyze_n1


PIPELINE RULES
--------------
- Use ONLY function names that exist in the provided registry.
- Do NOT invent function names.
- Maintain dependency-based execution order.
- Prefer the minimum valid pipeline that satisfies the user request.
- Include build_neighborhood_schema only if the user explicitly asks to generate
  a neighborhood dataset, EnergyPlus profiles, or a new CityLearn schema.


OUTPUT FORMAT
-------------
Return ONLY JSON in this exact format:

{
  "pipeline": [
     "function_A",
     "function_B"
  ]
}

No explanations, no markdown, no extra keys.
"""

    user_prompt = f"""
USER REQUEST:
{user_instruction}

AVAILABLE FUNCTIONS:
{json.dumps(registry, indent=2, ensure_ascii=False)}
"""

    llm_provider = get_llm_provider({"provider": _load_provider_name()})
    content = llm_provider.call(
        "SYSTEM:\n"
        + system_prompt
        + "\n\nUSER:\n"
        + user_prompt
    ).strip()
    if content.startswith("Error:"):
        raise RuntimeError(content)

    content = content[content.find("{"): content.rfind("}") + 1]
    if not content:
        raise RuntimeError("Pipeline selection returned no JSON content.")

    parsed = json.loads(content)
    return parsed["pipeline"]


def _load_provider_name() -> str:
    import os

    provider = os.environ.get("LLM_PROVIDER")
    if provider:
        return provider

    try:
        import yaml

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config.get("llm", {}).get("provider", "openai")
    except Exception:
        return "openai"
