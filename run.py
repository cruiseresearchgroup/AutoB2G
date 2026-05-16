import os
import json
import subprocess
import sys
import argparse
from pathlib import Path

from utils.api_usage import reset_usage, usage_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CityLearn-grid SOCIA workflow.")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "nvidia", "anthropic"],
        help="Override the LLM provider from config.yaml.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Override the selected provider model from config.yaml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "Grid_data"
    output_path = project_root / "output_grid" / "grid_sim_output"
    usage_path = output_path / "api_usage.json"

    os.environ["OUTPUT_PATH"] = str(output_path)
    os.environ["API_USAGE_PATH"] = str(usage_path)
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    reset_usage(usage_path)

    prompt = """
You are given a runnable building-grid co-simulation framework based on the
current runnable CityLearn-grid script. It couples CityLearn building control
with a distribution grid model that can use either pandapower or OpenDSS.

The current workflow is:
build_network -> create_citylearn_env -> build_building_bus_map ->
create_citylearn_agent -> run_citylearn -> run_grid

Important modeling rules:
- Building electricity demand must come from CityLearn as building_kw.
- Buildings must be assigned to grid buses through build_building_bus_map.
- Do not assume the same total building power is blindly replicated at every node.
- The CityLearn schema must be PROJECT_ROOT/data/datasets/annex96_ce1_vt_neighborhood/schema.json.
- Do not use citylearn/data/citylearn_challenge_2022_phase_1/schema.json or any CityLearn challenge dataset.
- Prefer create_citylearn_env(dataset=None, ...) so the template default resolves the annex96 schema.
- Grid analysis should use run_grid with building_kw, net, and building_bus_map.
- Optional outputs are produced through evaluate_citylearn_kpis, evaluate_grid_kpis,
  analyze_short_circuit, and analyze_n1 when requested.

Your task is to write the full simulation code so that it matches the user's requirement.

Important constraints:
- Use only necessary functions from the retrieved workflow code.
- Preserve dependency order between CityLearn, grid mapping, and grid analysis.
- Do not add extra explanatory text or comments to the code.
"""

    user_instruction = """
Specific User Requirement:

"""

    task_description = prompt + user_instruction

    os.environ["PROJECT_ROOT"] = str(project_root)
    os.environ["DATA_PATH"] = str(data_path)

    output_path.mkdir(parents=True, exist_ok=True)
    task_file = output_path / "task_description.json"
    task_file.write_text(
        json.dumps(
            {
                "task_objective": task_description,
                "data_folder": str(data_path),
                "data_files": [],
                "evaluation_metrics": [
                    "citylearn_kpis",
                    "grid_kpis",
                    "voltage_profiles",
                    "line_loading",
                    "short_circuit",
                    "n1_resilience"
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(project_root / "main.py"),
        "--task",
        "Generate the CityLearn-grid simulation code described in the task file.",
        "--task-file",
        str(task_file),
        "--output",
        str(output_path),
        "--mode",
        "lite",
    ]

    print(f"Output directory: {output_path}")
    if args.provider:
        print(f"LLM provider: {args.provider}")
    if args.model:
        print(f"LLM model: {args.model}")
    print()

    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode == 0:
        print("\n[OK] SOCIA simulation completed successfully.")
        print(f"Results saved in: {output_path}")
    else:
        print("\n[ERROR] SOCIA simulation failed. Check logs in:")
        print(f"{output_path / 'socia.log'}")

    print(f"{usage_summary(usage_path)}")
    print(f"API usage saved in: {usage_path}")


if __name__ == "__main__":
    main()
