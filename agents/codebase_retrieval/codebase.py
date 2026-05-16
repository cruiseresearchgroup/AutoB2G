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


class MultiObjectiveReward(RewardFunction):

    def __init__(
        self,
        env_metadata,
        reward_voltage: bool = True,
        reward_line_loading: bool = False,
        reward_cost: bool = False,
        reward_electricity: bool = False,
        reward_carbon: bool = False,
        reward_comfort: bool = False,
        line_loading_limit: float = 0.7,
        voltage_deadband: float = 0.01,
        voltage_linear_limit: float = 0.01,
        comfort_deadband: float = 0.5,
        comfort_linear_limit: float = 0.5,
        voltage_weight: float = 10000.0,
        line_loading_weight: float = 1.0,
        cost_weight: float = 100.0,
        electricity_weight: float = 10.0,
        carbon_weight: float = 1.0,
        comfort_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.reward_voltage = bool(reward_voltage)
        self.reward_line_loading = bool(reward_line_loading)
        self.reward_cost = bool(reward_cost)
        self.reward_electricity = bool(reward_electricity)
        self.reward_carbon = bool(reward_carbon)
        self.reward_comfort = bool(reward_comfort)
        self.line_loading_limit = float(line_loading_limit)
        self.voltage_deadband = float(voltage_deadband)
        self.voltage_linear_limit = float(voltage_linear_limit)
        self.comfort_deadband = float(comfort_deadband)
        self.comfort_linear_limit = float(comfort_linear_limit)
        self.voltage_weight = float(voltage_weight)
        self.line_loading_weight = float(line_loading_weight)
        self.cost_weight = float(cost_weight)
        self.electricity_weight = float(electricity_weight)
        self.carbon_weight = float(carbon_weight)
        self.comfort_weight = float(comfort_weight)
        self._printed_missing_grid_observations = False

    @staticmethod
    def _safe_mean(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return 0.0 if values.size == 0 else float(np.mean(values))

    @staticmethod
    def _piecewise_penalty(violation, linear_limit):
        violation = np.maximum(np.asarray(violation, dtype=float), 0.0)
        linear_limit = max(float(linear_limit), 0.0)
        return np.where(
            violation <= linear_limit,
            violation,
            linear_limit + np.square(violation - linear_limit),
        )

    def calculate(self, observations):
        components = {
            "voltage": 0.0,
            "line_loading": 0.0,
            "electricity": 0.0,
            "cost": 0.0,
            "carbon": 0.0,
            "comfort": 0.0,
        }

        first_observation = observations[0]

        if self.reward_voltage and "bus_voltages" in first_observation:
            voltages = np.asarray(first_observation["bus_voltages"], dtype=float)
            voltage_deviation = np.abs(np.subtract(voltages, 1.0))
            voltage_violation = np.maximum(voltage_deviation - self.voltage_deadband, 0.0)
            voltage_penalty = self._piecewise_penalty(
                voltage_violation,
                self.voltage_linear_limit,
            )
            voltage_rewards = -self.voltage_weight * voltage_penalty
            components["voltage"] = self._safe_mean(voltage_rewards)

        if self.reward_line_loading and "line_loading" in first_observation:
            line_loading = np.asarray(first_observation["line_loading"], dtype=float)
            loading_error = np.maximum(np.subtract(line_loading, self.line_loading_limit), 0.0)
            line_loading_rewards = -self.line_loading_weight * np.square(loading_error)
            components["line_loading"] = self._safe_mean(line_loading_rewards)

        if (
            (self.reward_voltage or self.reward_line_loading)
            and not self._printed_missing_grid_observations
            and (
                ("bus_voltages" not in first_observation and self.reward_voltage)
                or ("line_loading" not in first_observation and self.reward_line_loading)
            )
        ):
            print("[Reward] grid observations missing:", sorted(first_observation.keys()))
            self._printed_missing_grid_observations = True

        building_count = max(len(observations), 1)
        electricity_rewards = []
        cost_rewards = []
        carbon_rewards = []
        comfort_rewards = []

        for observation in observations:
            net_electricity = float(observation.get("net_electricity_consumption", 0.0))
            grid_electricity = max(net_electricity, 0.0)

            electricity_rewards.append(-self.electricity_weight * grid_electricity)

            electricity_price = float(observation.get("electricity_pricing", 0.0))
            cost_rewards.append(-self.cost_weight * grid_electricity * electricity_price)

            carbon_intensity = float(observation.get("carbon_intensity", 0.0))
            carbon_rewards.append(-self.carbon_weight * grid_electricity * carbon_intensity)

            indoor_temperature = float(
                observation.get("indoor_dry_bulb_temperature", 0.0)
            )
            set_point = float(
                observation.get(
                    "indoor_dry_bulb_temperature_cooling_set_point",
                    observation.get("indoor_dry_bulb_temperature_heating_set_point", indoor_temperature),
                )
            )
            comfort_violation = max(
                abs(indoor_temperature - set_point) - self.comfort_deadband,
                0.0,
            )
            comfort_penalty = self._piecewise_penalty(
                comfort_violation,
                self.comfort_linear_limit,
            )
            comfort_rewards.append(float(-self.comfort_weight * comfort_penalty))

        if self.reward_electricity:
            components["electricity"] = float(np.sum(electricity_rewards) / building_count)

        if self.reward_cost:
            components["cost"] = float(np.sum(cost_rewards) / building_count)

        if self.reward_carbon:
            components["carbon"] = float(np.sum(carbon_rewards) / building_count)

        if self.reward_comfort:
            components["comfort"] = float(np.sum(comfort_rewards) / building_count)

        reward = float(sum(components.values()))
        # print(
        #     "[Reward] "
        #     f"voltage={components['voltage']:.6f}, "
        #     f"line_loading={components['line_loading']:.6f}, "
        #     f"electricity={components['electricity']:.6f}, "
        #     f"cost={components['cost']:.6f}, "
        #     f"carbon={components['carbon']:.6f}, "
        #     f"comfort={components['comfort']:.6f}, "
        #     f"total={reward:.6f}"
        # )

        if self.central_agent:
            return [reward]

        return [reward] * len(observations)


def build_neighborhood_schema(idd_filepath: str, sample_count: int = 2, filters: dict = None, delete_energyplus_simulation_output: bool = True):
    from citylearn.end_use_load_profiles.neighborhood import Neighborhood, SampleMethod
    if filters is None:
        filters = {
            "in.resstock_county_id": ["TX, Travis County"],
            "in.vintage": ["2000s"],
        }
    build_kwargs = dict(
        idd_filepath=idd_filepath,
        delete_energyplus_simulation_output=delete_energyplus_simulation_output,
        sample_buildings_kwargs=dict(
            sample_method=SampleMethod.RANDOM,
            sample_count=sample_count,
            filters=filters,
        ),
    )
    neighborhood = Neighborhood()
    nb = neighborhood.build(**build_kwargs)
    schema_path = nb.schema_filepath
    print(f"[Neighborhood] schema generated at: {schema_path}")
    return schema_path


def build_network(
    grid_model="pandapower",
    case_name="case33bw",
    dss_path: str = None,
    max_i_ka=0.5,
    unbalanced=False,
):
    grid_model = str(grid_model).lower()
    case_name = case_name.lower()

    if grid_model == "pandapower" and unbalanced:
        raise ValueError(
            "unbalanced=True is only supported when grid_model='opendss'. "
            "Pandapower currently uses balanced runpp in this script."
        )

    if grid_model == "opendss":
        if dss is None:
            raise ImportError("opendssdirect is required when grid_model='opendss'.")
        if dss_path is None:
            dss_path = (
                Path(PROJECT_ROOT)
                / "RepresentativeLVNetworks-0.2.0" / "data" / "F"
            )
        dss_path = str(Path(dss_path))

        print(f"[NetInit] opendss: path={dss_path}")
        return {
            "grid_model": "opendss",
            "model": {
                "path": str(dss_path),
                "kvar_ratio": 0.2,
                "unbalanced": bool(unbalanced),
            },
        }

    case_map = {
        "case9": pn.case9,
        "case14": pn.case14,
        "case30": pn.case30,
        "case33bw": pn.case33bw,
        "case57": pn.case57,
        "case118": pn.case118,
        "lv": pn.create_cigre_network_lv,
    }

    if case_name not in case_map:
        supported = ", ".join(case_map.keys())
        raise ValueError(
            f"Unsupported case '{case_name}'. Supported cases: {supported}"
        )

    net = case_map[case_name]()
    if max_i_ka is not None:
        net.line["max_i_ka"] = max_i_ka
    if len(net.ext_grid) > 0:
        sc_defaults = {
            "s_sc_max_mva": 1000.0,
            "rx_max": 0.1,
            "x0x_max": 1.0,
            "r0x0_max": 0.1,
        }
        for column, value in sc_defaults.items():
            if column not in net.ext_grid.columns:
                net.ext_grid[column] = value
            else:
                net.ext_grid[column] = net.ext_grid[column].fillna(value)
    net.load.drop(net.load.index, inplace=True)
    print(f"[NetInit] {case_name}: buses={len(net.bus)}, lines={len(net.line)}")
    return {
        "grid_model": "pandapower",
        "model": net,
    }


def build_building_bus_map(env, net, slack_bus=0):
    if net["grid_model"] == "pandapower":
        available_buses = [
            int(bus) for bus in net["model"].bus.index if int(bus) != int(slack_bus)
        ]
    else:
        available_buses = list(env.unwrapped.available_grid_buses)

    building_bus_map = {
        i: available_buses[i % len(available_buses)]
        for i in range(len(env.unwrapped.buildings))
    }
    env.unwrapped.building_bus_map = dict(building_bus_map)

    print(f"[NetMap] building->bus mapping: {building_bus_map}")
    return building_bus_map

def create_citylearn_env(
    dataset: str = None,
    central: bool = True,
    observe_voltage: bool = True,
    observe_line_loading: bool = False,
    use_neighborhood: bool = False,
    idd_filepath: str = None,
    neighborhood_schema_path: str = None,
    neighborhood_build_kwargs: dict = None,
    reward_function=RewardFunction,
    reward_function_kwargs: dict = None,
    pp_net=None,
    building_bus_map=None,
    building_number: int = None,
):

    if use_neighborhood:
        if neighborhood_schema_path is None:
            if idd_filepath is None:
                raise ValueError(
                    "When use_neighborhood=True, either neighborhood_schema_path "
                    "or idd_filepath must be provided."
                )
            schema_path = build_neighborhood_schema(
                idd_filepath=idd_filepath,
                **(neighborhood_build_kwargs or {}),
            )
        else:
            schema_path = neighborhood_schema_path

        data = schema_path

    else:
        if dataset is None:
            dataset = str(
                Path(PROJECT_ROOT)
                / "data" / "datasets" / "annex96_ce1_vt_neighborhood" / "schema.json"
            )
        data = dataset

    selected_buildings = None
    building_load_scale = None

    if building_number is not None:
        with open(data, "r", encoding="utf-8-sig") as f:
            schema = json.load(f)

        all_buildings = list(schema["buildings"].keys())
        requested_buildings = int(building_number)

        if requested_buildings <= len(all_buildings):
            selected_buildings = all_buildings[:requested_buildings]
            building_load_scale = [1.0] * len(selected_buildings)
        else:
            selected_buildings = all_buildings
            base_scale, remainder = divmod(requested_buildings, len(all_buildings))
            building_load_scale = [
                float(base_scale + (1 if i < remainder else 0))
                for i in range(len(all_buildings))
            ]

    env_kwargs = dict(
        central_agent=central,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        observe_voltage=observe_voltage,
        observe_line_loading=observe_line_loading,
        grid_model=pp_net["grid_model"] if pp_net is not None else "pandapower",
        pp_net=pp_net["model"] if pp_net is not None and pp_net["grid_model"] == "pandapower" else None,
        dss_model=pp_net["model"] if pp_net is not None and pp_net["grid_model"] == "opendss" else None,
        building_bus_map=building_bus_map,
        buildings=selected_buildings,
        building_load_scale=building_load_scale,
    )

    if pp_net is not None and building_load_scale is not None:
        pp_net["building_load_scale"] = list(building_load_scale)

    env = CityLearnEnv(schema=data, **env_kwargs)

    return env

def create_citylearn_agent(env, strategy: str, episodes: int, **agent_kwargs):
    strategy = strategy.upper()
    if strategy == "RBC":
        agent = Agent_RBC(env)
    elif strategy in ("PPO", "SAC", "DDPG", "TD3"):
        try:
            from stable_baselines3 import DDPG, PPO, SAC, TD3
        except ImportError as exc:
            raise ImportError(
                "Stable-Baselines3 is required for PPO/SAC/DDPG. "
                "Install it first, for example: pip install stable-baselines3"
            ) from exc

        sb3_env = StableBaselines3Wrapper(env)
        algorithm_map = {
            "PPO": PPO,
            "SAC": SAC,
            "DDPG": DDPG,
            "TD3": TD3,
        }
        model_kwargs = {"verbose": 1}
        if strategy == "PPO":
            model_kwargs.update({"n_steps": 144, "batch_size": 12})
        if strategy in ("SAC", "DDPG", "TD3"):
            model_kwargs.update({"learning_starts": 800})

        total_timesteps = agent_kwargs.pop("total_timesteps", None)
        model_kwargs.update(agent_kwargs)

        agent = algorithm_map[strategy]("MlpPolicy", sb3_env, **model_kwargs)
        agent.citylearn_env = env
        agent.sb3_env = sb3_env

        if total_timesteps is None:
            total_timesteps = int(episodes) * int(env.unwrapped.time_steps)

        if total_timesteps > 0:
            agent.learn(total_timesteps=total_timesteps)
    elif strategy == "BASELINE":
        agent = Agent_Baseline(env)
    else:
        raise ValueError(f"Unsupported CityLearn strategy: {strategy}")
    return agent

def evaluate_citylearn_kpis(agent, save=False):
    env = getattr(agent, "citylearn_env", agent.env)
    kpis = env.evaluate()
    kpis = kpis.pivot(index="cost_function", columns="name", values="value").round(3)
    kpis = kpis.dropna(how="all")
    print(kpis)

    if save:
        kpis.to_csv(os.path.join(DATA_DIR, "citylearn_kpis.csv"))

    return kpis

def run_citylearn(env, model, trim_start: int = 20):
    observations, _ = env.reset()
    env.unwrapped.bus_voltages_history = []
    env.unwrapped.line_loading_history = []

    while not env.terminated:
        if hasattr(model, "sb3_env"):
            action, _ = model.predict(np.asarray(observations[0], dtype=np.float32), deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, env.action_space[0].low, env.action_space[0].high)
            actions = [action.tolist()]
        else:
            actions = model.predict(observations, deterministic=True)
        observations, reward, info, terminated, truncated = env.step(actions)

    buildings = env.unwrapped.buildings
    building_kw = np.stack(
        [b.net_electricity_consumption for b in buildings],
        axis=1
    )
    if trim_start > 0:
        building_kw = building_kw[int(trim_start):]
        print(f"[CityLearn] dropped first {int(trim_start)} time steps")
    print(f"[CityLearn] building_kw shape = {building_kw.shape}")
    return building_kw

def evaluate_grid_kpis(
    grid_results,
    voltage_lower_limit=0.95,
    voltage_upper_limit=1.05,
    loading_limit_percent=70.0,
    severe_loading_limit_percent=100.0,
    save=False,
):
    vm_ts = np.asarray(grid_results["vm_ts"], dtype=float)
    loading_ts = np.asarray(grid_results["loading_ts"], dtype=float)
    building_kw = np.asarray(grid_results.get("building_kw", []), dtype=float)
    district_kw = np.nansum(building_kw, axis=1) if building_kw.ndim == 2 else np.array([])
    finite_vm = vm_ts[np.isfinite(vm_ts)]
    finite_loading = loading_ts[np.isfinite(loading_ts)]
    finite_district_kw = district_kw[np.isfinite(district_kw)]
    voltage_deviation = np.abs(finite_vm - 1.0)
    undervoltage = finite_vm < voltage_lower_limit
    overvoltage = finite_vm > voltage_upper_limit
    voltage_violation = undervoltage | overvoltage
    overload = finite_loading > loading_limit_percent
    severe_overload = finite_loading > severe_loading_limit_percent

    voltage_violation_ts = (
        int(np.sum(np.any((vm_ts < voltage_lower_limit) | (vm_ts > voltage_upper_limit), axis=1)))
        if vm_ts.ndim == 2 and vm_ts.shape[0] > 0 else 0
    )
    overload_ts = (
        int(np.sum(np.any(loading_ts > loading_limit_percent, axis=1)))
        if loading_ts.ndim == 2 and loading_ts.shape[0] > 0 else 0
    )
    severe_overload_ts = (
        int(np.sum(np.any(loading_ts > severe_loading_limit_percent, axis=1)))
        if loading_ts.ndim == 2 and loading_ts.shape[0] > 0 else 0
    )

    total_district_energy_kwh = float(np.sum(finite_district_kw)) if finite_district_kw.size else np.nan
    mean_district_load_kw = float(np.mean(finite_district_kw)) if finite_district_kw.size else np.nan
    max_district_load_kw = float(np.max(finite_district_kw)) if finite_district_kw.size else np.nan
    min_district_load_kw = float(np.min(finite_district_kw)) if finite_district_kw.size else np.nan
    std_district_load_kw = float(np.std(finite_district_kw)) if finite_district_kw.size else np.nan

    mean_vm_pu = float(np.mean(finite_vm)) if finite_vm.size else np.nan
    min_vm_pu = float(np.min(finite_vm)) if finite_vm.size else np.nan
    max_vm_pu = float(np.max(finite_vm)) if finite_vm.size else np.nan
    std_vm_pu = float(np.std(finite_vm)) if finite_vm.size else np.nan
    mean_absolute_deviation_pu = float(np.mean(voltage_deviation)) if voltage_deviation.size else np.nan
    rmse_deviation_pu = (
        float(np.sqrt(np.mean(np.square(finite_vm - 1.0)))) if finite_vm.size else np.nan
    )
    max_absolute_deviation_pu = float(np.max(voltage_deviation)) if voltage_deviation.size else np.nan
    voltage_violation_rate = float(np.mean(voltage_violation.astype(float))) if voltage_violation.size else 0.0

    mean_loading_percent = float(np.mean(finite_loading)) if finite_loading.size else np.nan
    max_loading_percent = float(np.max(finite_loading)) if finite_loading.size else np.nan
    p95_loading_percent = float(np.percentile(finite_loading, 95)) if finite_loading.size else np.nan
    std_loading_percent = float(np.std(finite_loading)) if finite_loading.size else np.nan
    overload_rate = float(np.mean(overload.astype(float))) if overload.size else 0.0
    severe_overload_rate = float(np.mean(severe_overload.astype(float))) if severe_overload.size else 0.0

    rows = [
        ("simulation", "time_steps", vm_ts.shape[0], "count"),
        ("simulation", "bus_voltage_points", finite_vm.size, "count"),
        ("simulation", "line_loading_points", finite_loading.size, "count"),
        ("load", "total_district_energy_kwh", total_district_energy_kwh, "kWh"),
        ("load", "mean_district_load_kw", mean_district_load_kw, "kW"),
        ("load", "max_district_load_kw", max_district_load_kw, "kW"),
        ("load", "min_district_load_kw", min_district_load_kw, "kW"),
        ("load", "std_district_load_kw", std_district_load_kw, "kW"),
        ("voltage", "mean_vm_pu", mean_vm_pu, "p.u."),
        ("voltage", "min_vm_pu", min_vm_pu, "p.u."),
        ("voltage", "max_vm_pu", max_vm_pu, "p.u."),
        ("voltage", "std_vm_pu", std_vm_pu, "p.u."),
        ("voltage", "mean_absolute_deviation_pu", mean_absolute_deviation_pu, "p.u."),
        ("voltage", "rmse_deviation_pu", rmse_deviation_pu, "p.u."),
        ("voltage", "max_absolute_deviation_pu", max_absolute_deviation_pu, "p.u."),
        ("voltage", "undervoltage_count", int(np.sum(undervoltage)), "count"),
        ("voltage", "overvoltage_count", int(np.sum(overvoltage)), "count"),
        ("voltage", "voltage_violation_count", int(np.sum(voltage_violation)), "count"),
        ("voltage", "voltage_violation_rate", voltage_violation_rate, "fraction"),
        ("voltage", "time_steps_with_voltage_violation", voltage_violation_ts, "count"),
        ("line_loading", "mean_loading_percent", mean_loading_percent, "%"),
        ("line_loading", "max_loading_percent", max_loading_percent, "%"),
        ("line_loading", "p95_loading_percent", p95_loading_percent, "%"),
        ("line_loading", "std_loading_percent", std_loading_percent, "%"),
        ("line_loading", "overload_count", int(np.sum(overload)), "count"),
        ("line_loading", "overload_rate", overload_rate, "fraction"),
        ("line_loading", "severe_overload_count", int(np.sum(severe_overload)), "count"),
        ("line_loading", "severe_overload_rate", severe_overload_rate, "fraction"),
        ("line_loading", "time_steps_with_overload", overload_ts, "count"),
        ("line_loading", "time_steps_with_severe_overload", severe_overload_ts, "count"),
    ]
    kpis = pd.DataFrame(rows, columns=["category", "metric", "value", "unit"])
    print(kpis)

    if save:
        kpis.to_csv(os.path.join(DATA_DIR, "grid_kpis.csv"), index=False)

    return kpis

def run_grid(building_kw, net, building_bus_map, plot=False, save=False, save_kpis=False):
    T, _ = building_kw.shape
    building_load_scale = np.array(
        net.get("building_load_scale", [1.0] * building_kw.shape[1]),
        dtype=float
    )
    building_mw = building_kw * building_load_scale.reshape(1, -1) / 1000.0
    if net["grid_model"] == "pandapower":
        net_ts = {
            "grid_model": "pandapower",
            "model": copy.deepcopy(net["model"]),
            "building_load_scale": building_load_scale.tolist(),
        }
        load_idx = {}
        slack_buses = (
            set(net_ts["model"].ext_grid["bus"].astype(int).tolist())
            if hasattr(net_ts["model"], "ext_grid") and not net_ts["model"].ext_grid.empty
            else {0}
        )
        voltage_bus_indices = [idx for idx in net_ts["model"].bus.index if int(idx) not in slack_buses]
        for building_id, bus in building_bus_map.items():
            load_idx[building_id] = pp.create_load(
                net_ts["model"],
                bus=int(bus),
                p_mw=0.0,
                q_mvar=0.0,
                name=f"class{building_id}_bus{bus}",
            )

        vm_ts = []
        loading_ts = []

        for t in range(T):
            for building_id, idx in load_idx.items():
                net_ts["model"].load.at[idx, "p_mw"] = float(building_mw[t, building_id])
            pp.runpp(net_ts["model"])
            vm_ts.append(net_ts["model"].res_bus.loc[voltage_bus_indices, "vm_pu"].to_numpy().copy())
            loading_ts.append(net_ts["model"].res_line.loading_percent.values.copy())

        grid_results = {
            "vm_ts": np.array(vm_ts),
            "loading_ts": np.array(loading_ts),
            "net": net,
            "net_ts": net_ts,
            "building_kw": building_kw,
        }
    elif net["grid_model"] == "opendss":
        model = net["model"]
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{os.path.join(model["path"], "Master.dss")}"')
        if dss.Loads.Count() > 0:
            dss.Loads.First()
            while True:
                name = dss.Loads.Name()
                dss.Text.Command(f"edit Load.{name} enabled=no")
                if dss.Loads.Next() == 0:
                    break
        unbalanced = bool(model.get("unbalanced", False))
        for building_id, bus in building_bus_map.items():
            if unbalanced:
                phase = building_id % 3 + 1
                dss.Text.Command(
                    f"new Load.bldg_{building_id} bus1={bus}.{phase} phases=1 kV=0.4 kW=0 kvar=0"
                )
            else:
                dss.Text.Command(
                    f"new Load.bldg_{building_id} bus1={bus} phases=3 kV=0.4 kW=0 kvar=0"
                )

        line_names = dss.Lines.AllNames() if dss.Lines.Count() > 0 else []
        vm_ts = []
        loading_ts = []

        for t in range(T):
            kvar_ratio = float(model.get("kvar_ratio", 0.2))
            for building_id in building_bus_map:
                kw = float(building_kw[t, building_id] * building_load_scale[building_id])
                kvar = kw * kvar_ratio
                dss.Text.Command(f"edit Load.bldg_{building_id} kW={kw} kvar={kvar}")

            dss.Solution.Solve()
            slack_bus = None
            if dss.Vsources.Count() > 0:
                dss.Vsources.First()
                slack_bus = dss.CktElement.BusNames()[0].split('.')[0]
            bus_voltage = {}
            for node, voltage in zip(dss.Circuit.AllNodeNames(), np.array(dss.Circuit.AllBusMagPu(), dtype=float)):
                parts = node.split('.')
                bus = parts[0]
                if bus == slack_bus:
                    continue
                phase = parts[1] if len(parts) > 1 else "1"
                bus_voltage.setdefault(bus, {})
                bus_voltage[bus][phase] = voltage
            vm_ts.append(np.array(
                [voltage for phases in bus_voltage.values() for voltage in phases.values()],
                dtype=float
            ))

            line_loading = []
            for line_name in line_names:
                dss.Lines.Name(line_name)
                dss.Circuit.SetActiveElement(f"Line.{line_name}")
                currents = dss.CktElement.CurrentsMagAng()[0::2]
                norm_amps = dss.Lines.NormAmps()
                if norm_amps and len(currents) > 0:
                    line_loading.append(max(currents) / norm_amps * 100.0)
                else:
                    line_loading.append(np.nan)

            loading_ts.append(np.array(line_loading, dtype=float))

        grid_results = {
            "vm_ts": np.array(vm_ts),
            "loading_ts": np.array(loading_ts, dtype=float),
            "net": net,
            "net_ts": None,
            "building_kw": building_kw,
        }
    else:
        raise ValueError(f"Unsupported grid_model: {net['grid_model']}")

    if plot:
        vm_ts = np.asarray(grid_results["vm_ts"])
        loading_ts = np.asarray(grid_results["loading_ts"])

        volt_df = pd.DataFrame(vm_ts)
        plt.figure()
        volt_df.plot(legend=False)
        plt.xlabel("Time step")
        plt.ylabel("Voltage [p.u.]")
        plt.title("Bus voltages over time")
        plt.tight_layout()
        if save:
            plt.savefig(picture_path_voltages)
        plt.close()

        line_df = pd.DataFrame(loading_ts)
        plt.figure()
        line_df.plot(legend=False)
        plt.xlabel("Time step")
        plt.ylabel("Line loading [%]")
        plt.title("Line loadings over time")
        plt.tight_layout()
        if save:
            plt.savefig(picture_path_lines)
        plt.close()

    if save:
        vm_ts = np.asarray(grid_results["vm_ts"])
        loading_ts = np.asarray(grid_results["loading_ts"])

        pd.DataFrame(vm_ts).to_csv(
            os.path.join(DATA_DIR, "voltages.csv"), index=False
        )

        pd.DataFrame(loading_ts).to_csv(
            os.path.join(DATA_DIR, "line_loading.csv"), index=False
        )

    if save_kpis:
        grid_results["kpis"] = evaluate_grid_kpis(grid_results, save=True)

    print("[Grid] powerflow finished")
    return grid_results

def analyze_short_circuit(net, case="max", ip=False, ith=False, plot=False, save=False):
    if net["grid_model"] == "pandapower":
        net_sc = copy.deepcopy(net["model"])
        sc.calc_sc(net_sc, case=case, ip=ip, ith=ith)

        bus = net_sc.res_bus_sc.copy()
        bus["bus"] = bus.index.astype(str)
        bus = bus.reset_index(drop=True)
        bus = bus[[col for col in ["bus", "ikss_ka"] if col in bus.columns]]
        results = {"bus": bus}
    elif net["grid_model"] == "opendss":
        model = net["model"]
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{os.path.join(model["path"], "Master.dss")}"')
        slack_bus = None
        if dss.Vsources.Count() > 0:
            dss.Vsources.First()
            slack_bus = dss.CktElement.BusNames()[0].split('.')[0]
        all_buses = dss.Circuit.AllBusNames()
        bus_rows = []

        for bus in all_buses:
            if bus == slack_bus:
                continue
            dss.Basic.ClearAll()
            dss.Text.Command(f'compile "{os.path.join(model["path"], "Master.dss")}"')
            dss.Text.Command("set controlmode=off")
            dss.Text.Command(f"new Fault.sc_temp bus1={bus}.1.2.3 phases=3 r=0.0001")
            dss.Solution.Solve()
            dss.Circuit.SetActiveElement("Fault.sc_temp")
            currents = dss.CktElement.CurrentsMagAng()[0::2]
            ikss_ka = max(currents) / 1000.0 if len(currents) > 0 else np.nan
            bus_rows.append(
                {
                    "bus": bus,
                    "ikss_ka": ikss_ka,
                }
            )

        results = {
            "bus": pd.DataFrame(bus_rows, columns=["bus", "ikss_ka"]),
        }
    else:
        raise ValueError(f"Unsupported grid_model: {net['grid_model']}")

    results["table"] = results["bus"].copy()

    if plot and "ikss_ka" in results["bus"].columns and not results["bus"].empty:
        plt.figure()
        plot_series = results["bus"].set_index("bus")["ikss_ka"]
        plot_series.plot()
        plt.xlabel("Bus")
        plt.ylabel("Ikss [kA]")
        plt.title("Short-circuit Current")
        plt.tight_layout()
        plt.savefig(picture_path_sc)
        plt.close()

    if save:
        results["table"].to_csv(os.path.join(DATA_DIR, "short_circuit.csv"), index=False)

    return results

def analyze_n1(net, building_kw, building_bus_map,
               vm_tol=0.05,
               loading_threshold=70.0,
               T_run=24*7,
               plot=False,
               save=False):
    print("[N-1] analysis started")
    T_total, _ = building_kw.shape
    building_load_scale = np.array(
        net.get("building_load_scale", [1.0] * building_kw.shape[1]),
        dtype=float
    )
    building_mw = building_kw * building_load_scale.reshape(1, -1) / 1000.0
    T_use = T_total if T_run is None else min(int(T_run), T_total)
    under_ts, over_ts, overload_ts = [], [], []
    detail_rows = []
    if net["grid_model"] == "pandapower":
        net_ts = {
            "grid_model": "pandapower",
            "model": copy.deepcopy(net["model"]),
            "building_load_scale": list(net.get("building_load_scale", [1.0] * building_kw.shape[1])),
        }
        load_idx = {}
        slack_buses = (
            set(net_ts["model"].ext_grid["bus"].astype(int).tolist())
            if hasattr(net_ts["model"], "ext_grid") and not net_ts["model"].ext_grid.empty
            else {0}
        )
        voltage_bus_indices = [idx for idx in net_ts["model"].bus.index if int(idx) not in slack_buses]
        for building_id, bus in building_bus_map.items():
            load_idx[building_id] = pp.create_load(
                net_ts["model"],
                bus=int(bus),
                p_mw=0.0,
                q_mvar=0.0,
                name=f"class{building_id}_bus{bus}",
            )

        for t in range(T_use):
            for building_id, idx in load_idx.items():
                net_ts["model"].load.at[idx, "p_mw"] = float(building_mw[t, building_id])
            pp.runpp(net_ts["model"])

            total_under = 0
            total_over = 0
            total_overload = 0

            for line_idx in net_ts["model"].line.index:
                net_cont = copy.deepcopy(net_ts)
                net_cont["model"].line.at[line_idx, "in_service"] = False

                try:
                    pp.runpp(net_cont["model"])
                except Exception:
                    total_under += 1
                    detail_rows.append({
                        "time_step": t,
                        "line_outage": str(line_idx),
                        "solved": False,
                        "undervoltage_count": 1,
                        "overvoltage_count": 0,
                        "overload_count": 0,
                        "deenergized_count": np.nan,
                        "vmin": np.nan,
                        "vmax": np.nan,
                        "max_loading": np.nan,
                    })
                    continue

                if not net_cont["model"].converged:
                    total_under += 1
                    detail_rows.append({
                        "time_step": t,
                        "line_outage": str(line_idx),
                        "solved": False,
                        "undervoltage_count": 1,
                        "overvoltage_count": 0,
                        "overload_count": 0,
                        "deenergized_count": np.nan,
                        "vmin": np.nan,
                        "vmax": np.nan,
                        "max_loading": np.nan,
                    })
                    continue

                vm = net_cont["model"].res_bus.loc[voltage_bus_indices, "vm_pu"].to_numpy()
                loading = net_cont["model"].res_line.loading_percent.values
                energized_vm = vm[vm > 1e-6]
                deenergized_count = int((vm <= 1e-6).sum())

                under_count = int((vm < 1.0 - vm_tol).sum())
                over_count = int((vm > 1.0 + vm_tol).sum())
                overload_count = int((loading > loading_threshold).sum())
                total_under += under_count
                total_over += over_count
                total_overload += overload_count
                detail_rows.append({
                    "time_step": t,
                    "line_outage": str(line_idx),
                    "solved": True,
                    "undervoltage_count": under_count,
                    "overvoltage_count": over_count,
                    "overload_count": overload_count,
                    "deenergized_count": deenergized_count,
                    "vmin": float(np.min(energized_vm)) if energized_vm.size > 0 else np.nan,
                    "vmax": float(np.max(energized_vm)) if energized_vm.size > 0 else np.nan,
                    "max_loading": float(np.max(loading)) if loading.size > 0 else np.nan,
                })

            under_ts.append(total_under)
            over_ts.append(total_over)
            overload_ts.append(total_overload)
    elif net["grid_model"] == "opendss":
        model = net["model"]
        for t in range(T_use):
            total_under = 0
            total_over = 0
            total_overload = 0

            dss.Basic.ClearAll()
            dss.Text.Command(f'compile "{os.path.join(model["path"], "Master.dss")}"')
            if dss.Loads.Count() > 0:
                dss.Loads.First()
                while True:
                    name = dss.Loads.Name()
                    dss.Text.Command(f"edit Load.{name} enabled=no")
                    if dss.Loads.Next() == 0:
                        break

            unbalanced = bool(model.get("unbalanced", False))
            for building_id, bus in building_bus_map.items():
                if unbalanced:
                    phase = building_id % 3 + 1
                    dss.Text.Command(
                        f"new Load.bldg_{building_id} bus1={bus}.{phase} phases=1 kV=0.4 kW=0 kvar=0"
                    )
                else:
                    dss.Text.Command(
                        f"new Load.bldg_{building_id} bus1={bus} phases=3 kV=0.4 kW=0 kvar=0"
                    )

            line_names = dss.Lines.AllNames() if dss.Lines.Count() > 0 else []
            kvar_ratio = float(model.get("kvar_ratio", 0.2))
            for building_id in building_bus_map:
                kw = float(building_kw[t, building_id] * building_load_scale[building_id])
                kvar = kw * kvar_ratio
                dss.Text.Command(f"edit Load.bldg_{building_id} kW={kw} kvar={kvar}")

            for line_name in line_names:
                dss.Text.Command(f"edit Line.{line_name} enabled=no")
                dss.Solution.Solve()
                solved = bool(dss.Solution.Converged())

                slack_bus = None
                if dss.Vsources.Count() > 0:
                    dss.Vsources.First()
                    slack_bus = dss.CktElement.BusNames()[0].split('.')[0]

                bus_voltage = {}
                for node, voltage in zip(dss.Circuit.AllNodeNames(), np.array(dss.Circuit.AllBusMagPu(), dtype=float)):
                    parts = node.split('.')
                    bus = parts[0]
                    if bus == slack_bus:
                        continue
                    phase = parts[1] if len(parts) > 1 else "1"
                    bus_voltage.setdefault(bus, {})
                    bus_voltage[bus][phase] = voltage

                all_v = np.array(
                    [v for phases in bus_voltage.values() for v in phases.values()],
                    dtype=float
                )
                energized_v = all_v[all_v > 1e-6]
                deenergized_count = int((all_v <= 1e-6).sum()) if all_v.size > 0 else np.nan
                if (not solved) or all_v.size == 0:
                    total_under += 1
                    detail_rows.append({
                        "time_step": t,
                        "line_outage": str(line_name),
                        "solved": solved,
                        "undervoltage_count": 1,
                        "overvoltage_count": 0,
                        "overload_count": 0,
                        "deenergized_count": deenergized_count,
                        "vmin": np.nan,
                        "vmax": np.nan,
                        "max_loading": np.nan,
                    })
                    dss.Text.Command(f"edit Line.{line_name} enabled=yes")
                    continue
                else:
                    under_count = int((all_v < 1.0 - vm_tol).sum())
                    over_count = int((all_v > 1.0 + vm_tol).sum())
                    total_under += under_count
                    total_over += over_count

                overload_count = 0
                max_loading = np.nan
                for active_line in line_names:
                    if active_line == line_name:
                        continue
                    dss.Lines.Name(active_line)
                    dss.Circuit.SetActiveElement(f"Line.{active_line}")
                    currents = dss.CktElement.CurrentsMagAng()[0::2]
                    norm_amps = dss.Lines.NormAmps()
                    if norm_amps and len(currents) > 0:
                        loading_pct = max(currents) / norm_amps * 100.0
                        overload_count += int(loading_pct > loading_threshold)
                        max_loading = loading_pct if np.isnan(max_loading) else max(max_loading, loading_pct)

                total_overload += overload_count
                detail_rows.append({
                    "time_step": t,
                    "line_outage": str(line_name),
                    "solved": solved,
                    "undervoltage_count": under_count,
                    "overvoltage_count": over_count,
                    "overload_count": overload_count,
                    "deenergized_count": deenergized_count,
                    "vmin": float(np.min(energized_v)) if energized_v.size > 0 else np.nan,
                    "vmax": float(np.max(energized_v)) if energized_v.size > 0 else np.nan,
                    "max_loading": max_loading,
                })

                dss.Text.Command(f"edit Line.{line_name} enabled=yes")

            under_ts.append(total_under)
            over_ts.append(total_over)
            overload_ts.append(total_overload)
    else:
        raise ValueError(f"Unsupported grid_model: {net['grid_model']}")

    summary = pd.DataFrame({
        "time_step": np.arange(T_use, dtype=int),
        "undervoltage_count": np.array(under_ts, dtype=int),
        "overvoltage_count": np.array(over_ts, dtype=int),
        "overload_count": np.array(overload_ts, dtype=int),
    })
    detail = pd.DataFrame(detail_rows)

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        axes[0].plot(summary["time_step"], summary["undervoltage_count"], label="undervoltage")
        axes[0].plot(summary["time_step"], summary["overvoltage_count"], label="overvoltage")
        axes[0].plot(summary["time_step"], summary["overload_count"], label="overload")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        if not detail.empty:
            detail["total_violations"] = (
                detail["undervoltage_count"] + detail["overvoltage_count"] + detail["overload_count"]
            )
            heatmap = detail.pivot_table(
                index="line_outage",
                columns="time_step",
                values="total_violations",
                aggfunc="sum",
                fill_value=0.0,
            )
            im = axes[1].imshow(heatmap.values, aspect="auto", interpolation="nearest")
            axes[1].set_yticks(np.arange(len(heatmap.index)))
            axes[1].set_yticklabels(heatmap.index)
            axes[1].set_xlabel("Time step")
            axes[1].set_ylabel("Outaged line")
            axes[1].set_title("Total N-1 Violations by Outage")
            fig.colorbar(im, ax=axes[1], label="Violation count")

        if save:
            plt.savefig(picture_path_n1)
        plt.close()

    if save:
        detail.to_csv(os.path.join(DATA_DIR, "n1_result.csv"), index=False)

    print(
        f"[N-1] analysis finished: "
        f"steps={T_use}, "
        f"max_under={np.max(under_ts):.0f}, "
        f"max_over={np.max(over_ts):.0f}, "
        f"max_overload={np.max(overload_ts):.0f}"
    )

    return {
        "under_ts": np.array(under_ts, dtype=int),
        "over_ts": np.array(over_ts, dtype=int),
        "overload_ts": np.array(overload_ts, dtype=int),
        "summary": summary,
        "detail": detail,
    }


def run():
    net = build_network(grid_model='opendss', unbalanced=False)
    env = create_citylearn_env(
        reward_function=MultiObjectiveReward,
        reward_function_kwargs={
            "reward_voltage": True,
            "reward_line_loading": False,
            "reward_cost": True,
            "reward_electricity": False,
            "reward_comfort": False,
            "reward_carbon": False,
        },
        observe_voltage=True,
        observe_line_loading=False,
        pp_net=net,
        building_number=2,
    )
    building_bus_map = build_building_bus_map(env, net)
    model = create_citylearn_agent(env, strategy="SAC", episodes=2)
    building_kw = run_citylearn(env, model)
    evaluate_citylearn_kpis(model, save=True)
    run_grid(building_kw, net, building_bus_map, plot=True, save=True, save_kpis=True)
    analyze_short_circuit(net, plot=True, save=True)
    analyze_n1(net, building_kw, building_bus_map, plot=True, save=True)

if __name__ == "__main__":
    run()
