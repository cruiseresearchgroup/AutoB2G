import os
import numpy as np
import opendssdirect as dss
import matplotlib.pyplot as plt

# ========= 0. 工具函数 =========
def compile_network(path):
    dss.Basic.ClearAll()
    master_path = os.path.join(path, "Master.dss")
    dss.Text.Command(f'compile "{master_path}"')


def get_bus_voltage():
    nodes = dss.Circuit.AllNodeNames()
    v = np.array(dss.Circuit.AllBusMagPu())

    bus_voltage = {}
    for node, value in zip(nodes, v):
        bus = node.split('.')[0]
        if bus not in bus_voltage:
            bus_voltage[bus] = value
    return bus_voltage


def get_bus_load_info():
    bus_loads = {}
    bus_p = {}

    if dss.Loads.Count() == 0:
        return bus_loads, bus_p

    dss.Loads.First()
    while True:
        name = dss.Loads.Name()
        bus = dss.CktElement.BusNames()[0].split('.')[0]
        p = dss.Loads.kW()

        bus_loads.setdefault(bus, []).append(name)
        bus_p.setdefault(bus, 0.0)
        bus_p[bus] += p

        if dss.Loads.Next() == 0:
            break

    return bus_loads, bus_p


def get_line_losses_kw():
    losses = {}
    total_loss = 0.0

    if dss.Lines.Count() == 0:
        return losses, total_loss

    dss.Lines.First()
    while True:
        name = dss.Lines.Name()
        dss.Circuit.SetActiveElement(f"Line.{name}")
        loss_kw = dss.CktElement.Losses()[0] / 1000.0
        losses[name] = loss_kw
        total_loss += loss_kw

        if dss.Lines.Next() == 0:
            break

    return losses, total_loss


def print_static_summary():
    print("=== 基本信息 ===")
    print("Converged:", dss.Solution.Converged())
    print("Buses:", dss.Circuit.NumBuses())
    print("Loads:", dss.Loads.Count())
    print("Lines:", dss.Lines.Count())
    print("PV Systems:", dss.PVsystems.Count())
    print("Storage:", dss.Storages.Count())
    print("Generators:", dss.Generators.Count())

    bus_voltage = get_bus_voltage()
    print("\n=== Bus 电压 (p.u.) ===")
    for b, v in bus_voltage.items():
        print(f"{b}: {v:.4f}")

    bus_loads, bus_p = get_bus_load_info()
    print("\n=== Bus 负荷及有功功率 ===")
    for b in bus_voltage.keys():
        loads = bus_loads.get(b, [])
        total_p = bus_p.get(b, 0.0)
        print(f"\nBus {b}:")
        print("  Loads:", loads if loads else "None")
        print(f"  Total Load P: {total_p:.2f} kW")

    print("\n=== 系统功率平衡 ===")
    total_load = sum(bus_p.values())
    p_sys = dss.Circuit.TotalPower()[0]   # DSS里负荷通常显示为负号
    print(f"Total Load (sum loads): {total_load:.2f} kW")
    print(f"Total Power from Grid (from DSS): {-p_sys:.2f} kW")

    line_losses, total_loss = get_line_losses_kw()
    print("\n=== 每条线路损耗 (kW) ===")
    for name, loss in line_losses.items():
        print(f"{name}: {loss:.4f} kW")
    print(f"\nTotal Line Loss: {total_loss:.4f} kW")


def add_capacitor(bus="85", kv=0.4, kvar=100):
    dss.Text.Command(f"new Capacitor.cap1 bus1={bus} kV={kv} kvar={kvar}")


def add_multiple_loads(load_profiles, kv=0.4):

    for name, data in load_profiles.items():
        bus = data["bus"]
        profile = data["profile"]

        kw0 = profile[0]
        kvar = kw0 * 0.2

        phase = data.get("phase", None)  # ⭐ 关键：可选参数

        if phase is None:
            # ===== 默认三相 =====
            dss.Text.Command(
                f"new Load.{name} bus1={bus} kV={kv} kW={kw0} kvar={kvar}"
            )
        else:
            # ===== 单相（或指定相）=====
            dss.Text.Command(
                f"new Load.{name} bus1={bus}.{phase} phases=1 kV={kv} kW={kw0} kvar={kvar}"
            )

def disable_all_loads():

    if dss.Loads.Count() == 0:
        return

    dss.Loads.First()
    while True:
        name = dss.Loads.Name()

        dss.Text.Command(f"edit Load.{name} enabled=no")

        if dss.Loads.Next() == 0:
            break

def run_daily_simulation(load_profiles):

    dss.Vsources.First()
    slack_bus = dss.CktElement.BusNames()[0].split('.')[0]

    dss.Text.Command("set mode=daily")
    dss.Text.Command("set stepsize=1h")
    dss.Text.Command("set number=1")

    results = []

    print("\n=== 24小时仿真 ===")

    for hour in range(24):

        dss.Solution.Hour(hour)

        # ========= 负荷更新 =========
        for name, data in load_profiles.items():
            kw = data["profile"][hour]
            kvar = kw * 0.2

            dss.Text.Command(
                f"edit Load.{name} kW={kw} kvar={kvar}"
            )

        # ========= 电容控制（示例：白天开） =========
        # if 6 <= hour <= 18:
        #     dss.Text.Command("edit Capacitor.cap1 states=1")
        #     cap_state = 1
        # else:
        #     dss.Text.Command("edit Capacitor.cap1 states=0")
        #     cap_state = 0

        dss.Solution.Solve()

        # ========= 电压 =========
        V = np.array(dss.Circuit.AllBusMagPu())
        nodes = dss.Circuit.AllNodeNames()

        bus_voltage = {}

        for node, v in zip(nodes, V):
            parts = node.split('.')
            bus = parts[0]

            if bus == slack_bus:
                continue

            phase = parts[1] if len(parts) > 1 else "1"

            bus_voltage.setdefault(bus, {})
            bus_voltage[bus][phase] = v

        all_v = [v for b in bus_voltage.values() for v in b.values()]
        vmin = min(all_v) if all_v else np.nan
        vmax = max(all_v) if all_v else np.nan

        # ========= 功率 =========
        total_load = 0.0
        dss.Loads.First()
        while True:
            total_load += dss.Loads.kW()
            if dss.Loads.Next() == 0:
                break

        grid_kw = -dss.Circuit.TotalPower()[0]

        total_loss = 0.0
        dss.Lines.First()
        while True:
            dss.Circuit.SetActiveElement(f"Line.{dss.Lines.Name()}")
            total_loss += dss.CktElement.Losses()[0] / 1000
            if dss.Lines.Next() == 0:
                break

        results.append({
            "hour": hour,
            "bus_voltage": bus_voltage,
            "vmin": vmin,
            "vmax": vmax,
            "grid_kw": grid_kw,
            "line_loss_kw": total_loss,
            "load_kw": total_load,
        })

        print(
            f"Hour {hour:02d} | "
            f"Load={total_load:7.2f} | "
            f"Grid={grid_kw:7.2f} | "
            f"Loss={total_loss:6.4f} | "
            f"Vmin={vmin:.4f} | "
            f"Vmax={vmax:.4f}"
        )

    return results

def plot_bus_voltage_3phase(results, bus_list=None, top_n=None):

    all_buses = list(results[0]["bus_voltage"].keys())

    if bus_list is not None:
        buses = bus_list
    elif top_n is not None:
        buses = all_buses[:top_n]
    else:
        buses = all_buses

    hours = [r["hour"] for r in results]

    n = len(buses)

    # ===== 自动布局（关键优化）=====
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2 * nrows), sharex=True)

    axes = np.array(axes).reshape(-1)  # 展平成1D方便索引

    linestyles = {
        "1": "-",
        "2": "--",
        "3": ":"
    }

    for idx, b in enumerate(buses):
        ax = axes[idx]

        # ===== 收集所有相 =====
        phases = set()
        for r in results:
            phases.update(r["bus_voltage"].get(b, {}).keys())

        # ===== 每个相画一条 =====
        for ph in sorted(phases):
            v_series = [
                r["bus_voltage"].get(b, {}).get(ph, np.nan)
                for r in results
            ]

            ax.plot(hours, v_series,
                    linestyle=linestyles.get(ph, "-"),
                    label=f"Phase {ph}")

        # ===== 电压限制 =====
        ax.axhline(0.95, linestyle="--", color='r')
        ax.axhline(1.05, linestyle="--", color='r')

        ax.set_title(f"Bus {b}")
        ax.set_ylabel("Voltage (p.u.)")
        ax.grid()

        # 只在第一个图显示 legend（避免重复）
        if idx == 0:
            ax.legend()

    # ===== 删除多余子图（很重要）=====
    for i in range(len(buses), len(axes)):
        fig.delaxes(axes[i])

    axes[-1].set_xlabel("Hour")

    plt.tight_layout()

# ========= 1. 加载网络 =========
path = r"data/C"
compile_network(path)
dss.Solution.Solve()

print("############ 原始网络 ############")
print_static_summary()

# ========= 2. 关闭原始负荷 =========
disable_all_loads()

# ========= 3. 自定义负荷 =========
load_profiles = {
    "user_load_0": {
        "bus": "85",
        "profile": [x * -10 for x in [
            6, 6, 5, 5, 5, 6,
            8, 10, 11, 12, 12, 11,
            10, 9, 9, 10, 11, 13,
            14, 13, 12, 10, 8, 7]]
    },
    "user_load_1": {
        "bus": "44",
        "phase": 1,
        "profile": [x * -5 for x in [
    4,4,3,3,3,4,
    6,8,9,10,10,9,
    8,7,7,8,9,10,
    11,10,9,8,6,5
        ]]
    }
}

add_multiple_loads(load_profiles)

# ========= 4. 加电容 =========
add_capacitor(bus="45", kvar=100)

print("\n############ 自定义负荷网络 ############")
print_static_summary()

# ========= 5. 仿真 =========
daily_results = run_daily_simulation(load_profiles)

# ========= 6. 可视化 =========
plot_bus_voltage_3phase(daily_results)

plt.show()




