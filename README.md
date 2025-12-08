# DRL-QBD: 深度强化学习 + QBD 理论的并联排队实验框架

本项目研究 **复杂到达过程 (MAP)** 驱动的并联系统 (parallel queues) 中，各种路由策略（包括 DRL 策略和经典规则）的性能，并借助
**QBD 理论解** 作为“地面真值”对比仿真结果。

核心组件：

- `env/parallel_queue_env.py`：基于 `simpy` + `gymnasium` 的并联 M/M/1 环境，外部到达为 MAP。
- `agents/`：多种路由策略的 DRL 实现（DQN, A2C, PPO, SAC）。
- `qbd/qbd_solver.py`：给定 MAP + 路由概率，计算理论平均队长向量 \(L_\text{theory}\)。
- `experiments/run_experiment.py`：串行版本的网格实验 (相关性 × 负载 × 策略)。
- `experiments/run_experiment_parallel.py`：细粒度并行版本，每个 (mode, corr, load, algo) 组合单独子进程。
- `utils/plotting.py`、`utils/export_latex.py`、`utils/persistence.py`：绘图、导出 LaTeX 表格、保存/重绘实验结果。

---

## 1. 环境与依赖

建议使用 Python 3.10+，并在虚拟环境中安装依赖。

```bash
cd /Volumes/extend/drl-queue/drl-qbd
python -m venv .venv
source .venv/bin/activate  # Windows 下用 .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 目前包含：

- `numpy`
- `matplotlib`
- `torch`
- `gymnasium`
- `simpy`
- `tqdm`

如需 GPU，请根据本机环境单独安装合适的 `torch` 版本。

---

## 2. 运行串行网格实验

串行版本的主入口是 `experiments/run_experiment.py`，会在 **多种到达模型 (map_mode)**、多种相关性、多种负载以及多种路由策略上跑完整网格，并：

- 保存所有结果到 `figures/<timestamp>/`；
- 将每个模式的结果打包为 JSON + `.npz`；
- 生成多种性能图和 LaTeX 表格。

运行方式：

```bash
cd /Volumes/extend/drl-queue/drl-qbd
python -m experiments.run_experiment
```

运行完成后，可在：

- `figures/<timestamp>/<map_mode>/` 下找到：
    - 利用率-平均队长曲线
    - 误差 (理论 vs 仿真) 随相关性/负载变化的图
    - DQN / JSQ 的误差热力图
    - `results_<map_mode>.tex` 等 LaTeX 表格

---

## 3. 运行细粒度并行实验

并行版本入口是 `experiments/run_experiment_parallel.py`：

- 使用 `ProcessPoolExecutor`；
- 每个 `(mode, corr, load, algo)` 组合单独子进程训练 + 评估；
- 主进程汇总成和串行版本兼容的结果结构；
- 同样输出图像、LaTeX 表格以及 JSON/NPZ 数据。

运行方式：

```bash
cd /Volumes/extend/drl-queue/drl-qbd
python -m experiments.run_experiment_parallel
```

部分参数（如 `workers`、`train_episodes`、相关性 / 负载网格）可以直接在 `run_experiment_parallel.py` 顶部修改。

**注意**：并行版本对资源占用较高，建议根据本机 CPU/GPU 情况调小 `workers` 或减少网格规模。

---

## 4. 从已保存数据重新绘图

如果已经跑过大规模实验并生成了 `experiment_data.json`，可以用 `experiments/replot_from_saved.py` 仅根据 JSON/NPZ
重绘所有图表，而不重新仿真：

```bash
cd /Volumes/extend/drl-queue/drl-qbd
python -m experiments.replot_from_saved \
  --json-path figures/<timestamp>/experiment_data.json
```

脚本会：

- 读取 JSON + 对应的 `per_mode/*/arrays.npz`；
- 重新生成所有性能图和 LaTeX 表格到同一目录（或通过参数指定的新目录）。

---

## 5. 目录结构概览

- `agents/`：DQN / A2C / PPO / SAC 等智能体与训练代码
- `env/`：并联队列环境与 MAP 生成 (`MAPSource`)
- `experiments/`：各类实验脚本（网格实验、ACF 分析、相关性可视化等）
- `qbd/`：QBD 理论求解器
- `utils/`：通用工具（设备选择、绘图、持久化、LaTeX 导出等）
- `figures/`：按时间戳归档的实验输出
- `training_figs/`：单次 DQN 训练的 Q-loss 曲线

---

## 6. 常见问题

1. **`ModuleNotFoundError: No module named 'scenario_design'`**
    - 请确保在项目根目录下运行模块：
      ```bash
      cd /Volumes/extend/drl-queue/drl-qbd
      python -m experiments.run_experiment
      ```
    - 或在 IDE/Notebook 中将项目根目录加入 `PYTHONPATH`。

2. **中文字体/Matplotlib 报告缺字形**
    - `utils/plotting.configure_matplotlib_for_chinese()` 会自动尝试选择系统中的 CJK 字体（如 Heiti /
      PingFang），若本地字体缺失，请根据注释手动替换为可用字体。

3. **并行实验占用资源过高**
    - 在 `experiments/run_experiment_parallel.py` 顶部调小 `workers`，或缩小 `corr_levels` / `load_factors` 网格规模。

---

## 7. 后续改进方向（TODO）

- 增加更多经典路由策略（如随机轮询的变体、基于响应时间的策略）。
- 支持参数化的成本函数，并在 QBD 理论中同步建模。
- 增加配置文件/命令行参数以便更灵活地控制实验网格。

如果你在使用过程中遇到问题，建议先查看对应脚本开头的参数设置注释；若仍有疑问，可以在 README 的基础上继续补充自己的实验记录与说明。

