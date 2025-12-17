 # 🚀 SoDistinct — Scalable Diffusion Simulation Framework

## 📌 Overview

**SoDistinct** is a modular and scalable framework for simulating **information diffusion processes on complex networks**.
It supports **multiple diffusion models**, **parallel and distributed execution**, and provides tools for **benchmarking, visualization, and performance analysis**.

The project is designed to study **how information spreads in social networks** and to compare the impact of **sequential, parallel, and distributed computing strategies** on large-scale simulations.

---

## 🎯 Objectives

The main objectives of this project are:

* Implement standard **diffusion models** used in network science
* Provide a **unified and extensible architecture** for graph-based simulations
* Compare execution strategies:

  * Sequential
  * Multithreading
  * Multiprocessing
  * Distributed computing (Dask)
* Demonstrate the **practical benefits of parallel and distributed programming**
* Enable reproducible experiments, benchmarks, and visual analytics

---

## 🧠 Diffusion Models Implemented

SoDistinct supports the following classical diffusion models:

| Model   | Description                                    |
| ------- | ---------------------------------------------- |
| **SI**  | Susceptible → Infected (no recovery)           |
| **SIR** | Susceptible → Infected → Recovered             |
| **IC**  | Independent Cascade (probabilistic activation) |
| **LT**  | Linear Threshold (influence-based activation)  |

Each model follows a **common interface**, ensuring consistency and extensibility.

---

## 🏗️ System Architecture

The system is organized into **clear and independent layers**:

```
User Input (Graph, Seeds, Parameters)
        ↓
Graph Loading & Validation (loader.py)
        ↓
Graph Abstraction (GraphWrapper)
        ↓
Diffusion Model (SI / SIR / IC / LT)
        ↓
Simulation Engine (engine.py)
        ↓
Orchestration Layer
   - Sequential
   - Parallel (Threads / Processes)
   - Async (asyncio)
   - Distributed (Dask)
        ↓
Results & Metrics
        ↓
Visualization & Dashboards
```

---

## 📂 Project Structure

```
SoDistinct/
├── src/sodistinct/
│   ├── io/              # Graph loading (sync & async)
│   ├── core/            # Graph abstraction, models, engine, metrics
│   ├── orchestrator/    # Parallel & async execution
│   ├── distributed/    # Dask backend
│   ├── viz/             # Visualization & dashboards
│   └── api/             # FastAPI endpoints
├── bench/               # Performance benchmarks
├── experiments/         # Reproducible experiments
├── tests/               # Unit, integration & performance tests
├── notebooks/           # Demonstrations
└── docs/                # Documentation & slides
```

---

## ⚙️ Key Components

### 🔹 Graph Abstraction

* `GraphWrapper` unifies graph manipulation
* Backend-independent (NetworkX today, extensible to others)
* Exposes only operations required by diffusion models

### 🔹 Simulation Engine

* Executes **one simulation step-by-step**
* Independent of the diffusion model
* Produces a standardized `SimulationResult`

### 🔹 Orchestrators

* **ParallelExecutor**: multiprocessing / multithreading
* **AsyncLocalOrchestrator**: asyncio-based execution
* **DaskBackend**: distributed execution on clusters

### 🔹 Distributed Computing

* Uses **Dask Distributed**
* Supports graph/model broadcasting
* Scales from local machine to cluster environments

---

## 📊 Metrics & Visualization

For each simulation, SoDistinct computes and exposes:

* Number of activated nodes
* Propagation speed (steps)
* Temporal evolution of diffusion
* Runtime and performance metrics

These results are exploited in:

* Dashboards (Streamlit)
* Plots (matplotlib)
* Benchmarks and performance comparisons

---

## 🧪 Benchmarking & Performance Analysis

The project includes benchmarks to compare:

* Sequential vs Multithreading vs Multiprocessing
* Local parallelism vs Distributed execution (Dask)
* Impact of graph size and model complexity

These experiments highlight:

* Python GIL limitations
* CPU-bound vs I/O-bound workloads
* Benefits of multiprocessing and distributed systems

---

## 🛠️ Installation

```bash
git clone https://github.com/USERNAME/SoDistinct.git
cd SoDistinct
pip install -r requirements.txt
```

(Optional for distributed mode)

```bash
pip install dask[distributed]
```

---

## ▶️ Example Usage

```python
from sodistinct.core.models import ICModel
from sodistinct.core.engine import run_simulation
from sodistinct.io.loader import load_graph

graph = load_graph("network.edgelist")
model = ICModel()
seed_set = [1, 5, 10]
params = {"p": 0.05}

result = run_simulation(model, graph, seed_set, params)
print(result.active_final)
```

---

## 🎓 Academic Context

This project was developed as part of an academic work focusing on:

* Network science
* Information diffusion
* Parallel and distributed programming
* Performance evaluation and scalability

---

## 📌 Conclusion

**SoDistinct** demonstrates how a well-designed architecture can combine:

* Graph theory
* Diffusion modeling
* Parallel & distributed computing

to efficiently simulate and analyze information propagation at scale.

---

## 👤 Author

**Nour Ben Brahim**
📧 Contact: nour.benbrahim@ymail.com
