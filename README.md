# 🧺 WashWise — Smart Hostel Laundry Scheduling

> **A Bio-Inspired Computing Mini-Project (50 Marks)**  
> *Smarter Laundry. Shorter Queues.*  
> **Topic:** Multi-Objective Genetic Algorithm Optimization for Shared Campus Resource Scheduling vs. First-Come, First-Served (FCFS) Baseline.

---

## 📑 Table of Contents

1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Key Features & Highlights](#2-key-features--highlights)
3. [Technology Stack](#3-technology-stack)
4. [System Architecture & Repository Structure](#4-system-architecture--repository-structure)
5. [Bio-Inspired Computing Algorithm Deep Dive](#5-bio-inspired-computing-algorithm-deep-dive)
   - [Chromosome Representation](#a-chromosome-representation)
   - [Initial Population & Heuristic Seeding](#b-initial-population--heuristic-seeding)
   - [Deterministic Repair Decoder (Hard Constraints)](#c-deterministic-repair-decoder-hard-constraints)
   - [Multi-Objective Fitness Function](#d-multi-objective-fitness-function)
   - [Selection, Crossover, Mutation & Elitism](#e-genetic-operators)
6. [Baseline Scheduler: First-Come, First-Served (FCFS)](#6-baseline-scheduler-first-come-first-served-fcfs)
7. [Streamlit Web Dashboard Walkthrough](#7-streamlit-web-dashboard-walkthrough)
8. [Installation & Execution Guide](#8-installation--execution-guide)
9. [Automated Verification & Test Results](#9-automated-verification--test-results)
10. [Academic Viva Voce Q&A Defense Guide](#10-academic-viva-voce-qa-defense-guide)
11. [Limitations & Future Roadmap](#11-limitations--future-roadmap)

---

## 1. Executive Summary & Problem Statement

In student residences and university hostels, a small cluster of washing machines (typically 4–8 machines) is shared among hundreds of residents. Without an intelligent scheduling mechanism, this leads to:
- **Severe peak-hour queuing** (especially during 09:00–12:00 and 18:00–21:00).
- **Physical conflicts and abandoned laundry loads**.
- **Washing machine underutilization** during off-peak hours alongside gridlock during peaks.
- **Inability to accommodate rigid student schedules** (e.g., classes, exams, lab shifts).
- **Zero priority awareness** (emergency medical laundry or students departing for travel are treated identically to casual laundry).

**WashWise** resolves this resource contention by framing laundry allocation as an **NP-Hard Resource-Constrained Multi-Objective Scheduling Problem**. It employs a genuine **Genetic Algorithm (GA)** to evolve optimal machine and time-slot allocations while strictly enforcing physical and temporal hard constraints.

---

## 2. Key Features & Highlights

- 🧬 **Genuine Bio-Inspired Metaheuristic:** Evolving chromosomes using tournament selection, uniform crossover, multi-operator mutation, and elitism.
- 🛡️ **Guaranteed Constraint Integrity:** Hard constraint repair decoder guarantees **0 machine overlaps**, **0 availability breaches**, and **0 closing-hour violations**.
- ⚖️ **Rigorous FCFS Comparison:** Unbiased, side-by-side benchmark comparison against a realistic First-Come, First-Served baseline.
- 🗓️ **Interactive Plotly Gantt Timeline:** One row per washing machine with priority-coded colored blocks, hover tooltips, and time axes.
- ➕ **Dynamic Request Management:** Interactive Streamlit form allowing students to submit new laundry requests with immediate re-optimization.
- 📈 **Real-Time Evolutionary Convergence Curves:** Live Plotly line charts displaying Best Fitness, Average Population Fitness, and Waiting Time reduction across generations.
- 📥 **Export Center:** One-click CSV downloads for final schedules, comparative metrics, and raw request datasets.
- 📚 **Comprehensive 16-Section Academic Viva Guide:** Built directly into the dashboard for classroom demonstration and oral examination.

---

## 3. Technology Stack

- **Language:** Python 3 (Tested on Python 3.13 & 3.14)
- **Web Interface:** Streamlit (Native components, custom CSS cards, badges, and responsive tabs)
- **Data Engineering:** Pandas & NumPy
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Testing Framework:** Pytest & Python Standard Library `unittest`
- **External Dependencies:** Zero paid APIs, no external databases, 100% self-contained and reproducible.

---

## 4. System Architecture & Repository Structure

```
WashWise/
├── app.py                      # Main Streamlit web application & interactive UI
├── scheduler.py                # Core Bio-Inspired GA engine, FCFS baseline, & metrics
├── requirements.txt            # Minimal, exact Python dependencies
├── README.md                   # Comprehensive documentation & Viva defense manual
├── data/
│   └── sample_requests.csv     # 25 realistic, reproducible student laundry requests
├── tests/
│   └── test_scheduler.py       # 10 automated test cases (constraints, decoder, edge cases)
└── .streamlit/
    ├── config.toml             # Streamlit server & theme configuration
    └── credentials.toml        # Local headless credential bypass
```

---

## 5. Bio-Inspired Computing Algorithm Deep Dive

### A. Chromosome Representation
Each candidate schedule is represented as an array of $N$ genes corresponding to the $N$ student requests. Each gene is a triplet:
$$g_i = \langle m_i, s_i, \delta_i \rangle$$

- $m_i \in \{0, 1, \dots, K-1\}$: Assigned washing machine index ($K$ machines).
- $s_i \in [e_i, l_i - d_i]$: Preferred starting time slot within the student's valid availability window.
- $\delta_i \in [0.0, 1.0]$: Dispatch priority offset, used to evolve the sequence in which requests claim machine slots.

The effective dispatch priority for request $i$ is calculated as:
$$P_i = (\text{PriorityWeight}_i \times 10.0) + \delta_i$$
where weights are **Urgent = 4, High = 3, Normal = 2, Low = 1**. This guarantees that urgent requests are evaluated ahead of high, high ahead of normal, while the GA evolves optimal tie-breaking within tiers!

### B. Initial Population & Heuristic Seeding
Rather than starting from purely random noise, the initial population of size $P$ (default 40) is seeded with domain heuristics:
1. **Earliest Arrival Heuristic:** Requests assigned earliest ready slot on round-robin machines.
2. **Priority-First Heuristic:** Urgent/High requests placed at earliest times on dedicated machines.
3. **Staggered Spread Heuristic:** Offsets start times to distribute peak demand.
4. **Natural Student Arrival Distribution:** Remainder ($P - 3$) generated with power-law distributions skewed towards students' earliest availability ($s = e_i + \text{int}(r^{1.8} \times (l_i - d_i - e_i))$).

### C. Deterministic Repair Decoder (Hard Constraints)
To ensure the system never outputs an invalid schedule:
1. Sort requests by effective score $P_i$ descending.
2. **Primary Attempt:** Test if gene machine $m_i$ is completely unoccupied from slot $s_i$ to $s_i + d_i$.
3. **Machine Repair:** If occupied, search $m_i$ for the closest unoccupied window $[s', s' + d_i]$ within the student's availability $[e_i, l_i]$.
4. **Alternative Machine Repair:** If $m_i$ has no valid slot, search all other machines $m \ne m_i$ for the closest feasible slot.
5. **Unscheduled Flag:** If all machines are saturated during $[e_i, l_i]$, mark request as **Unscheduled** with an explicit reason ("No available slot across machines").

### D. Multi-Objective Fitness Function
The objective function maximizes throughput and efficiency while heavily penalizing delays on high-priority requests:

$$\mathcal{F} = \mathcal{C}_{base} + \mathcal{R}_{sched} + \mathcal{R}_{util} - \mathcal{P}_{wait} - \mathcal{P}_{unsched} - \mathcal{P}_{disp} - \mathcal{P}_{frag}$$

- $\mathcal{C}_{base} = 100,000$: Ensures positive values.
- $\mathcal{R}_{sched} = \sum_{j \in \mathcal{S}} (w_j^{1.5} \times 800 + d_j \times 1.5)$: Reward for scheduling requests.
- $\mathcal{P}_{wait} = \sum_{j \in \mathcal{S}} (w_j^2 \times \text{wait\_mins}_j \times 2.0)$: Quadratic penalty for student waiting time ($16\times$ higher for Urgent than Low!).
- $\mathcal{P}_{unsched} = \sum_{j \in \mathcal{U}} (w_j^2 \times 5,000)$: Heavy penalty for unscheduled requests.
- $\mathcal{R}_{util} = \text{Utilization}_{\%} \times 20.0$: Reward for overall machine utilization.
- $\mathcal{P}_{disp}$: Penalty for displacement away from chromosome target machine and preferred slot.
- $\mathcal{P}_{frag}$: Penalty for fragmented 15–30 min idle gaps between bookings.

### E. Genetic Operators
- **Tournament Selection ($k=3$):** Randomly draws 3 chromosomes and chooses the fittest individual.
- **Uniform Crossover ($P_c = 0.85$):** Offspring inherit each request gene from parent 1 or parent 2 with 50% probability.
- **Multi-Operator Mutation ($P_m = 0.15$):**
  - Reassigns machine index.
  - Perturbs preferred slot by $\pm 1..3$ slots (or uniform re-roll).
  - Mutates dispatch priority offset with Gaussian noise $\mathcal{N}(0, 0.15)$.
- **Elitism ($E = 2$):** Preserves the 2 fittest individuals directly into the next generation, guaranteeing monotonically non-decreasing best fitness.

---

## 6. Baseline Scheduler: First-Come, First-Served (FCFS)

To rigorously benchmark the Genetic Algorithm, WashWise includes a deterministic **First-Come, First-Served (FCFS)** baseline:
- Processes requests in strict arrival order (arrival timestamp / request index).
- Scans all available machines for the earliest possible free time slot $t \ge e_i$ where $t + d_i \le l_i$ and $t + d_i \le T_{max}$.
- Allocates the earliest slot found. If no machine can accommodate the request within its availability window, it is marked as Unscheduled.

**Why GA Outperforms FCFS in Contested Scenarios:**
FCFS is a greedy heuristic. If an early-arriving low-priority student requests a 2-hour wash during a peak window (e.g. 10:00–12:00), FCFS locks machine 1. A subsequent urgent student available only from 10:30–11:30 may find all machines blocked and get rejected! GA evaluates the global combination of requests, repacking flexible tasks and granting urgent tasks priority slots.

---

## 7. Streamlit Web Dashboard Walkthrough

The application interface consists of 7 modular sections:
- **📊 Overview & KPIs:** Real-time KPI cards (Total Requests, Scheduled, Average Waiting Time, Utilization, Unscheduled Requests), Hard Constraint Verification badge, and machine workload balance bars.
- **📋 Laundry Requests:** Comprehensive searchable table of all requests with priority badges, machine allocations, and an expandable form to add new requests dynamically.
- **🗓️ Machine Gantt Timeline:** Interactive Plotly Gantt chart with one horizontal row per machine, priority color coding, hover tooltips, and an audit table of unscheduled requests.
- **⚖️ Algorithm Comparison:** Side-by-side metric comparison table, grouped bar charts comparing average wait times across priority tiers, and an automatically generated academic interpretation.
- **🧬 GA Evolutionary Visuals:** Live Plotly convergence curves displaying Best Fitness vs. Population Average across generations, wait time reduction, and explanation cards.
- **📥 Export Center:** One-click CSV download buttons for Final Schedule, Comparative Metrics, and Raw Requests.
- **📚 Academic Project Report & Viva:** Complete 16-section technical documentation, mathematical formulas, and live metrics formatted for 50-mark viva evaluation.

---

## 8. Installation & Execution Guide

### Prerequisites
- Python 3.9+ (Python 3.13 or 3.14 recommended)
- A modern web browser

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or `python3 -m pip install -r requirements.txt`)*

### Step 2: Run the Streamlit Application
```bash
python3 -m streamlit run app.py
```

### Step 3: Access the Web Dashboard
Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 9. Automated Verification & Test Results

The project includes an automated test suite in `tests/test_scheduler.py` covering all hard constraints, decoder logic, and edge cases.

### Running Automated Tests
```bash
python3 -m pytest tests/test_scheduler.py -v
```

### Actual Test Execution Output:
```
============================= test session starts ==============================
platform darwin -- Python 3.14.5, pytest-9.2.0, pluggy-1.7.0
rootdir: /Users/shivaniagrawal/Documents/antigravity/noble-rutherford

tests/test_scheduler.py::test_time_conversions PASSED                    [ 10%]
tests/test_scheduler.py::test_request_preparation PASSED                 [ 20%]
tests/test_scheduler.py::test_no_machine_overlaps_in_fcfs PASSED         [ 30%]
tests/test_scheduler.py::test_no_machine_overlaps_in_ga PASSED           [ 40%]
tests/test_scheduler.py::test_availability_and_closing_constraints_respected PASSED [ 50%]
tests/test_scheduler.py::test_impossible_window_marks_unscheduled PASSED [ 60%]
tests/test_scheduler.py::test_empty_schedule_handling PASSED             [ 70%]
tests/test_scheduler.py::test_csv_persistence_roundtrip PASSED           [ 80%]
tests/test_scheduler.py::test_ga_reproducibility PASSED                  [ 90%]
tests/test_scheduler.py::test_comparison_table_structure PASSED          [100%]

======================== 10 passed, 1 warning in 0.78s =========================
```

---

## 10. Academic Viva Voce Q&A Defense Guide

### Q1: What makes this problem suitable for a Genetic Algorithm rather than a greedy heuristic?
**Answer:** Laundry scheduling with distinct availability windows, multi-priority levels, and variable durations is a Resource-Constrained Multi-Machine Scheduling problem, which is NP-Hard. Greedy heuristics like FCFS make irreversible early allocations that cause bottlenecks during peak hours. A Genetic Algorithm searches the global combinatorial space, exploring alternative machine-time assignments to simultaneously optimize wait time, utilization, and priority compliance.

### Q2: How do you guarantee that no two laundry sessions overlap on the same washing machine?
**Answer:** Hard constraints are enforced by our **Deterministic Repair Decoder**. The decoder maintains an occupancy boolean grid $\text{occupancy}[m][t]$. A request is only allocated if all consecutive time slots for its duration are strictly free. If a collision is detected on the gene's target machine, the decoder searches for the nearest feasible conflict-free slot. If no feasible slot exists across any machine within the student's availability window, the request is flagged as Unscheduled rather than producing an overlapping schedule.

### Q3: Why does your fitness function use a quadratic wait penalty for priority tiers?
**Answer:** By scaling the waiting time penalty as $(\text{PriorityWeight}^2 \times \text{wait\_time})$, an Urgent request ($w=4 \implies 16$) penalizes delay 16 times more heavily than a Low priority request ($w=1 \implies 1$). This creates strong evolutionary selection pressure that forces the GA to minimize delay for students with emergency or urgent laundry needs.

### Q4: How is Elitism implemented, and why is it essential?
**Answer:** Elitism directly copies the top $E=2$ best individuals from generation $g$ into generation $g+1$ without crossover or mutation. In Genetic Algorithms, stochastic crossover and mutation can inadvertently disrupt high-quality schemata. Elitism mathematically guarantees that the best fitness score found so far is monotonically non-decreasing over time.

### Q5: What does the Chromosome represent in your implementation?
**Answer:** Each chromosome is an array of $N$ request genes. Each gene contains:
1. Target washing machine index $m \in [0, K-1]$.
2. Preferred starting slot index $s \in [e_i, l_i - d_i]$.
3. Dispatch priority score offset $\delta \in [0.0, 1.0]$.
When decoded, requests are dispatched in descending order of effective priority score, mapping preferred slots onto conflict-free machine schedules.

---

## 11. Limitations & Future Roadmap

### Current Limitations:
- Assumes homogeneous washing machines (identical cycle speeds and capacities).
- Assumes deterministic wash durations without accounting for machine mechanical faults or student collection delays.

### Future Enhancements:
- 📱 **IoT Smart Plug Integration:** Connect smart current-monitoring plugs to detect cycle completion and trigger automated push/SMS notifications.
- ⚡ **Dynamic Machine Heterogeneity:** Differentiate between heavy-duty commercial washers, quick-washers (30m), and dedicated dryers.
- 🔄 **Stochastic Online Rescheduling:** Real-time rescheduling using Co-evolutionary Genetic Algorithms to dynamically adjust for machine downtime or late student arrivals.

---

**Developed for Bio-Inspired Computing Examination (50 Marks)**  
*WashWise — Smarter Laundry. Shorter Queues.*
