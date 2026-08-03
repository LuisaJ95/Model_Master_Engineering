# Food Donation Supply Chain Optimization

A Mixed-Integer Linear Programming (MILP) model that maximizes the net benefit of donating surplus food to food banks, balancing tax incentives, transportation costs, and waste penalties across a multi-period planning horizon.

**Author**: Luisa Jimenez — candidate for the Master's Degree in Engineering, Universidad de Antioquia.  
Developed as part of her master's thesis in operations research / supply chain management.

---

## Problem Statement

Food companies (manufacturers, distributors, clients) hold surplus or near-expiry inventory that can either be donated to food banks or wasted. This model helps companies decide:

- **How much** to donate per period
- **Which route** to use — direct delivery or indirect pickup via a distribution center (CEDI)
- **Own fleet vs. outsourcing** for transportation
- **When** to deliver, respecting product expiration dates and food bank capacity

The objective is to **maximize net financial benefit**: tax deductions from donations minus transportation and outsourcing costs minus penalties for wasted food.

---

## Mathematical Model

### Sets
| Symbol | Description |
|--------|-------------|
| R_direct | Requirements with direct routes (Manufacturing / Distribution Centers → Food Bank) |
| R_indirect | Requirements with indirect routes (Client → CEDI → Food Bank) |
| T | Planning periods (days) |
| B | Food banks |
| P | Product types |

### Decision Variables
| Variable | Description |
|----------|-------------|
| y_deliv[r, t] | Quantity delivered for requirement r on period t (kg) |
| x_out[r, t] | Quantity outsourced for requirement r on period t (kg) |
| y_pickup[r, t] | Quantity picked up on indirect route on period t (kg) |
| w[r] | Quantity wasted for requirement r (kg) |

### Objective Function

Maximize **Z = Tax Benefits (products + transport) − Fleet Costs − Outsourcing Costs − Waste Penalties**

$$Z = \beta_1 \cdot c_{prod} \cdot Q_{donated} + \beta_1 \cdot c_{trans} \cdot d \cdot (y_{deliv} + \alpha \cdot x_{out}) - c_{trans} \cdot d \cdot y_{deliv} - \alpha \cdot c_{trans} \cdot d \cdot x_{out} - \pi \cdot w$$

### Constraints
1. **Flow conservation (direct)** — all supply is delivered, outsourced, or wasted
2. **Flow conservation (indirect)** — same, for client-origin requirements
3. **Food bank capacity** — daily intake per bank cannot exceed its capacity
4. **Transportation capacity** — own fleet usage per period is bounded
5. **Synchronization (indirect)** — pickup on day t equals delivery on day t+1

### Key Parameters
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Fiscal incentive rate | β₁ | Tax deduction fraction (0–1) |
| Waste penalty | π | Cost per kg of wasted food |
| Transport cost | c_trans | Cost per km·kg |
| Outsourcing multiplier | α | Ratio of outsourcing cost to own-fleet cost |
| Product value | c_prod | $/kg by product type |

---

## Repository Structure

```
├── model_final.py          # Core optimization model (FoodDonationOptimizer class)
├── Inference_Model.py      # Batch analysis over 12 monthly instances
├── model_validation.py     # Scenario & sensitivity validation
├── sensitivity_analysis.py # Univariate sensitivity analysis with elasticity
├── Exp_1.py                # Experiment 1 — 2² Full Factorial Design
├── Exp_2.py                # Experiment 2 — Response Surface Methodology (CCD)
└── Exp_3.py                # Experiment 3 — 3×3 Factorial Design
```

### Data Files (not included in repo)
- `instance_distance_exp.json` — base test instance
- `instance_distance_2024_08.json` … `instance_distance_2025_07.json` — 12 monthly instances

---

## Scripts

### `model_final.py` — Core Model

Contains the `FoodDonationOptimizer` class. Solves a single instance.

```python
optimizer = FoodDonationOptimizer("instance_distance_exp.json")
optimizer.load_data_from_json()
optimizer.build_model()
optimizer.optimize()
optimizer.display_results()
optimizer.export_solution("solution.json")
```

**Gurobi settings**: 10-minute time limit, MIPFocus=1 (feasibility), aggressive presolve, all CPU threads.

**Outputs**: `solution.json`, `model.lp`

---

### `Inference_Model.py` — Multi-Instance Analysis

Runs the optimizer across all 12 monthly instances and produces comparative statistics.

```python
run_multi_instance_analysis()
print_comparative_analysis()
generate_markdown_report()
correlation_analysis()
trend_analysis()
efficiency_analysis()
generate_insights_report()
export_summary_for_presentation()
```

**Outputs**:
- `multi_instance_results.csv` — monthly metrics
- `multi_instance_summary.json` — aggregated summary
- `analysis_report.md` — full markdown report
- `correlation_matrix.csv` — Pearson correlations
- `insights_report.json` — insights and recommendations
- `presentation_summary.csv / .xlsx` — clean export for presentations

---

### `model_validation.py` — Validation & Scenario Testing

**Scenario 1 — Insufficient Infrastructure**: food bank capacity reduced to 10%, fleet to 15%, outsourcing cost 5×. Verifies the model responds correctly (waste increases, outsourcing stays minimal).

**Sensitivity Analysis (12 runs)**: Crosses three parameters (β, π, c_trans) at low / standard / high levels to confirm model behavior under all combinations.

**Outputs**: `scenario1_comparison.png`, `scenario1_results.json`

---

### `sensitivity_analysis.py` — Univariate Sensitivity & Elasticity

Varies each of six parameters individually across seven multipliers (0.5 – 1.5) and computes point elasticities.

**Parameters**: β (incentive), π (penalty), c_trans (transport cost), α (outsourcing), food bank capacity, fleet capacity.

**Elasticity classification**:
| Class | |ε| | Interpretation |
|-------|------|----------------|
| High | > 1.0 | Critical — precise estimation required |
| Medium | 0.5–1.0 | Important |
| Low | < 0.5 | Robust to estimation error |

**Outputs**: `sensitivity_analysis_results.xlsx`, `sensitivity_analysis_plots.png`, `elasticity_ranking.png`

---

### `Exp_1.py` — 2² Full Factorial Design

Tests main effects and interaction between **transportation cost** and **waste penalty** (2 factors × 2 levels × 3 replicates = 12 runs). Analyzed via ANOVA.

---

### `Exp_1.2.py` — Waste Penalty Sensitivity Analysis

Complementary experiment to the 2² factorial design. Analyzes the functional relationship between waste penalty and response variables (net benefit and food waste) while keeping transport cost fixed at its optimal level. Identifies critical thresholds where the system's behavior changes (1 factor × 15 levels = 15 runs). Analyzed via threshold identification, correlation analysis, and trade-off assessment.

---

### `Exp_2.py` — 3² Full Factorial Design

Tests main effects and interaction between fleet capacity and indirect route percentage (2 factors × 3 levels × 3 replicates = 27 runs). Analyzed via main effects, interactions, and response surface visualization.

---

### `Exp_3.py` — 3×3 Factorial Design

Tests main effects and interaction between tax incentive (β) and fleet capacity (2 factors × 3 levels each = 9 runs). Analyzed via interaction plots and eta-squared effect sizes.

---

## Requirements

```
gurobipy       # Gurobi optimizer (license required)
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
openpyxl       # for Excel export
```

A valid **Gurobi license** is required. Academic licenses are available free of charge at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Analysis Workflow

```
JSON instance data
       │
       ▼
model_final.py          ← single-instance optimization
       │
       ├── Inference_Model.py      ← 12 monthly instances → CSV / Excel reports
       ├── model_validation.py     ← scenario & sensitivity validation
       └── sensitivity_analysis.py ← univariate elasticity analysis
                │
                ▼
 Exp_1 / Exp_1.2 / Exp_2 / Exp_3      ← designed experiments (factorial, RSM)
                │
                ▼
       Statistical analysis & visualizations
```

---

## Instance Data Format

Each JSON instance contains:

```json
{
  "requirements": [
    {
      "id": "req_001",
      "origin": "Plant_A",
      "destination": "FoodBank_1",
      "origin_type": "Manufacturing",
      "product": "canned_goods",
      "quantity": 500,
      "release_date": 1,
      "expiration_date": 5,
      "distance": 12.3,
      "dist_indirect": 18.7
    }
  ],
  "planning_horizon": {"start": 1, "end": 14},
  "sets": {"food_banks": [...], "products": [...]},
  "parameters": {
    "beta_1": 0.35,
    "pi_penalty": 0.5,
    "c_trans": 0.02,
    "alpha_outsource": 1.5,
    "product_costs": {"canned_goods": 1.2},
    "food_bank_capacity": {"FoodBank_1": 2000},
    "transport_capacity_per_period": 8000
  }
}
```

`origin_type` determines the routing strategy: `Manufacturing` and `DistributionCenter` use direct routes; `Client` uses the indirect two-day route through a CEDI.

---

## Metrics Reported

| Metric | Description |
|--------|-------------|
| Objective value | Net financial benefit ($) |
| Total donated | kg delivered to food banks |
| Total wasted | kg that could not be donated |
| Total outsourced | kg moved via third-party fleet |
| Donation rate | donated / available (%) |
| Waste rate | wasted / available (%) |
| Fleet utilization | own-fleet usage vs. capacity (%) |
| Cost per kg donated | efficiency metric |
| Solve time | Gurobi wall-clock time (s) |
