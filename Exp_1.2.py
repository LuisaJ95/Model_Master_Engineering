"""
EXPERIMENT 1.2: Sensitivity Analysis - Waste Penalty
Complementary Experiment to the 2² Factorial

Objective: Analyze the functional relationship between waste penalty and 
response variables (benefit and waste) while keeping transport cost fixed 
at its optimal level.

Research question: How do net benefit and food waste behave as the penalty 
gradually increases? Is there a critical threshold where the system changes 
its behavior?
"""

import gurobipy as gp
from gurobipy import GRB
import json
import numpy as np
import pandas as pd
import time
import os
from copy import deepcopy

# Try to import matplotlib for plots
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib is not installed. No plots will be generated automatically.")


class FoodDonationOptimizer:
    """
    Optimizer for food donation network with direct and indirect routes.
    """
    
    def __init__(self, env):
        """Initialize optimizer with Gurobi environment."""
        self.env = env
        self.model = None
        
        # Data attributes
        self.requirements = []
        self.requirements_dict = {}
        self.R_direct = []
        self.R_indirect = []
        self.T = []
        self.B = []
        self.P = []
        
        # Parameters
        self.beta = 0
        self.pi = 0
        self.c_trans = 0
        self.alpha = 0
        self.product_costs = {}
        self.food_bank_capacity = {}
        self.transport_capacity = {}
        
        # Decision variables
        self.y_deliv = {}
        self.x_out = {}
        self.y_pickup = {}
        self.w = {}
        
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and dispose model."""
        if self.model is not None:
            self.model.dispose()
        return False
    
    def load_data_from_json(self, json_file):
        """Load instance data from JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.requirements = data['requirements']
        
        for req in self.requirements:
            self.requirements_dict[req['id']] = req
        
        for req in self.requirements:
            req_id = req['id']
            origin_type = req['origin_type']
            
            if origin_type in ['Manufacturing', 'DistributionCenter']:
                self.R_direct.append(req_id)
            else:
                self.R_indirect.append(req_id)
        
        self.T = list(range(
            data['planning_horizon']['start_period'],
            data['planning_horizon']['end_period'] + 1
        ))
        self.B = data['sets']['food_banks']
        self.P = data['sets']['products']
        
        params = data['parameters']
        self.beta = params['beta_1']
        self.pi = params['pi_penalty']
        self.c_trans = params['c_trans']
        self.alpha = params['alpha_outsource']
        self.product_costs = params['product_costs']
        self.food_bank_capacity = params['food_bank_capacity']
        
        trans_cap = params['transport_capacity_per_period']
        self.transport_capacity = {t: trans_cap for t in self.T}
    
    def _get_requirement_by_id(self, req_id):
        """Helper function to get requirement data by ID."""
        return self.requirements_dict.get(req_id)
    
    def build_model(self):
        """Build the optimization model."""
        self.model = gp.Model("FoodDonationOptimization", env=self.env)
        self._create_variables()
        self._set_objective()
        self._add_constraints()
        self.model.update()
    
    def _create_variables(self):
        """Create all decision variables."""
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                self.y_deliv[r, t] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS, name=f"y_deliv_D_{r}_{t}"
                )
                self.x_out[r, t] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS, name=f"x_out_{r}_{t}"
                )
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r):
                self.y_pickup[r, t] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS, name=f"y_pickup_{r}_{t}"
                )
            
            for t in range(l_r + 1, e_r + 1):
                self.y_deliv[r, t] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS, name=f"y_deliv_I_{r}_{t}"
                )
        
        for r in self.R_direct + self.R_indirect:
            self.w[r] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"w_{r}")
    
    def _set_objective(self):
        """Set the objective function to maximize net benefit."""
        product_benefit_terms = []
        transport_benefit_terms = []
        transport_cost_terms = []
        outsource_cost_terms = []
        
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                product_benefit_terms.append(self.beta * c_prod * self.y_deliv[r, t])
                product_benefit_terms.append(self.beta * c_prod * self.x_out[r, t])
                transport_benefit_terms.append(self.beta * self.c_trans * dist * self.y_deliv[r, t])
                transport_benefit_terms.append(self.beta * self.c_trans * dist * self.alpha * self.x_out[r, t])
                transport_cost_terms.append(self.c_trans * dist * self.y_deliv[r, t])
                outsource_cost_terms.append(self.alpha * self.c_trans * dist * self.x_out[r, t])
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                product_benefit_terms.append(self.beta * c_prod * self.y_deliv[r, t])
                transport_benefit_terms.append(self.beta * self.c_trans * dist_indir * self.y_deliv[r, t])
                transport_cost_terms.append(self.c_trans * dist_indir * self.y_deliv[r, t])
        
        waste_terms = [self.pi * self.w[r] for r in self.R_direct + self.R_indirect]
        
        obj = (
            gp.quicksum(product_benefit_terms) +
            gp.quicksum(transport_benefit_terms) -
            gp.quicksum(transport_cost_terms) -
            gp.quicksum(outsource_cost_terms) -
            gp.quicksum(waste_terms)
        )
        
        self.model.setObjective(obj, GRB.MAXIMIZE)
    
    def _add_constraints(self):
        """Add all constraints to the model."""
        # Flow conservation for DIRECT requirements
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            q_r = req['quantity']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            self.model.addConstr(
                gp.quicksum(self.y_deliv[r, t] + self.x_out[r, t] for t in range(l_r, e_r + 1)) + self.w[r] == q_r,
                name=f"flow_direct_{r}"
            )
        
        # Flow conservation for INDIRECT requirements
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            q_r = req['quantity']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            self.model.addConstr(
                gp.quicksum(self.y_deliv[r, t] for t in range(l_r + 1, e_r + 1)) + self.w[r] == q_r,
                name=f"flow_indirect_{r}"
            )
        
        # Food bank receiving capacity
        direct_reqs_by_bank = {j: [] for j in self.B}
        indirect_reqs_by_bank = {j: [] for j in self.B}
        
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dest = req['destination']
            if dest in direct_reqs_by_bank:
                direct_reqs_by_bank[dest].append(r)
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            dest = req['destination']
            if dest in indirect_reqs_by_bank:
                indirect_reqs_by_bank[dest].append(r)
        
        for j in self.B:
            cap_j = self.food_bank_capacity.get(j, 0)
            
            for t in self.T:
                deliveries = []
                
                for r in direct_reqs_by_bank[j]:
                    req = self._get_requirement_by_id(r)
                    l_r = req['release_date']
                    e_r = req['expiration_date']
                    if l_r <= t <= e_r:
                        deliveries.append(self.y_deliv[r, t] + self.x_out[r, t])
                
                for r in indirect_reqs_by_bank[j]:
                    req = self._get_requirement_by_id(r)
                    l_r = req['release_date']
                    e_r = req['expiration_date']
                    if l_r + 1 <= t <= e_r:
                        deliveries.append(self.y_deliv[r, t])
                
                if deliveries:
                    self.model.addConstr(
                        gp.quicksum(deliveries) <= cap_j,
                        name=f"foodbank_cap_{j}_{t}"
                    )
        
        # Transportation capacity
        for t in self.T:
            cap_trans = self.transport_capacity[t]
            fleet_usage = []
            
            for r in self.R_direct:
                req = self._get_requirement_by_id(r)
                l_r = req['release_date']
                e_r = req['expiration_date']
                if l_r <= t <= e_r:
                    fleet_usage.append(self.y_deliv[r, t])
            
            for r in self.R_indirect:
                req = self._get_requirement_by_id(r)
                l_r = req['release_date']
                e_r = req['expiration_date']
                if l_r <= t < e_r:
                    fleet_usage.append(self.y_pickup[r, t])
            
            for r in self.R_indirect:
                req = self._get_requirement_by_id(r)
                l_r = req['release_date']
                e_r = req['expiration_date']
                if l_r + 1 <= t <= e_r:
                    fleet_usage.append(self.y_deliv[r, t])
            
            if fleet_usage:
                self.model.addConstr(
                    gp.quicksum(fleet_usage) <= cap_trans,
                    name=f"transport_cap_{t}"
                )
        
        # Synchronization for INDIRECT requirements
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r):
                self.model.addConstr(
                    self.y_pickup[r, t] == self.y_deliv[r, t + 1],
                    name=f"sync_{r}_{t}"
                )
    
    def optimize(self):
        """Solve the optimization model."""
        self.model.optimize()
        return self.model.Status
    
    def get_results(self):
        """Extract results from the solved model."""
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        
        if self.model.SolCount == 0:
            return None
        
        total_waste = sum(self.w[r].X for r in self.R_direct + self.R_indirect)
        total_donated_own = sum(
            self.y_deliv[r, t].X 
            for r in self.R_direct 
            for t in range(
                self._get_requirement_by_id(r)['release_date'],
                self._get_requirement_by_id(r)['expiration_date'] + 1
            )
        )
        total_donated_own += sum(
            self.y_deliv[r, t].X 
            for r in self.R_indirect 
            for t in range(
                self._get_requirement_by_id(r)['release_date'] + 1,
                self._get_requirement_by_id(r)['expiration_date'] + 1
            )
        )
        total_outsourced = sum(
            self.x_out[r, t].X 
            for r in self.R_direct 
            for t in range(
                self._get_requirement_by_id(r)['release_date'],
                self._get_requirement_by_id(r)['expiration_date'] + 1
            )
        )
        
        return {
            'objective_value': self.model.ObjVal,
            'total_waste': total_waste,
            'total_donated_own': total_donated_own,
            'total_outsourced': total_outsourced,
            'total_donated': total_donated_own + total_outsourced,
            'status': self.model.Status
        }


class SensitivityAnalyzer:
    """
    Class to run sensitivity analysis for one factor.
    """
    
    def __init__(self, instance_file, output_dir='sensitivity_results'):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            instance_file: Path to the JSON instance file
            output_dir: Directory to save results
        """
        self.instance_file = instance_file
        self.output_dir = output_dir
        self.results = []
        self.results_df = None
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def run_sensitivity_analysis(
        self, 
        fixed_penalty=50,          # Fixed penalty (does not vary)
        base_c_trans=0.138,        # Base transport cost
        c_trans_multipliers=None,  # List of multipliers for c_trans
        time_limit=300
    ):
        """
        Run sensitivity analysis varying transport cost.
        Waste penalty is kept fixed.
        """
        if c_trans_multipliers is None:
            c_trans_multipliers = [1 + i*0.2 for i in range(11)]  # 1, 1.2, ..., 3

        self.c_trans_multipliers = c_trans_multipliers
        self.base_c_trans = base_c_trans
        self.fixed_penalty = fixed_penalty

        print("=" * 80)
        print("EXPERIMENT 2: SENSITIVITY ANALYSIS - TRANSPORT COST")
        print("=" * 80)

        print(f"\nExperiment Configuration:")
        print(f"  • Fixed penalty: {fixed_penalty}")
        print(f"  • Base transport cost: {base_c_trans} $/kg-km")
        print(f"  • Cost multipliers: {c_trans_multipliers}")
        print(f"  • Actual transport costs: {[base_c_trans*m for m in c_trans_multipliers]}")
        print(f"  • Time limit: {time_limit}s per run")

        for idx, multiplier in enumerate(c_trans_multipliers, 1):
            c_trans_value = base_c_trans * multiplier
            print(f"\n{'-'*80}")
            print(f"Run {idx}/{len(c_trans_multipliers)}: Multiplier = {multiplier:.1f}  →  Cost = ${c_trans_value:,.2f}")
            print(f"{'-'*80}")

            start_time = time.time()
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.setParam('TimeLimit', time_limit)
                env.start()
                with FoodDonationOptimizer(env) as optimizer:
                    try:
                        optimizer.load_data_from_json(self.instance_file)
                        # Assign parameters: c_trans variable, pi fixed
                        optimizer.c_trans = c_trans_value
                        optimizer.pi = fixed_penalty
                        optimizer.build_model()
                        status = optimizer.optimize()
                        results = optimizer.get_results()
                        elapsed_time = time.time() - start_time

                        if results is not None:
                            result_record = {
                                'run': idx,
                                'multiplier': multiplier,
                                'c_trans': c_trans_value,
                                'penalty_fixed': fixed_penalty,
                                'benefit': results['objective_value'],
                                'waste': results['total_waste'],
                                'donated_own': results['total_donated_own'],
                                'outsourced': results['total_outsourced'],
                                'total_donated': results['total_donated'],
                                'solve_time': elapsed_time,
                                'status': 'Optimal' if status == GRB.OPTIMAL else 'Time_Limit'
                            }
                            self.results.append(result_record)
                            print(f"  ✓ Benefit: ${results['objective_value']:,.2f}")
                            print(f"  ✓ Waste: {results['total_waste']:,.2f} kg")
                            print(f"  ✓ Donated (own fleet): {results['total_donated_own']:,.2f} kg")
                            print(f"  ✓ Outsourced: {results['total_outsourced']:,.2f} kg")
                            print(f"  ✓ Time: {elapsed_time:.2f}s")
                        else:
                            print(f"  ✗ No feasible solution found")
                    except Exception as e:
                        print(f"  ✗ Error: {e}")

        self.results_df = pd.DataFrame(self.results)
        print(f"\n{'='*80}")
        print("SENSITIVITY ANALYSIS COMPLETED")
        print(f"{'='*80}")
        print(f"Successful runs: {len(self.results)}/{len(c_trans_multipliers)}")
    
    def analyze_results(self):
        """
        Analyze results from the sensitivity analysis (varying c_trans).
        """
        if self.results_df is None or len(self.results_df) == 0:
            print("No results to analyze")
            return

        print(f"\n{'='*80}")
        print("RESULTS ANALYSIS")
        print(f"{'='*80}")

        df = self.results_df

        # 1. Descriptive statistics
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 80)
        print(f"  Minimum transport cost evaluated: ${df['c_trans'].min():,.2f}")
        print(f"  Maximum transport cost evaluated: ${df['c_trans'].max():,.2f}")
        print(f"  Maximum benefit: ${df['benefit'].max():,.2f}")
        print(f"  Minimum benefit: ${df['benefit'].min():,.2f}")
        print(f"  Maximum waste: {df['waste'].max():,.2f} kg")
        print(f"  Minimum waste: {df['waste'].min():,.2f} kg")

        # 2. Identify critical thresholds (inflection point)
        print("\n2. CRITICAL THRESHOLD ANALYSIS")
        print("-" * 80)

        # Benefit change rate
        df_sorted = df.sort_values('c_trans')
        df_sorted['benefit_change'] = df_sorted['benefit'].diff()
        df_sorted['benefit_change_rate'] = df_sorted['benefit_change'] / df_sorted['c_trans'].diff()

        max_rate_change_idx = df_sorted['benefit_change_rate'].abs().idxmax()
        if pd.notna(max_rate_change_idx):
            threshold_c_trans = df_sorted.loc[max_rate_change_idx, 'c_trans']
            print(f"  • Transport cost with highest benefit change rate: ${threshold_c_trans:,.2f}")

        # 3. Trade-off analysis
        print("\n3. TRADE-OFF ANALYSIS")
        print("-" * 80)

        corr_c_trans_waste = df['c_trans'].corr(df['waste'])
        corr_c_trans_benefit = df['c_trans'].corr(df['benefit'])
        print(f"  • Correlation cost-waste: {corr_c_trans_waste:.4f}")
        print(f"  • Correlation cost-benefit: {corr_c_trans_benefit:.4f}")

        # 4. Recommendations
        print("\n4. RECOMMENDATIONS")
        print("-" * 80)

        # Optimal cost (maximizes benefit)
        optimal_idx = df['benefit'].idxmax()
        optimal_c_trans = df.loc[optimal_idx, 'c_trans']
        optimal_benefit = df.loc[optimal_idx, 'benefit']
        optimal_waste = df.loc[optimal_idx, 'waste']

        print(f"  • Optimal transport cost (max benefit): ${optimal_c_trans:,.2f}")
        print(f"    - Benefit: ${optimal_benefit:,.2f}")
        print(f"    - Waste: {optimal_waste:.2f} kg")

        # Cost for minimum waste
        min_waste_idx = df['waste'].idxmin()
        min_waste_c_trans = df.loc[min_waste_idx, 'c_trans']
        min_waste_benefit = df.loc[min_waste_idx, 'benefit']
        min_waste = df.loc[min_waste_idx, 'waste']

        print(f"  • Transport cost for minimum waste: ${min_waste_c_trans:,.2f}")
        print(f"    - Benefit: ${min_waste_benefit:,.2f}")
        print(f"    - Waste: {min_waste:.2f} kg")

        # Save analysis to file
        analysis_file = f"{self.output_dir}/sensitivity_analysis.txt"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("SENSITIVITY ANALYSIS - TRANSPORT COST\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Fixed penalty: ${df['penalty_fixed'].iloc[0]}\n")
            f.write(f"Multipliers evaluated: {self.c_trans_multipliers}\n")
            f.write(f"Transport cost range: ${df['c_trans'].min():,.2f} - ${df['c_trans'].max():,.2f}\n\n")
            f.write(f"Optimal cost: ${optimal_c_trans:,.2f}\n")
            f.write(f"Maximum benefit: ${optimal_benefit:,.2f}\n")
            f.write(f"Waste at optimum: {optimal_waste:.2f} kg\n\n")
            f.write(f"Correlation cost-waste: {corr_c_trans_waste:.4f}\n")
            f.write(f"Correlation cost-benefit: {corr_c_trans_benefit:.4f}\n")

        print(f"\n✓ Analysis saved to: {analysis_file}")
    
    def export_results(self):
        """
        Export results to CSV and JSON files.
        """
        if self.results_df is None or len(self.results_df) == 0:
            print("No results to export")
            return

        print(f"\n{'='*80}")
        print("EXPORTING RESULTS")
        print(f"{'='*80}")

        csv_file = f"{self.output_dir}/sensitivity_transport_cost.csv"
        self.results_df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✓ Results exported to CSV: {csv_file}")

        json_file = f"{self.output_dir}/sensitivity_transport_cost.json"
        self.results_df.to_json(json_file, orient='records', indent=2)
        print(f"✓ Results exported to JSON: {json_file}")

        # Create plotting data file (with English column names)
        plot_data_file = f"{self.output_dir}/plotting_data.csv"
        plot_df = self.results_df[['c_trans', 'benefit', 'waste', 'total_donated']].copy()
        plot_df.columns = ['Transport Cost ($/kg-km)', 'Benefit ($)', 'Waste (kg)', 'Donated (kg)']
        plot_df.to_csv(plot_data_file, index=False, encoding='utf-8')
        print(f"✓ Plotting data exported: {plot_data_file}")
    
    def generate_plots(self):
        """
        Generate the three requested plots from plotting_data.csv.
        All texts are in English.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("\n❌ matplotlib is not installed. No plots will be generated.")
            print("   Install with: pip install matplotlib")
            return
        
        plot_data_file = f"{self.output_dir}/plotting_data.csv"
        if not os.path.exists(plot_data_file):
            print(f"\n❌ File {plot_data_file} not found. Run export_results first.")
            return
        
        df = pd.read_csv(plot_data_file)
        
        print(f"\n{'='*80}")
        print("GENERATING PLOTS")
        print(f"{'='*80}")
        
        # 1. Benefit vs. Transport Cost
        plt.figure(figsize=(10, 6))
        plt.plot(df['Transport Cost ($/kg-km)'], df['Benefit ($)'], 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Transport Cost ($/kg-km)', fontsize=12)
        plt.ylabel('Net Benefit ($)', fontsize=12)
        plt.title('Benefit vs. Transport Cost', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/benefit_vs_transport_cost.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Plot 1: benefit_vs_transport_cost.png")
        
        # 2. Waste vs. Transport Cost
        plt.figure(figsize=(10, 6))
        plt.plot(df['Transport Cost ($/kg-km)'], df['Waste (kg)'], 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Transport Cost ($/kg-km)', fontsize=12)
        plt.ylabel('Waste (kg)', fontsize=12)
        plt.title('Waste vs. Transport Cost', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/waste_vs_transport_cost.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Plot 2: waste_vs_transport_cost.png")
        
        # 3. Dual-axis plot (Benefit and Waste)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Benefit (left axis)
        color1 = 'blue'
        ax1.set_xlabel('Transport Cost ($/kg-km)', fontsize=12)
        ax1.set_ylabel('Net Benefit ($)', color=color1, fontsize=12)
        ax1.plot(df['Transport Cost ($/kg-km)'], df['Benefit ($)'], color=color1, marker='o', linewidth=2, label='Benefit')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Waste (right axis)
        ax2 = ax1.twinx()
        color2 = 'red'
        ax2.set_ylabel('Waste (kg)', color=color2, fontsize=12)
        ax2.plot(df['Transport Cost ($/kg-km)'], df['Waste (kg)'], color=color2, marker='s', linewidth=2, label='Waste')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title('Trade-off: Benefit vs. Waste', fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.savefig(f"{self.output_dir}/dual_plot_transport_cost.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Plot 3: dual_plot_transport_cost.png")
        
        print(f"\n✓ All plots saved in: {self.output_dir}/")
    
    def generate_report(self):
        if self.results_df is None or len(self.results_df) == 0:
            print("No results to generate report")
            return

        print(f"\n{'='*80}")
        print("GENERATING REPORT")
        print(f"{'='*80}")

        df = self.results_df
        report_file = f"{self.output_dir}/sensitivity_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# EXPERIMENT 2: Sensitivity Analysis - Transport Cost\n\n")
            f.write("## Complementary Experiment to the 2² Factorial\n\n")

            f.write("### 1. Objective\n\n")
            f.write("Analyze the functional relationship between transport cost and ")
            f.write("response variables (net benefit and food waste) while keeping ")
            f.write("the waste penalty fixed.\n\n")

            f.write("### 2. Experiment Configuration\n\n")
            f.write(f"- **Fixed penalty:** ${df['penalty_fixed'].iloc[0]}\n")
            f.write(f"- **Base transport cost:** ${self.base_c_trans}\n")
            f.write(f"- **Multipliers evaluated:** {self.c_trans_multipliers}\n")
            f.write(f"- **Transport cost range:** ${df['c_trans'].min():,.2f} - ${df['c_trans'].max():,.2f}\n")
            f.write(f"- **Number of levels evaluated:** {len(df)}\n\n")

            f.write("### 3. Main Results\n\n")
            optimal_idx = df['benefit'].idxmax()
            min_waste_idx = df['waste'].idxmin()

            f.write("#### 3.1. Optimal Configuration (Maximum Benefit)\n\n")
            f.write(f"- **Transport cost:** ${df.loc[optimal_idx, 'c_trans']:,.2f}\n")
            f.write(f"- **Benefit:** ${df.loc[optimal_idx, 'benefit']:,.2f}\n")
            f.write(f"- **Waste:** {df.loc[optimal_idx, 'waste']:,.2f} kg\n\n")

            f.write("#### 3.2. Minimum Waste Configuration\n\n")
            f.write(f"- **Transport cost:** ${df.loc[min_waste_idx, 'c_trans']:,.2f}\n")
            f.write(f"- **Benefit:** ${df.loc[min_waste_idx, 'benefit']:,.2f}\n")
            f.write(f"- **Waste:** {df.loc[min_waste_idx, 'waste']:,.2f} kg\n\n")

            corr_c_w = df['c_trans'].corr(df['waste'])
            corr_c_b = df['c_trans'].corr(df['benefit'])
            f.write("#### 3.3. Correlations\n\n")
            f.write(f"- **Cost vs. Waste:** {corr_c_w:.4f}\n")
            f.write(f"- **Cost vs. Benefit:** {corr_c_b:.4f}\n\n")

            # Trade-off
            if df.loc[optimal_idx, 'c_trans'] != df.loc[min_waste_idx, 'c_trans']:
                benefit_loss = df.loc[optimal_idx, 'benefit'] - df.loc[min_waste_idx, 'benefit']
                waste_saved = df.loc[optimal_idx, 'waste'] - df.loc[min_waste_idx, 'waste']
                f.write("#### 3.4. Trade-off Analysis\n\n")
                f.write(f"Moving from the optimal configuration (max benefit) to the minimum waste one:\n\n")
                f.write(f"- Saves {waste_saved:.2f} kg of waste\n")
                f.write(f"- Reduces benefit by ${benefit_loss:,.2f}\n")
                f.write(f"- Cost per kg saved: ${benefit_loss/waste_saved:,.2f}\n\n")

            f.write("### 4. Generated Plots\n\n")
            f.write("1. **Benefit vs. Transport Cost** (`benefit_vs_transport_cost.png`)\n")
            f.write("2. **Waste vs. Transport Cost** (`waste_vs_transport_cost.png`)\n")
            f.write("3. **Dual-axis plot** (`dual_plot_transport_cost.png`)\n\n")

            f.write("---\n")
            f.write(f"*Report automatically generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

        print(f"✓ Report generated in: {report_file}")    


def main():
    print("=" * 80)
    print("SENSITIVITY ANALYSIS SYSTEM")
    print("Complementary Experiment: Transport Cost Variation")
    print("=" * 80)

    instance_file = "instance_distance_exp.json"
    output_dir = "sensitivity_results"

    analyzer = SensitivityAnalyzer(instance_file, output_dir)

    try:
        # Generate multipliers from 1 to 30 (15 points) to cover 0.138 to 4.14
        multiplicadores = np.linspace(1, 30, 15)  # adjust number as needed

        analyzer.run_sensitivity_analysis(
            fixed_penalty=50,
            base_c_trans=0.138,
            c_trans_multipliers=multiplicadores,
            time_limit=200
        )

        analyzer.analyze_results()
        analyzer.export_results()
        analyzer.generate_plots()
        analyzer.generate_report()

        print(f"\n{'='*80}")
        print("SENSITIVITY ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"\nAll results are in: {output_dir}/")

    except FileNotFoundError:
        print(f"\n❌ Error: File '{instance_file}' not found.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
