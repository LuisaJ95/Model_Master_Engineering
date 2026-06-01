# %%
"""
===================================================================================
EXPERIMENT 3: Interaction Between Tax Incentives and Fleet Capacity
COMPLETE STANDALONE VERSION
===================================================================================
Design: 3x3 Factorial Design (9 treatment combinations, 4 replicates = 36 runs)

This file contains:
1. FoodDonationOptimizer class (simplified for experiments)
2. Experimental design setup
3. Experiment execution engine
4. Statistical analysis (ANOVA, contrasts, regression)
5. Visualization tools
6. Main execution

Usage:
    python experiment_3_complete.py

Requirements:
    - instance_distance_exp.json (base data file)
    - gurobipy, numpy, pandas, matplotlib, seaborn, scipy, statsmodels
===================================================================================
"""

import gurobipy as gp
from gurobipy import GRB
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import time
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("EXPERIMENT 3: TAX INCENTIVES × FLEET CAPACITY INTERACTION")
print("="*80)
print("\n✓ Libraries imported successfully\n")


# ===================================================================================
# OPTIMIZER CLASS (Simplified for Experiments)
# ===================================================================================

class FoodDonationOptimizer:
    """Simplified optimizer for experimental runs."""
    
    def __init__(self, env):
        self.env = env
        self.model = None
        self.requirements = []
        self.requirements_dict = {}
        self.R_direct = []
        self.R_indirect = []
        self.T = []
        self.B = []
        self.P = []
        self.beta = 0
        self.pi = 0
        self.c_trans = 0
        self.alpha = 0
        self.product_costs = {}
        self.food_bank_capacity = {}
        self.transport_capacity = {}
        self.y_deliv = {}
        self.x_out = {}
        self.y_pickup = {}
        self.w = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
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
        return self.requirements_dict.get(req_id)
    
    def build_model(self):
        """Build the optimization model."""
        self.model = gp.Model("FoodDonation", env=self.env)
        self._create_variables()
        self._set_objective()
        self._add_constraints()
        self.model.update()
    
    def _create_variables(self):
        """Create decision variables."""
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            l_r, e_r = req['release_date'], req['expiration_date']
            for t in range(l_r, e_r + 1):
                self.y_deliv[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"y_deliv_D_{r}_{t}")
                self.x_out[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"x_out_{r}_{t}")
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r, e_r = req['release_date'], req['expiration_date']
            for t in range(l_r, e_r):
                self.y_pickup[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"y_pickup_{r}_{t}")
            for t in range(l_r + 1, e_r + 1):
                self.y_deliv[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"y_deliv_I_{r}_{t}")
        
        for r in self.R_direct + self.R_indirect:
            self.w[r] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"w_{r}")
    
    def _set_objective(self):
        """Set objective function."""
        product_benefit_terms = []
        transport_benefit_terms = []
        transport_cost_terms = []
        outsource_cost_terms = []
        
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            product, dist = req['product'], req['distance']
            c_prod = self.product_costs.get(product, 0)
            l_r, e_r = req['release_date'], req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                product_benefit_terms.append(self.beta * c_prod * (self.y_deliv[r, t] + self.x_out[r, t]))
                transport_benefit_terms.append(self.beta * self.c_trans * dist * (self.y_deliv[r, t] + self.alpha * self.x_out[r, t]))
                transport_cost_terms.append(self.c_trans * dist * self.y_deliv[r, t])
                outsource_cost_terms.append(self.alpha * self.c_trans * dist * self.x_out[r, t])
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            product, dist_indir = req['product'], req['dist_indirect']
            c_prod = self.product_costs.get(product, 0)
            l_r, e_r = req['release_date'], req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                product_benefit_terms.append(self.beta * c_prod * self.y_deliv[r, t])
                transport_benefit_terms.append(self.beta * self.c_trans * dist_indir * self.y_deliv[r, t])
                transport_cost_terms.append(self.c_trans * dist_indir * self.y_deliv[r, t])
        
        waste_terms = [self.pi * self.w[r] for r in self.R_direct + self.R_indirect]
        
        obj = (gp.quicksum(product_benefit_terms) + gp.quicksum(transport_benefit_terms) -
               gp.quicksum(transport_cost_terms) - gp.quicksum(outsource_cost_terms) -
               gp.quicksum(waste_terms))
        
        self.model.setObjective(obj, GRB.MAXIMIZE)
    
    def _add_constraints(self):
        """Add constraints."""
        # Flow conservation - Direct
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            q_r, l_r, e_r = req['quantity'], req['release_date'], req['expiration_date']
            self.model.addConstr(
                gp.quicksum(self.y_deliv[r, t] + self.x_out[r, t] for t in range(l_r, e_r + 1)) + self.w[r] == q_r,
                name=f"flow_direct_{r}"
            )
        
        # Flow conservation - Indirect
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            q_r, l_r, e_r = req['quantity'], req['release_date'], req['expiration_date']
            self.model.addConstr(
                gp.quicksum(self.y_deliv[r, t] for t in range(l_r + 1, e_r + 1)) + self.w[r] == q_r,
                name=f"flow_indirect_{r}"
            )
        
        # Food bank capacity
        direct_reqs_by_bank = {j: [] for j in self.B}
        indirect_reqs_by_bank = {j: [] for j in self.B}
        
        for r in self.R_direct:
            dest = self._get_requirement_by_id(r)['destination']
            if dest in direct_reqs_by_bank:
                direct_reqs_by_bank[dest].append(r)
        
        for r in self.R_indirect:
            dest = self._get_requirement_by_id(r)['destination']
            if dest in indirect_reqs_by_bank:
                indirect_reqs_by_bank[dest].append(r)
        
        for j in self.B:
            cap_j = self.food_bank_capacity.get(j, 0)
            for t in self.T:
                deliveries = []
                for r in direct_reqs_by_bank[j]:
                    req = self._get_requirement_by_id(r)
                    l_r, e_r = req['release_date'], req['expiration_date']
                    if l_r <= t <= e_r:
                        deliveries.append(self.y_deliv[r, t] + self.x_out[r, t])
                
                for r in indirect_reqs_by_bank[j]:
                    req = self._get_requirement_by_id(r)
                    l_r, e_r = req['release_date'], req['expiration_date']
                    if l_r + 1 <= t <= e_r:
                        deliveries.append(self.y_deliv[r, t])
                
                if deliveries:
                    self.model.addConstr(gp.quicksum(deliveries) <= cap_j, name=f"foodbank_cap_{j}_{t}")
        
        # Transport capacity
        for t in self.T:
            cap_trans = self.transport_capacity[t]
            fleet_usage = []
            
            for r in self.R_direct:
                req = self._get_requirement_by_id(r)
                l_r, e_r = req['release_date'], req['expiration_date']
                if l_r <= t <= e_r:
                    fleet_usage.append(self.y_deliv[r, t])
            
            for r in self.R_indirect:
                req = self._get_requirement_by_id(r)
                l_r, e_r = req['release_date'], req['expiration_date']
                if l_r <= t < e_r:
                    fleet_usage.append(self.y_pickup[r, t])
                if l_r + 1 <= t <= e_r:
                    fleet_usage.append(self.y_deliv[r, t])
            
            if fleet_usage:
                self.model.addConstr(gp.quicksum(fleet_usage) <= cap_trans, name=f"transport_cap_{t}")
        
        # Synchronization - Indirect
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r, e_r = req['release_date'], req['expiration_date']
            for t in range(l_r, e_r):
                self.model.addConstr(self.y_pickup[r, t] == self.y_deliv[r, t + 1], name=f"sync_{r}_{t}")
    
    def optimize(self):
        """Solve the model."""
        self.model.optimize()
        return self.model.Status
    
    def _calculate_metrics(self):
        """Calculate detailed metrics."""
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or self.model.SolCount == 0:
            return None
        
        metrics = {
            'tax_benefit_products': 0,
            'tax_benefit_transport': 0,
            'own_fleet_costs': 0,
            'outsourcing_costs': 0,
            'waste_penalty': 0,
            'total_donated': 0,
            'total_wasted': 0,
            'total_outsourced': 0,
            'total_available': 0,
            'donation_rate': 0
        }
        
        # Calculate metrics from solution
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            product, dist = req['product'], req['distance']
            c_prod = self.product_costs.get(product, 0)
            l_r, e_r = req['release_date'], req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                y_val = self.y_deliv[r, t].X
                x_val = self.x_out[r, t].X
                metrics['tax_benefit_products'] += self.beta * c_prod * (y_val + x_val)
                metrics['tax_benefit_transport'] += self.beta * self.c_trans * dist * (y_val + self.alpha * x_val)
                metrics['own_fleet_costs'] += self.c_trans * dist * y_val
                metrics['outsourcing_costs'] += self.alpha * self.c_trans * dist * x_val
                metrics['total_donated'] += y_val + x_val
                metrics['total_outsourced'] += x_val
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            product, dist_indir = req['product'], req['dist_indirect']
            c_prod = self.product_costs.get(product, 0)
            l_r, e_r = req['release_date'], req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                y_val = self.y_deliv[r, t].X
                metrics['tax_benefit_products'] += self.beta * c_prod * y_val
                metrics['tax_benefit_transport'] += self.beta * self.c_trans * dist_indir * y_val
                metrics['own_fleet_costs'] += self.c_trans * dist_indir * y_val
                metrics['total_donated'] += y_val
        
        for r in self.R_direct + self.R_indirect:
            w_val = self.w[r].X
            metrics['waste_penalty'] += self.pi * w_val
            metrics['total_wasted'] += w_val
            metrics['total_available'] += self._get_requirement_by_id(r)['quantity']
        
        if metrics['total_available'] > 0:
            metrics['donation_rate'] = (metrics['total_donated'] / metrics['total_available']) * 100
        
        return metrics
    
    def get_solution(self):
        """Extract solution details."""
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or self.model.SolCount == 0:
            return None
        
        solution = {
            'objective_value': self.model.ObjVal,
            'total_donated': 0,
            'total_wasted': 0,
            'total_outsourced': 0
        }
        
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            l_r, e_r = req['release_date'], req['expiration_date']
            for t in range(l_r, e_r + 1):
                solution['total_donated'] += self.y_deliv[r, t].X + self.x_out[r, t].X
                solution['total_outsourced'] += self.x_out[r, t].X
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r, e_r = req['release_date'], req['expiration_date']
            for t in range(l_r + 1, e_r + 1):
                solution['total_donated'] += self.y_deliv[r, t].X
        
        for r in self.R_direct + self.R_indirect:
            solution['total_wasted'] += self.w[r].X
        
        return solution


# ===================================================================================
# EXPERIMENTAL DESIGN
# ===================================================================================

class ExperimentalDesign:
    """Factorial experimental design 3×3."""
    
    def __init__(self):
        self.factor_A_levels = {'Low': 0.10, 'Medium': 0.37, 'High': 0.90}
        self.factor_B_levels = {'Limited': 200, 'Standard': 8000, 'Expanded': 14000}
        self.num_replicates = 4
        self.treatments = self._generate_treatments()
        self.run_order = self._randomize_runs()
    
    def _generate_treatments(self):
        treatments = []
        treatment_id = 1
        for a_label, a_value in self.factor_A_levels.items():
            for b_label, b_value in self.factor_B_levels.items():
                treatments.append({
                    'treatment_id': treatment_id,
                    'factor_A_label': a_label,
                    'factor_A_value': a_value,
                    'factor_B_label': b_label,
                    'factor_B_value': b_value,
                    'combination': f"{a_label}_{b_label}"
                })
                treatment_id += 1
        return treatments
    
    def _randomize_runs(self):
        runs = []
        run_id = 1
        for replicate in range(1, self.num_replicates + 1):
            for treatment in self.treatments:
                runs.append({
                    'run_id': run_id,
                    'replicate': replicate,
                    **treatment
                })
                run_id += 1
        
        np.random.seed(42)
        np.random.shuffle(runs)
        for idx, run in enumerate(runs, 1):
            run['run_id'] = idx
        return runs
    
    def display_design(self):
        print("\n" + "="*80)
        print("EXPERIMENTAL DESIGN")
        print("="*80)
        print(f"\n3×3 Factorial Design: {len(self.treatments)} treatments × {self.num_replicates} replicates = {len(self.run_order)} runs")
        print(f"\nFactor A (Tax Benefit β₁): {list(self.factor_A_levels.values())}")
        print(f"Factor B (Fleet Capacity): {list(self.factor_B_levels.values())} kg")
        return pd.DataFrame(self.run_order)


# ===================================================================================
# EXPERIMENT RUNNER
# ===================================================================================

class ExperimentRunner:
    """Execute experiments."""
    
    def __init__(self, base_data_file, design):
        self.base_data_file = base_data_file
        self.design = design
        self.results = []
        with open(base_data_file, 'r') as f:
            self.base_data = json.load(f)
    
    def _modify_instance(self, run_config):
        data = deepcopy(self.base_data)
        data['parameters']['beta_1'] = run_config['factor_A_value']
        data['parameters']['transport_capacity_per_period'] = run_config['factor_B_value']
        
        # Scenario: 50% demand increase
        for req in data['requirements']:
            req['quantity'] = req['quantity'] * 1.5
        
        return data
    
    def _run_optimization(self, data, run_config):
        try:
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.setParam('TimeLimit', 300)
                env.setParam('MIPGap', 0.01)
                env.start()
                
                temp_file = f"temp_run_{run_config['run_id']}.json"
                with open(temp_file, 'w') as f:
                    json.dump(data, f)
                
                with FoodDonationOptimizer(env) as optimizer:
                    optimizer.load_data_from_json(temp_file)
                    optimizer.build_model()
                    status = optimizer.optimize()
                    
                    if status == GRB.OPTIMAL or (status == GRB.TIME_LIMIT and optimizer.model.SolCount > 0):
                        metrics = optimizer._calculate_metrics()
                        solution = optimizer.get_solution()
                        
                        y1_profit = optimizer.model.ObjVal
                        y2_compliance = metrics['donation_rate']
                        
                        total_transported = metrics['total_donated']
                        y3_outsourcing = (solution['total_outsourced'] / total_transported * 100) if total_transported > 0 else 0
                        
                        return {
                            'status': 'optimal',
                            'Y1_profit': y1_profit,
                            'Y2_compliance': y2_compliance,
                            'Y3_outsourcing': y3_outsourcing,
                            'total_donated': metrics['total_donated'],
                            'total_wasted': metrics['total_wasted'],
                            'total_outsourced': solution['total_outsourced']
                        }
                    else:
                        return {'status': 'infeasible', 'Y1_profit': np.nan, 'Y2_compliance': np.nan, 'Y3_outsourcing': np.nan}
        except Exception as e:
            print(f"  Error in run {run_config['run_id']}: {e}")
            return {'status': 'error', 'Y1_profit': np.nan, 'Y2_compliance': np.nan, 'Y3_outsourcing': np.nan}
    
    def run_all_experiments(self):
        print("\n" + "="*80)
        print("RUNNING EXPERIMENTS")
        print("="*80)
        
        total = len(self.design.run_order)
        for idx, run_config in enumerate(self.design.run_order, 1):
            print(f"\n[{idx}/{total}] Run {run_config['run_id']}: β₁={run_config['factor_A_value']:.2f}, cap={run_config['factor_B_value']:.0f}kg")
            
            modified_data = self._modify_instance(run_config)
            result = self._run_optimization(modified_data, run_config)
            
            self.results.append({**run_config, **result})
            
            if result['status'] == 'optimal':
                print(f"  Profit: ${result['Y1_profit']:,.2f}, Compliance: {result['Y2_compliance']:.1f}%, Outsourcing: {result['Y3_outsourcing']:.1f}%")
        
        print("\n" + "="*80)
        print("✓ EXPERIMENTS COMPLETED")
        print("="*80)
        
        return pd.DataFrame(self.results)


# ===================================================================================
# STATISTICAL ANALYSIS
# ===================================================================================

class StatisticalAnalysis:
    """Statistical analysis of experimental results."""
    
    def __init__(self, results_df):
        self.df = results_df.dropna(subset=['Y1_profit', 'Y2_compliance', 'Y3_outsourcing'])
    
    def perform_two_way_anova(self, response_var='Y1_profit'):
        print(f"\n{'='*80}")
        print(f"TWO-WAY ANOVA: {response_var}")
        print("="*80)
        
        formula = f'{response_var} ~ C(factor_A_label) + C(factor_B_label) + C(factor_A_label):C(factor_B_label)'
        model = ols(formula, data=self.df).fit()
        anova_table = anova_lm(model, typ=2)
        
        print("\nANOVA Table:")
        print(anova_table)
        
        print("\n📊 Interpretation:")
        for effect in anova_table.index[:-1]:
            p_value = anova_table.loc[effect, 'PR(>F)']
            f_value = anova_table.loc[effect, 'F']
            
            if 'factor_A' in effect and 'factor_B' in effect:
                effect_name = "Interaction (A × B)"
            elif 'factor_A' in effect:
                effect_name = "Main effect A (Tax Benefit)"
            elif 'factor_B' in effect:
                effect_name = "Main effect B (Fleet Capacity)"
            else:
                effect_name = effect
            
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  {effect_name:40s}: F = {f_value:8.2f}, p = {p_value:.4f} {sig}")
        
        return anova_table, model
    
    def comprehensive_analysis(self):
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*80)
        
        results = {}
        for var_name, var_label in [('Y1_profit', 'Profit'), ('Y2_compliance', 'Compliance'), ('Y3_outsourcing', 'Outsourcing')]:
            print(f"\n{'='*80}")
            print(f"ANALYSIS: {var_label}")
            print("="*80)
            
            anova_table, model = self.perform_two_way_anova(var_name)
            results[var_name] = {'anova_table': anova_table, 'model': model}
        
        return results


# ===================================================================================
# VISUALIZATION
# ===================================================================================

class ExperimentVisualizer:
    """Create visualizations."""
    
    def __init__(self, results_df):
        self.df = results_df.dropna(subset=['Y1_profit', 'Y2_compliance', 'Y3_outsourcing'])
    
    def plot_interaction_profiles(self, save=True):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        vars = [('Y1_profit', 'Total Net Profit ($)'), 
                ('Y2_compliance', 'Compliance Rate (%)'),
                ('Y3_outsourcing', 'Outsourcing (%)')]
        
        for idx, (var_name, var_label) in enumerate(vars):
            ax = axes[idx]
            means = self.df.groupby(['factor_A_label', 'factor_B_label'])[var_name].mean().reset_index()
            pivot = means.pivot(index='factor_A_label', columns='factor_B_label', values=var_name)
            pivot = pivot.reindex(['Low', 'Medium', 'High'])
            
            for col in pivot.columns:
                ax.plot(['Low', 'Medium', 'High'], pivot[col], 
                       marker='o', linewidth=2, markersize=8, label=f'{col}')
            
            ax.set_xlabel('Tax Benefit (β₁)', fontsize=12, fontweight='bold')
            ax.set_ylabel(var_label, fontsize=12, fontweight='bold')
            ax.set_title(f'Interaction: {var_label}', fontsize=13, fontweight='bold')
            ax.legend(title='Fleet Capacity')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig('experiment3_interactions.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: experiment3_interactions.png")
        plt.show()
    
    def create_all_visualizations(self):
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        self.plot_interaction_profiles()


# ===================================================================================
# MAIN EXECUTION
# ===================================================================================

def main():
    print("\n" + "="*80)
    print("EXPERIMENT 3: MAIN EXECUTION")
    print("="*80)
    
    # Step 1: Design
    print("\n[1/4] Creating experimental design...")
    design = ExperimentalDesign()
    design_df = design.display_design()
    design_df.to_csv('experiment3_design.csv', index=False)
    print("✓ Saved: experiment3_design.csv")
    
    # Step 2: Run experiments
    print("\n[2/4] Running experiments...")
    runner = ExperimentRunner('instance_distance_exp.json', design)
    results_df = runner.run_all_experiments()
    results_df.to_csv('experiment3_results.csv', index=False)
    print("✓ Saved: experiment3_results.csv")
    
    # Step 3: Statistical analysis
    print("\n[3/4] Performing statistical analysis...")
    analyzer = StatisticalAnalysis(results_df)
    analysis_results = analyzer.comprehensive_analysis()
    
    # Step 4: Visualizations
    print("\n[4/4] Creating visualizations...")
    visualizer = ExperimentVisualizer(results_df)
    visualizer.create_all_visualizations()
    
    # Summary
    print("\n" + "="*80)
    print("✓ EXPERIMENT 3 COMPLETED")
    print("="*80)
    print("\nFiles generated:")
    print("  1. experiment3_design.csv - Experimental design")
    print("  2. experiment3_results.csv - Results")
    print("  3. experiment3_interactions.png - Interaction plots")
    print("="*80)
    
    return results_df, analysis_results


if __name__ == "__main__":
    results_df, analysis_results = main()



