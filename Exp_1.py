# %%
"""
================================================================================
EXPERIMENT 1: Profit Sensitivity to Operating Costs and Penalties
================================================================================
Design Type: Full Factorial Design 2²
Objective: Determine how transportation costs and waste penalties affect 
           system profitability and identify substitution effects.
================================================================================
"""

import gurobipy as gp
from gurobipy import GRB
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import product
import time
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class FoodDonationOptimizer:
    """Optimizer for food donation network (simplified for experiments)"""
    
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
    
    def load_data_from_json(self, json_file):
        """Load instance data from JSON file"""
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
        """Get requirement data by ID"""
        return self.requirements_dict.get(req_id)
    
    def build_model(self):
        """Build the optimization model"""
        self.model = gp.Model("FoodDonation_Exp", env=self.env)
        self._create_variables()
        self._set_objective()
        self._add_constraints()
        self.model.update()
    
    def _create_variables(self):
        """Create decision variables"""
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            for t in range(l_r, e_r + 1):
                self.y_deliv[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"y_deliv_D_{r}_{t}")
                self.x_out[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"x_out_{r}_{t}")
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            for t in range(l_r, e_r):
                self.y_pickup[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"y_pickup_{r}_{t}")
            for t in range(l_r + 1, e_r + 1):
                self.y_deliv[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"y_deliv_I_{r}_{t}")
        
        for r in self.R_direct + self.R_indirect:
            self.w[r] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"w_{r}")
    
    def _set_objective(self):
        """Set objective function"""
        product_benefit_terms = []
        transport_benefit_terms = []
        transport_cost_terms = []
        outsource_cost_terms = []
        
        # Direct requirements
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
        
        # Indirect requirements
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
        """Add constraints"""
        # Flow conservation - Direct
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            q_r = req['quantity']
            l_r = req['release_date']
            e_r = req['expiration_date']
            self.model.addConstr(
                gp.quicksum(self.y_deliv[r, t] + self.x_out[r, t] for t in range(l_r, e_r + 1)) + self.w[r] == q_r,
                name=f"flow_direct_{r}"
            )
        
        # Flow conservation - Indirect
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            q_r = req['quantity']
            l_r = req['release_date']
            e_r = req['expiration_date']
            self.model.addConstr(
                gp.quicksum(self.y_deliv[r, t] for t in range(l_r + 1, e_r + 1)) + self.w[r] == q_r,
                name=f"flow_indirect_{r}"
            )
        
        # Food bank capacity
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
                    self.model.addConstr(gp.quicksum(deliveries) <= cap_j, name=f"foodbank_cap_{j}_{t}")
        
        # Transport capacity
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
                if l_r + 1 <= t <= e_r:
                    fleet_usage.append(self.y_deliv[r, t])
            
            if fleet_usage:
                self.model.addConstr(gp.quicksum(fleet_usage) <= cap_trans, name=f"transport_cap_{t}")
        
        # Synchronization for indirect
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            for t in range(l_r, e_r):
                self.model.addConstr(self.y_pickup[r, t] == self.y_deliv[r, t + 1], name=f"sync_{r}_{t}")
    
    def optimize(self, silent=True):
        """Solve the model"""
        if silent:
            self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        return self.model.Status
    
    def get_results(self):
        """Extract results"""
        if self.model.Status == GRB.OPTIMAL:
            total_waste = sum(self.w[r].X for r in self.R_direct + self.R_indirect)
            return {
                'objective': self.model.ObjVal,
                'total_waste': total_waste,
                'status': 'optimal'
            }
        else:
            return {
                'objective': None,
                'total_waste': None,
                'status': 'not_optimal'
            }
    
    def dispose(self):
        """Dispose model"""
        if self.model is not None:
            self.model.dispose()


# ================================================================================
# FACTORIAL EXPERIMENT CLASS
# ================================================================================

class FactorialExperiment:
    """
    Full Factorial Design 2² Experiment
    Factor A: Transportation cost (c_trans)
    Factor B: Waste penalty (pi)
    """
    
    def __init__(self, json_file, n_replicates=5):
        """
        Initialize experiment
        
        Args:
            json_file: Path to instance JSON file
            n_replicates: Number of replicates per treatment (default: 5)
        """
        self.json_file = json_file
        self.n_replicates = n_replicates
        
        # Load base parameters
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.base_c_trans = data['parameters']['c_trans']
        self.base_pi = data['parameters']['pi_penalty']
        
        # Define factor levels
        self.factor_levels = {
            'A': {
                'low': 0.2,    # Reduce by 80%
                'high': 2.0    # Double
            },
            'B': {
                'low': 0.1,    # Reduce by 90%
                'high': 2.0    # Double
            }
        }
        
        # Create treatment combinations
        self.treatments = list(product(['low', 'high'], repeat=2))
        self.treatment_labels = [
            f"A_{a}_B_{b}" for a, b in self.treatments
        ]
        
        # Results storage
        self.results = []
        self.results_df = None
        
        print("="*80)
        print("EXPERIMENT 1: Full Factorial Design 2²")
        print("="*80)
        print(f"\nBase parameters:")
        print(f"  - c_trans (base): ${self.base_c_trans:.4f}")
        print(f"  - pi (base): ${self.base_pi:.4f}")
        print(f"\nFactor A (Transportation Cost) levels:")
        print(f"  - Low:  {self.factor_levels['A']['low']} × base = ${self.base_c_trans * self.factor_levels['A']['low']:.4f}")
        print(f"  - High: {self.factor_levels['A']['high']} × base = ${self.base_c_trans * self.factor_levels['A']['high']:.4f}")
        print(f"\nFactor B (Waste Penalty) levels:")
        print(f"  - Low:  {self.factor_levels['B']['low']} × base = ${self.base_pi * self.factor_levels['B']['low']:.4f}")
        print(f"  - High: {self.factor_levels['B']['high']} × base = ${self.base_pi * self.factor_levels['B']['high']:.4f}")
        print(f"\nTreatments: {len(self.treatments)}")
        print(f"Replicates per treatment: {n_replicates}")
        print(f"Total runs: {len(self.treatments) * n_replicates}")
        print("="*80)
    
    def run_experiment(self):
        """Execute all experimental runs"""
        print("\n🚀 Starting experimental runs...\n")
        
        total_runs = len(self.treatments) * self.n_replicates
        run_count = 0
        start_time = time.time()
        
        # Create Gurobi environment
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.setParam('TimeLimit', 300)
        env.start()
        
        try:
            for treatment_idx, (level_a, level_b) in enumerate(self.treatments):
                treatment_name = self.treatment_labels[treatment_idx]
                
                # Calculate parameter values
                c_trans = self.base_c_trans * self.factor_levels['A'][level_a]
                pi = self.base_pi * self.factor_levels['B'][level_b]
                
                print(f"Treatment {treatment_idx + 1}/{len(self.treatments)}: {treatment_name}")
                print(f"  c_trans = ${c_trans:.4f}, pi = ${pi:.4f}")
                
                for rep in range(self.n_replicates):
                    run_count += 1
                    
                    # Load and modify data
                    with open(self.json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    data['parameters']['c_trans'] = c_trans
                    data['parameters']['pi_penalty'] = pi
                    
                    # Save temporary instance
                    temp_file = f"temp_instance_{treatment_name}_rep{rep+1}.json"
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f)
                    
                    # Run optimization
                    optimizer = FoodDonationOptimizer(env)
                    optimizer.load_data_from_json(temp_file)
                    optimizer.build_model()
                    status = optimizer.optimize(silent=True)
                    results = optimizer.get_results()
                    optimizer.dispose()
                    
                    # Store results
                    self.results.append({
                        'treatment': treatment_name,
                        'factor_A_level': level_a,
                        'factor_B_level': level_b,
                        'factor_A_value': c_trans,
                        'factor_B_value': pi,
                        'replicate': rep + 1,
                        'Y1_profit': results['objective'],
                        'Y2_waste': results['total_waste'],
                        'status': results['status']
                    })
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    avg_time = elapsed / run_count
                    remaining = avg_time * (total_runs - run_count)
                    
                    print(f"    Replicate {rep + 1}/{self.n_replicates}: " +
                          f"Profit=${results['objective']:,.2f}, " +
                          f"Waste={results['total_waste']:.2f} kg " +
                          f"[{run_count}/{total_runs}, ETA: {remaining:.0f}s]")
                
                print()
        
        finally:
            env.dispose()
        
        # Create DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        total_time = time.time() - start_time
        print(f"\n✓ Experiment completed in {total_time:.2f}s")
        print(f"  Average time per run: {total_time/total_runs:.2f}s")
    
    def calculate_summary_statistics(self):
        """Calculate means and variances by treatment"""
        if self.results_df is None:
            print("Error: No results available. Run experiment first.")
            return None
        
        summary = self.results_df.groupby('treatment').agg({
            'Y1_profit': ['mean', 'std', 'var', 'min', 'max'],
            'Y2_waste': ['mean', 'std', 'var', 'min', 'max']
        }).round(2)
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS BY TREATMENT")
        print("="*80)
        print(summary)
        
        return summary
    
    def perform_anova(self):
        """Perform two-way ANOVA with interaction"""
        if self.results_df is None:
            print("Error: No results available. Run experiment first.")
            return None
        
        print("\n" + "="*80)
        print("TWO-WAY ANOVA WITH INTERACTION")
        print("="*80)
        
        # Convert categorical levels to numerical codes
        df = self.results_df.copy()
        df['A_code'] = (df['factor_A_level'] == 'high').astype(int)
        df['B_code'] = (df['factor_B_level'] == 'high').astype(int)
        df['AB_interaction'] = df['A_code'] * df['B_code']
        
        # ANOVA for Y1 (Profit)
        print("\n📊 ANOVA for Y1 (Total Net Profit)")
        print("-" * 80)
        
        groups_A = [df[df['factor_A_level'] == level]['Y1_profit'].values 
                    for level in ['low', 'high']]
        groups_B = [df[df['factor_B_level'] == level]['Y1_profit'].values 
                    for level in ['low', 'high']]
        
        f_stat_A, p_value_A = stats.f_oneway(*groups_A)
        f_stat_B, p_value_B = stats.f_oneway(*groups_B)
        
        print(f"Main Effect A (Transportation Cost):")
        print(f"  F-statistic: {f_stat_A:.4f}")
        print(f"  p-value: {p_value_A:.6f}")
        print(f"  Significant: {'Yes' if p_value_A < 0.05 else 'No'} (α=0.05)")
        
        print(f"\nMain Effect B (Waste Penalty):")
        print(f"  F-statistic: {f_stat_B:.4f}")
        print(f"  p-value: {p_value_B:.6f}")
        print(f"  Significant: {'Yes' if p_value_B < 0.05 else 'No'} (α=0.05)")
        
        # ANOVA for Y2 (Waste)
        print("\n📊 ANOVA for Y2 (Total Waste)")
        print("-" * 80)
        
        groups_A_waste = [df[df['factor_A_level'] == level]['Y2_waste'].values 
                          for level in ['low', 'high']]
        groups_B_waste = [df[df['factor_B_level'] == level]['Y2_waste'].values 
                          for level in ['low', 'high']]
        
        f_stat_A_waste, p_value_A_waste = stats.f_oneway(*groups_A_waste)
        f_stat_B_waste, p_value_B_waste = stats.f_oneway(*groups_B_waste)
        
        print(f"Main Effect A (Transportation Cost):")
        print(f"  F-statistic: {f_stat_A_waste:.4f}")
        print(f"  p-value: {p_value_A_waste:.6f}")
        print(f"  Significant: {'Yes' if p_value_A_waste < 0.05 else 'No'} (α=0.05)")
        
        print(f"\nMain Effect B (Waste Penalty):")
        print(f"  F-statistic: {f_stat_B_waste:.4f}")
        print(f"  p-value: {p_value_B_waste:.6f}")
        print(f"  Significant: {'Yes' if p_value_B_waste < 0.05 else 'No'} (α=0.05)")
        
        return {
            'Y1': {'A': (f_stat_A, p_value_A), 'B': (f_stat_B, p_value_B)},
            'Y2': {'A': (f_stat_A_waste, p_value_A_waste), 'B': (f_stat_B_waste, p_value_B_waste)}
        }
    
    def perform_regression_analysis(self):
        """Perform linear regression with interaction term"""
        if self.results_df is None:
            print("Error: No results available. Run experiment first.")
            return None
        
        print("\n" + "="*80)
        print("LINEAR REGRESSION ANALYSIS WITH INTERACTION")
        print("="*80)
        
        df = self.results_df.copy()
        
        # Code factors as -1 (low) and +1 (high) for better interpretation
        df['A'] = df['factor_A_level'].map({'low': -1, 'high': 1})
        df['B'] = df['factor_B_level'].map({'low': -1, 'high': 1})
        df['A*B'] = df['A'] * df['B']
        
        # Regression for Y1 (Profit)
        print("\n📈 Regression for Y1 (Total Net Profit)")
        print("-" * 80)
        print("Model: Y1 = β₀ + β₁·A + β₂·B + β₁₂·A·B + ε")
        
        X = df[['A', 'B', 'A*B']].values
        y = df['Y1_profit'].values
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Least squares solution
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        
        print(f"\nCoefficients:")
        print(f"  β₀ (Intercept):     {beta[0]:>12,.2f}")
        print(f"  β₁ (Factor A):      {beta[1]:>12,.2f}")
        print(f"  β₂ (Factor B):      {beta[2]:>12,.2f}")
        print(f"  β₁₂ (Interaction):  {beta[3]:>12,.2f}")
        
        # Calculate R²
        y_pred = X_with_intercept @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"\nModel Fit:")
        print(f"  R²: {r_squared:.4f}")
        
        # Interpretation of interaction
        print(f"\n💡 Interpretation:")
        if abs(beta[3]) > 0.1 * abs(beta[1]):  # Interaction is substantial
            print(f"  The interaction term β₁₂ = {beta[3]:.2f} indicates a {'positive' if beta[3] > 0 else 'negative'}")
            print(f"  interaction between transportation costs and waste penalty.")
            print(f"  This suggests a SUBSTITUTION EFFECT is present.")
        else:
            print(f"  The interaction term β₁₂ = {beta[3]:.2f} is relatively small,")
            print(f"  suggesting limited interaction between factors.")
        
        # Regression for Y2 (Waste)
        print("\n📈 Regression for Y2 (Total Waste)")
        print("-" * 80)
        print("Model: Y2 = β₀ + β₁·A + β₂·B + β₁₂·A·B + ε")
        
        y_waste = df['Y2_waste'].values
        beta_waste = np.linalg.lstsq(X_with_intercept, y_waste, rcond=None)[0]
        
        print(f"\nCoefficients:")
        print(f"  β₀ (Intercept):     {beta_waste[0]:>12,.2f}")
        print(f"  β₁ (Factor A):      {beta_waste[1]:>12,.2f}")
        print(f"  β₂ (Factor B):      {beta_waste[2]:>12,.2f}")
        print(f"  β₁₂ (Interaction):  {beta_waste[3]:>12,.2f}")
        
        y_pred_waste = X_with_intercept @ beta_waste
        ss_res_waste = np.sum((y_waste - y_pred_waste) ** 2)
        ss_tot_waste = np.sum((y_waste - np.mean(y_waste)) ** 2)
        r_squared_waste = 1 - (ss_res_waste / ss_tot_waste)
        
        print(f"\nModel Fit:")
        print(f"  R²: {r_squared_waste:.4f}")
        
        return {
            'Y1': {'beta': beta, 'r_squared': r_squared},
            'Y2': {'beta': beta_waste, 'r_squared': r_squared_waste}
        }
    
    def plot_results(self):
        """Generate all plots for the experiment"""
        if self.results_df is None:
            print("Error: No results available. Run experiment first.")
            return
        
        print("\n📊 Generating plots...")
        
        df = self.results_df.copy()
        
        # Calculate means for interaction plots
        means = df.groupby(['factor_A_level', 'factor_B_level']).agg({
            'Y1_profit': 'mean',
            'Y2_waste': 'mean'
        }).reset_index()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Experiment 1: Full Factorial Design 2² Results', fontsize=16, fontweight='bold')
        
        # 1. Main effects plot for Y1 (Profit)
        ax = axes[0, 0]
        means_A = df.groupby('factor_A_level')['Y1_profit'].mean()
        means_B = df.groupby('factor_B_level')['Y1_profit'].mean()
        
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, means_A.values, width, label='Factor A (c_trans)', alpha=0.8)
        ax.bar(x + width/2, means_B.values, width, label='Factor B (π)', alpha=0.8)
        ax.set_ylabel('Mean Profit ($)', fontweight='bold')
        ax.set_title('Main Effects on Profit (Y1)')
        ax.set_xticks(x)
        ax.set_xticklabels(['Low', 'High'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Interaction plot for Y1 (Profit)
        ax = axes[0, 1]
        for b_level in ['low', 'high']:
            data = means[means['factor_B_level'] == b_level]
            ax.plot(['Low', 'High'], data['Y1_profit'].values, 
                   marker='o', linewidth=2, markersize=8, 
                   label=f'B (π) = {b_level}')
        ax.set_xlabel('Factor A (c_trans)', fontweight='bold')
        ax.set_ylabel('Mean Profit ($)', fontweight='bold')
        ax.set_title('Interaction Plot for Profit (Y1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Box plot for Y1 by treatment
        ax = axes[0, 2]
        df.boxplot(column='Y1_profit', by='treatment', ax=ax)
        ax.set_xlabel('Treatment', fontweight='bold')
        ax.set_ylabel('Profit ($)', fontweight='bold')
        ax.set_title('Profit Distribution by Treatment')
        plt.sca(ax)
        plt.xticks(rotation=45)
        
        # 4. Main effects plot for Y2 (Waste)
        ax = axes[1, 0]
        means_A_waste = df.groupby('factor_A_level')['Y2_waste'].mean()
        means_B_waste = df.groupby('factor_B_level')['Y2_waste'].mean()
        
        ax.bar(x - width/2, means_A_waste.values, width, label='Factor A (c_trans)', alpha=0.8)
        ax.bar(x + width/2, means_B_waste.values, width, label='Factor B (π)', alpha=0.8)
        ax.set_ylabel('Mean Waste (kg)', fontweight='bold')
        ax.set_title('Main Effects on Waste (Y2)')
        ax.set_xticks(x)
        ax.set_xticklabels(['Low', 'High'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 5. Interaction plot for Y2 (Waste)
        ax = axes[1, 1]
        for b_level in ['low', 'high']:
            data = means[means['factor_B_level'] == b_level]
            ax.plot(['Low', 'High'], data['Y2_waste'].values, 
                   marker='o', linewidth=2, markersize=8, 
                   label=f'B (π) = {b_level}')
        ax.set_xlabel('Factor A (c_trans)', fontweight='bold')
        ax.set_ylabel('Mean Waste (kg)', fontweight='bold')
        ax.set_title('Interaction Plot for Waste (Y2)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Scatter plot: Waste vs Profit
        ax = axes[1, 2]
        for treatment in df['treatment'].unique():
            data = df[df['treatment'] == treatment]
            ax.scatter(data['Y2_waste'], data['Y1_profit'], 
                      label=treatment, alpha=0.6, s=100)
        ax.set_xlabel('Total Waste (kg)', fontweight='bold')
        ax.set_ylabel('Total Profit ($)', fontweight='bold')
        ax.set_title('Waste vs Profit Relationship')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiment_1_results.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: experiment_1_results.png")
        plt.show()
        
        # Additional heatmap for mean responses
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Mean Response Heatmaps', fontsize=14, fontweight='bold')
        
        # Heatmap for Y1 (Profit)
        pivot_profit = means.pivot(index='factor_B_level', columns='factor_A_level', values='Y1_profit')
        pivot_profit = pivot_profit.reindex(['high', 'low'])
        pivot_profit = pivot_profit[['low', 'high']]
        
        sns.heatmap(pivot_profit, annot=True, fmt='.2f', cmap='RdYlGn', 
                   ax=axes[0], cbar_kws={'label': 'Mean Profit ($)'})
        axes[0].set_title('Y1: Mean Profit by Factor Levels')
        axes[0].set_xlabel('Factor A (c_trans)')
        axes[0].set_ylabel('Factor B (π)')
        
        # Heatmap for Y2 (Waste)
        pivot_waste = means.pivot(index='factor_B_level', columns='factor_A_level', values='Y2_waste')
        pivot_waste = pivot_waste.reindex(['high', 'low'])
        pivot_waste = pivot_waste[['low', 'high']]
        
        sns.heatmap(pivot_waste, annot=True, fmt='.2f', cmap='YlOrRd', 
                   ax=axes[1], cbar_kws={'label': 'Mean Waste (kg)'})
        axes[1].set_title('Y2: Mean Waste by Factor Levels')
        axes[1].set_xlabel('Factor A (c_trans)')
        axes[1].set_ylabel('Factor B (π)')
        
        plt.tight_layout()
        plt.savefig('experiment_1_heatmaps.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: experiment_1_heatmaps.png")
        plt.show()
    
    def export_results(self, filename='experiment_1_results.csv'):
        """Export results to CSV"""
        if self.results_df is not None:
            self.results_df.to_csv(filename, index=False)
            print(f"\n✓ Results exported to '{filename}'")
        else:
            print("Error: No results to export.")
    
    def generate_report(self):
        """Generate comprehensive experiment report"""
        print("\n" + "="*80)
        print("EXPERIMENT 1: COMPREHENSIVE REPORT")
        print("="*80)
        
        # Summary statistics
        self.calculate_summary_statistics()
        
        # ANOVA
        anova_results = self.perform_anova()
        
        # Regression
        reg_results = self.perform_regression_analysis()
        
        # Conclusions
        print("\n" + "="*80)
        print("CONCLUSIONS")
        print("="*80)
        
        df = self.results_df.copy()
        
        # Best treatment
        best_treatment = df.groupby('treatment')['Y1_profit'].mean().idxmax()
        best_profit = df.groupby('treatment')['Y1_profit'].mean().max()
        
        print(f"\n✓ Best treatment: {best_treatment}")
        print(f"  Average profit: ${best_profit:,.2f}")
        
        # Worst treatment
        worst_treatment = df.groupby('treatment')['Y1_profit'].mean().idxmin()
        worst_profit = df.groupby('treatment')['Y1_profit'].mean().min()
        
        print(f"\n✓ Worst treatment: {worst_treatment}")
        print(f"  Average profit: ${worst_profit:,.2f}")
        
        # Profit range
        print(f"\n✓ Profit range: ${worst_profit:,.2f} to ${best_profit:,.2f}")
        print(f"  Difference: ${best_profit - worst_profit:,.2f} ({(best_profit - worst_profit)/worst_profit*100:.1f}%)")
        
        # Waste analysis
        min_waste_treatment = df.groupby('treatment')['Y2_waste'].mean().idxmin()
        min_waste = df.groupby('treatment')['Y2_waste'].mean().min()
        max_waste = df.groupby('treatment')['Y2_waste'].mean().max()
        
        print(f"\n✓ Minimum average waste: {min_waste:.2f} kg (treatment: {min_waste_treatment})")
        print(f"✓ Maximum average waste: {max_waste:.2f} kg")
        
        # Substitution effect
        interaction_coef = reg_results['Y1']['beta'][3]
        print(f"\n✓ Substitution effect (β₁₂ for Y1): {interaction_coef:.2f}")
        if abs(interaction_coef) > 100:
            print("  → Strong interaction detected!")
            print("  → The effect of transportation cost on profit DEPENDS on the penalty level.")
        else:
            print("  → Weak interaction detected.")
            print("  → Factors act mostly independently.")
        
        print("\n" + "="*80)


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("FOOD DONATION OPTIMIZATION - FACTORIAL EXPERIMENT 2²")
    print("="*80)
    
    # Configuration
    JSON_FILE = 'instance_distance_exp.json'
    N_REPLICATES = 5
    
    # Create experiment
    experiment = FactorialExperiment(JSON_FILE, n_replicates=N_REPLICATES)
    
    # Run experiment
    experiment.run_experiment()
    
    # Generate full report
    experiment.generate_report()
    
    # Generate plots
    experiment.plot_results()
    
    # Export results
    experiment.export_results('experiment_1_results.csv')
    
    print("\n" + "="*80)
    print("✓ EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print("  - experiment_1_results.csv (raw data)")
    print("  - experiment_1_results.png (main plots)")
    print("  - experiment_1_heatmaps.png (heatmaps)")
    print("\n")


if __name__ == "__main__":
    main()



