# %%
"""
EXPERIMENT 1: Full 2² Factorial Design
Sensitivity of Profit to Operating Costs and Penalties

Objective: Determine how transportation costs and waste penalties 
affect system profitability, identifying whether a "substitution effect" exists.

Research question: To what extent does the interaction between own transportation 
cost and food waste penalty condition system profitability?

Factors:
- Factor A: Own transportation cost (ctrans) - Low vs. High
- Factor B: Destruction penalty (pi) - Low vs. High

Response Variables:
- Y1: Total Net Benefit (Objective Function Z)
- Y2: Total amount of food destroyed (wr)
"""

import gurobipy as gp
from gurobipy import GRB
import json
import numpy as np
import pandas as pd
from itertools import product
import time
from copy import deepcopy
import sys
import os


class FoodDonationOptimizer:
    """
    Optimizer for food donation network with direct and indirect routes.
    
    Direct routes: Origin -> Food Bank (same day)
    Indirect routes: Client -> Distribution Center -> Food Bank (two days)
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
        
        # Load requirements and classify them
        self.requirements = data['requirements']
        
        # Create dictionary for O(1) lookup
        for req in self.requirements:
            self.requirements_dict[req['id']] = req
        
        for req in self.requirements:
            req_id = req['id']
            origin_type = req['origin_type']
            
            if origin_type in ['Manufacturing', 'DistributionCenter']:
                self.R_direct.append(req_id)
            else:
                self.R_indirect.append(req_id)
        
        # Load sets
        self.T = list(range(
            data['planning_horizon']['start_period'],
            data['planning_horizon']['end_period'] + 1
        ))
        self.B = data['sets']['food_banks']
        self.P = data['sets']['products']
        
        # Load parameters
        params = data['parameters']
        self.beta = params['beta_1']
        self.pi = params['pi_penalty']
        self.c_trans = params['c_trans']
        self.alpha = params['alpha_outsource']
        self.product_costs = params['product_costs']
        self.food_bank_capacity = params['food_bank_capacity']
        
        # Transport capacity
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
        # Variables for DIRECT requirements
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
        
        # Variables for INDIRECT requirements
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
        
        # Waste variables
        for r in self.R_direct + self.R_indirect:
            self.w[r] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"w_{r}")
    
    def _set_objective(self):
        """Set the objective function to maximize net benefit."""
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
                # Tax benefit on products
                product_benefit_terms.append(self.beta * c_prod * self.y_deliv[r, t])
                product_benefit_terms.append(self.beta * c_prod * self.x_out[r, t])
                
                # Tax benefit on transport
                transport_benefit_terms.append(self.beta * self.c_trans * dist * self.y_deliv[r, t])
                transport_benefit_terms.append(self.beta * self.c_trans * dist * self.alpha * self.x_out[r, t])
                
                # Transport costs
                transport_cost_terms.append(self.c_trans * dist * self.y_deliv[r, t])
                
                # Outsourcing costs
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
                # Tax benefit on products
                product_benefit_terms.append(self.beta * c_prod * self.y_deliv[r, t])
                
                # Tax benefit on transport
                transport_benefit_terms.append(self.beta * self.c_trans * dist_indir * self.y_deliv[r, t])
                
                # Transport costs
                transport_cost_terms.append(self.c_trans * dist_indir * self.y_deliv[r, t])
        
        # Waste penalties
        waste_terms = [self.pi * self.w[r] for r in self.R_direct + self.R_indirect]
        
        # Combine all components
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
        """
        Extract results from the solved model.
        
        Returns:
            dict: Dictionary with objective value and total waste
        """
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        
        if self.model.SolCount == 0:
            return None
        
        total_waste = sum(self.w[r].X for r in self.R_direct + self.R_indirect)
        
        return {
            'objective_value': self.model.ObjVal,
            'total_waste': total_waste,
            'status': self.model.Status
        }


class FactorialExperiment:
    """
    Class to run and analyze a full 2² factorial design.
    """
    
    def __init__(self, instance_file, output_dir='experiment_results'):
        """
        Initialize the factorial experiment.
        
        Args:
            instance_file: Path to the JSON file with instance data
            output_dir: Directory to save the results
        """
        self.instance_file = instance_file
        self.output_dir = output_dir
        self.results = []
        self.results_df = None

        # Create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load data from file to get base values for c_trans and pi
        with open(instance_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        params = data.get('parameters', {})
        self.base_c_trans = params.get('c_trans', 0.0)
        self.base_pi = params.get('pi_penalty', 0.0)
    
    def define_factor_levels(self):
        """
        Define the factor levels for the experiment.
        Factor A: Transportation cost (ctrans)
        Factor B: Destruction penalty (pi)
        """
        # Use base values loaded from the file
        factor_levels = {
            'c_trans': {
                'Low': self.base_c_trans * 0.10,   # 10% of base value
                'High': self.base_c_trans * 3.0     # 300% of base value
            },
            'pi_penalty': {
                'Low': self.base_pi * 0.10,
                'High': self.base_pi * 3.0
            }
        }
        return factor_levels
    
    def run_experiment(self, replications=1, time_limit=300):
        """
        Run the full 2² factorial design.
        
        Args:
            replications: Number of replications per treatment
            time_limit: Time limit per run (seconds)
        """
        print("=" * 80)
        print("EXPERIMENT 1: FULL 2² FACTORIAL DESIGN")
        print("Sensitivity of Profit to Operating Costs and Penalties")
        print("=" * 80)
        
        factor_levels = self.define_factor_levels()
        
        # Generate all factor combinations (2² = 4 treatments)
        c_trans_levels = list(factor_levels['c_trans'].items())
        pi_levels = list(factor_levels['pi_penalty'].items())
        
        treatments = list(product(c_trans_levels, pi_levels))
        
        print(f"\nNumber of treatments: {len(treatments)}")
        print(f"Replications per treatment: {replications}")
        print(f"Total runs: {len(treatments) * replications}")
        print(f"Time limit per run: {time_limit}s")
        
        # Run each treatment
        run_number = 1
        total_runs = len(treatments) * replications
        
        for (c_trans_label, c_trans_value), (pi_label, pi_value) in treatments:
            print(f"\n{'='*80}")
            print(f"TREATMENT: ctrans={c_trans_label}, pi={pi_label}")
            print(f"  c_trans = {c_trans_value} $/kg-km")
            print(f"  pi = {pi_value} $")
            print(f"{'='*80}")
            
            for rep in range(1, replications + 1):
                print(f"\nRun {run_number}/{total_runs} (Replication {rep}/{replications})")
                
                start_time = time.time()
                
                # Create Gurobi environment
                with gp.Env(empty=True) as env:
                    env.setParam('OutputFlag', 0)  # Suppress solver output
                    env.setParam('TimeLimit', time_limit)
                    env.start()
                    
                    # Create and configure the optimizer
                    with FoodDonationOptimizer(env) as optimizer:
                        try:
                            # Load data
                            optimizer.load_data_from_json(self.instance_file)
                            
                            # Modify parameters according to the treatment
                            optimizer.c_trans = c_trans_value
                            optimizer.pi = pi_value
                            
                            # Build and solve model
                            optimizer.build_model()
                            status = optimizer.optimize()
                            
                            # Extract results
                            results = optimizer.get_results()
                            
                            elapsed_time = time.time() - start_time
                            
                            if results is not None:
                                # Store results
                                result_record = {
                                    'run': run_number,
                                    'replication': rep,
                                    'c_trans_level': c_trans_label,
                                    'c_trans_value': c_trans_value,
                                    'pi_level': pi_label,
                                    'pi_value': pi_value,
                                    'treatment': f"{c_trans_label}_{pi_label}",
                                    'Y1_benefit': results['objective_value'],
                                    'Y2_waste': results['total_waste'],
                                    'solve_time': elapsed_time,
                                    'status': 'Optimal' if status == GRB.OPTIMAL else 'Time_Limit'
                                }
                                
                                self.results.append(result_record)
                                
                                print(f"  ✓ Net Benefit (Y1): ${results['objective_value']:,.2f}")
                                print(f"  ✓ Total Waste (Y2): {results['total_waste']:,.2f} kg")
                                print(f"  ✓ Solution time: {elapsed_time:.2f}s")
                            else:
                                print(f"  ✗ No feasible solution found")
                                
                        except Exception as e:
                            print(f"  ✗ Error: {e}")
                
                run_number += 1
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETED")
        print(f"{'='*80}")
        print(f"Total successful runs: {len(self.results)}/{total_runs}")
    
    def analyze_results(self):
        """
        Analyze the results of the factorial experiment.
        Calculate main effects and interactions.
        """
        if self.results_df is None or len(self.results_df) == 0:
            print("No results to analyze")
            return
        
        print(f"\n{'='*80}")
        print("RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        # Descriptive statistics by treatment
        print("\n1. DESCRIPTIVE STATISTICS BY TREATMENT")
        print("-" * 80)
        
        grouped = self.results_df.groupby(['c_trans_level', 'pi_level']).agg({
            'Y1_benefit': ['mean', 'std', 'min', 'max'],
            'Y2_waste': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        print(grouped)
        
        # Calculate main effects and interactions using -1, +1 coding
        print("\n2. MAIN EFFECTS AND INTERACTIONS")
        print("-" * 80)
        
        # Encode factors: Low = -1, High = +1
        df_coded = self.results_df.copy()
        df_coded['A'] = df_coded['c_trans_level'].map({'Low': -1, 'High': 1})
        df_coded['B'] = df_coded['pi_level'].map({'Low': -1, 'High': 1})
        df_coded['AB'] = df_coded['A'] * df_coded['B']
        
        # Averages by factor level
        mean_Y1_low_A = df_coded[df_coded['A'] == -1]['Y1_benefit'].mean()
        mean_Y1_high_A = df_coded[df_coded['A'] == 1]['Y1_benefit'].mean()
        mean_Y1_low_B = df_coded[df_coded['B'] == -1]['Y1_benefit'].mean()
        mean_Y1_high_B = df_coded[df_coded['B'] == 1]['Y1_benefit'].mean()
        
        mean_Y2_low_A = df_coded[df_coded['A'] == -1]['Y2_waste'].mean()
        mean_Y2_high_A = df_coded[df_coded['A'] == 1]['Y2_waste'].mean()
        mean_Y2_low_B = df_coded[df_coded['B'] == -1]['Y2_waste'].mean()
        mean_Y2_high_B = df_coded[df_coded['B'] == 1]['Y2_waste'].mean()
        
        # Calculate main effects
        effect_A_Y1 = mean_Y1_high_A - mean_Y1_low_A
        effect_B_Y1 = mean_Y1_high_B - mean_Y1_low_B
        
        effect_A_Y2 = mean_Y2_high_A - mean_Y2_low_A
        effect_B_Y2 = mean_Y2_high_B - mean_Y2_low_B
        
        # Calculate AB interaction
        mean_both_high = df_coded[(df_coded['A'] == 1) & (df_coded['B'] == 1)]['Y1_benefit'].mean()
        mean_both_low = df_coded[(df_coded['A'] == -1) & (df_coded['B'] == -1)]['Y1_benefit'].mean()
        mean_A_high_B_low = df_coded[(df_coded['A'] == 1) & (df_coded['B'] == -1)]['Y1_benefit'].mean()
        mean_A_low_B_high = df_coded[(df_coded['A'] == -1) & (df_coded['B'] == 1)]['Y1_benefit'].mean()
        
        interaction_AB_Y1 = ((mean_both_high + mean_both_low) - (mean_A_high_B_low + mean_A_low_B_high)) / 2
        
        mean_both_high_Y2 = df_coded[(df_coded['A'] == 1) & (df_coded['B'] == 1)]['Y2_waste'].mean()
        mean_both_low_Y2 = df_coded[(df_coded['A'] == -1) & (df_coded['B'] == -1)]['Y2_waste'].mean()
        mean_A_high_B_low_Y2 = df_coded[(df_coded['A'] == 1) & (df_coded['B'] == -1)]['Y2_waste'].mean()
        mean_A_low_B_high_Y2 = df_coded[(df_coded['A'] == -1) & (df_coded['B'] == 1)]['Y2_waste'].mean()
        
        interaction_AB_Y2 = ((mean_both_high_Y2 + mean_both_low_Y2) - (mean_A_high_B_low_Y2 + mean_A_low_B_high_Y2)) / 2
        
        print("\nEffects on Y1 (Net Benefit):")
        print(f"  Main Effect A (transport cost): {effect_A_Y1:,.2f}")
        print(f"  Main Effect B (penalty): {effect_B_Y1:,.2f}")
        print(f"  Interaction Effect AB: {interaction_AB_Y1:,.2f}")
        
        print("\nEffects on Y2 (Total Waste):")
        print(f"  Main Effect A (transport cost): {effect_A_Y2:,.2f} kg")
        print(f"  Main Effect B (penalty): {effect_B_Y2:,.2f} kg")
        print(f"  Interaction Effect AB: {interaction_AB_Y2:,.2f} kg")
        
        # Interpretation
        print("\n3. INTERPRETATION OF RESULTS")
        print("-" * 80)
        
        print("\nNet Benefit (Y1):")
        if effect_A_Y1 < 0:
            print(f"  • Increasing transport cost REDUCES benefit by ${abs(effect_A_Y1):,.2f}")
        else:
            print(f"  • Increasing transport cost INCREASES benefit by ${effect_A_Y1:,.2f}")
        
        if effect_B_Y1 < 0:
            print(f"  • Increasing penalty REDUCES benefit by ${abs(effect_B_Y1):,.2f}")
        else:
            print(f"  • Increasing penalty INCREASES benefit by ${effect_B_Y1:,.2f}")
        
        if abs(interaction_AB_Y1) > 0.1 * max(abs(effect_A_Y1), abs(effect_B_Y1)):
            print(f"  • There is a SIGNIFICANT interaction (AB = ${interaction_AB_Y1:,.2f})")
            print("    The effect of one factor depends on the level of the other factor")
        else:
            print("  • There is no significant interaction between factors")
        
        print("\nTotal Waste (Y2):")
        if effect_A_Y2 < 0:
            print(f"  • Increasing transport cost REDUCES waste by {abs(effect_A_Y2):,.2f} kg")
        else:
            print(f"  • Increasing transport cost INCREASES waste by {effect_A_Y2:,.2f} kg")
        
        if effect_B_Y2 < 0:
            print(f"  • Increasing penalty REDUCES waste by {abs(effect_B_Y2):,.2f} kg")
        else:
            print(f"  • Increasing penalty INCREASES waste by {effect_B_Y2:,.2f} kg")
        
        # Substitution analysis
        print("\n4. SUBSTITUTION EFFECT ANALYSIS")
        print("-" * 80)
        
        # Compare extreme scenarios
        scenario_low_trans_low_penalty = df_coded[(df_coded['A'] == -1) & (df_coded['B'] == -1)]['Y1_benefit'].mean()
        scenario_high_trans_low_penalty = df_coded[(df_coded['A'] == 1) & (df_coded['B'] == -1)]['Y1_benefit'].mean()
        scenario_low_trans_high_penalty = df_coded[(df_coded['A'] == -1) & (df_coded['B'] == 1)]['Y1_benefit'].mean()
        scenario_high_trans_high_penalty = df_coded[(df_coded['A'] == 1) & (df_coded['B'] == 1)]['Y1_benefit'].mean()
        
        print(f"\nNet Benefit by Scenario:")
        print(f"  • Low transport cost + Low penalty: ${scenario_low_trans_low_penalty:,.2f}")
        print(f"  • High transport cost + Low penalty: ${scenario_high_trans_low_penalty:,.2f}")
        print(f"  • Low transport cost + High penalty: ${scenario_low_trans_high_penalty:,.2f}")
        print(f"  • High transport cost + High penalty: ${scenario_high_trans_high_penalty:,.2f}")
        
        # Evaluate if a substitution effect exists
        trade_off_1 = scenario_high_trans_low_penalty - scenario_low_trans_high_penalty
        
        print("\nDoes a substitution effect exist?")
        if trade_off_1 > 0:
            print(f"  • YES: Paying high transport cost with low penalty is BETTER")
            print(f"    than low transport cost with high penalty")
            print(f"    Difference: ${trade_off_1:,.2f}")
        else:
            print(f"  • NO: Low transport cost with high penalty is BETTER")
            print(f"    than high transport cost with low penalty")
            print(f"    Difference: ${abs(trade_off_1):,.2f}")
        
        # Save analysis to file
        analysis_file = f"{self.output_dir}/effects_analysis.txt"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("EFFECTS ANALYSIS - 2² FACTORIAL DESIGN\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Main Effect A (transport cost) on Y1: {effect_A_Y1:,.2f}\n")
            f.write(f"Main Effect B (penalty) on Y1: {effect_B_Y1:,.2f}\n")
            f.write(f"Interaction Effect AB on Y1: {interaction_AB_Y1:,.2f}\n\n")
            f.write(f"Main Effect A (transport cost) on Y2: {effect_A_Y2:,.2f} kg\n")
            f.write(f"Main Effect B (penalty) on Y2: {effect_B_Y2:,.2f} kg\n")
            f.write(f"Interaction Effect AB on Y2: {interaction_AB_Y2:,.2f} kg\n")
        
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
        
        # Export to CSV
        csv_file = f"{self.output_dir}/experiment_results.csv"
        self.results_df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✓ Results exported to CSV: {csv_file}")
        
        # Export to JSON
        json_file = f"{self.output_dir}/experiment_results.json"
        self.results_df.to_json(json_file, orient='records', indent=2)
        print(f"✓ Results exported to JSON: {json_file}")
        
        # Export statistical summary
        summary = self.results_df.groupby(['c_trans_level', 'pi_level']).agg({
            'Y1_benefit': ['mean', 'std', 'min', 'max', 'count'],
            'Y2_waste': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        summary_file = f"{self.output_dir}/statistical_summary.csv"
        summary.to_csv(summary_file, encoding='utf-8')
        print(f"✓ Statistical summary exported: {summary_file}")
    
    def generate_report(self):
        """
        Generate a complete experiment report in Markdown format.
        """
        if self.results_df is None or len(self.results_df) == 0:
            print("No results to generate report")
            return
        
        print(f"\n{'='*80}")
        print("GENERATING REPORT")
        print(f"{'='*80}")
        
        report_file = f"{self.output_dir}/experiment_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# EXPERIMENT 1: Full 2² Factorial Design\n\n")
            f.write("## Sensitivity of Profit to Operating Costs and Penalties\n\n")
            
            f.write("### 1. Experiment Objective\n\n")
            f.write("To determine how transport costs and waste penalties ")
            f.write("affect system profitability, identifying whether a \"substitution effect\" exists ")
            f.write("where it is preferable to pay a high transport cost rather than incurring ")
            f.write("destruction penalties.\n\n")
            
            f.write("### 2. Research Question\n\n")
            f.write("To what extent does the interaction between own transportation ")
            f.write("cost and food waste penalty condition the profitability of the donation ")
            f.write("system? Is there a cost threshold where food destruction becomes ")
            f.write("economically superior to transportation?\n\n")
            
            f.write("### 3. Experimental Design\n\n")
            f.write("**Design Type:** Full 2² Factorial\n\n")
            
            f.write("**Factors and Levels:**\n\n")
            factor_levels = self.define_factor_levels()
            f.write(f"- **Factor A (Transportation cost):**\n")
            f.write(f"  - Low: {factor_levels['c_trans']['Low']} $/kg-km\n")
            f.write(f"  - High: {factor_levels['c_trans']['High']} $/kg-km\n\n")
            f.write(f"- **Factor B (Destruction penalty):**\n")
            f.write(f"  - Low: {factor_levels['pi_penalty']['Low']} $ (marginal cost)\n")
            f.write(f"  - High: {factor_levels['pi_penalty']['High']} $ (punitive cost)\n\n")
            
            f.write("**Response Variables:**\n\n")
            f.write("- **Y1:** Total Net Benefit (Objective Function Z)\n")
            f.write("- **Y2:** Total amount of food destroyed (kg)\n\n")
            
            f.write("**Scenario:**\n\n")
            f.write("- 30-day operation\n")
            f.write("- Constant demand from direct and indirect requirements\n")
            f.write("- Tax incentives at medium level (β = 0.35)\n")
            f.write("- Food Bank capacities at 100%\n")
            f.write("- Origins evenly distributed between factories (M) and clients (C)\n\n")
            
            f.write("### 4. Results\n\n")
            
            # Table of average results
            grouped = self.results_df.groupby(['c_trans_level', 'pi_level']).agg({
                'Y1_benefit': ['mean', 'std'],
                'Y2_waste': ['mean', 'std']
            }).round(2)
            
            f.write("#### Table 1: Average Results by Treatment\n\n")
            f.write("| Trans. Cost | Penalty | Net Benefit (Y1) | Std. Dev. | Waste (Y2) | Std. Dev. |\n")
            f.write("|-------------|---------|------------------|-----------|------------|-----------|\n")
            
            for idx, row in grouped.iterrows():
                c_trans_level, pi_level = idx
                f.write(f"| {c_trans_level:11} | {pi_level:7} | ")
                f.write(f"${row[('Y1_benefit', 'mean')]:>16,.2f} | ")
                f.write(f"${row[('Y1_benefit', 'std')]:>9,.2f} | ")
                f.write(f"{row[('Y2_waste', 'mean')]:>10,.2f} | ")
                f.write(f"{row[('Y2_waste', 'std')]:>9,.2f} |\n")
            
            f.write("\n")
            
            # Calculate effects
            df_coded = self.results_df.copy()
            df_coded['A'] = df_coded['c_trans_level'].map({'Low': -1, 'High': 1})
            df_coded['B'] = df_coded['pi_level'].map({'Low': -1, 'High': 1})
            
            mean_Y1_low_A = df_coded[df_coded['A'] == -1]['Y1_benefit'].mean()
            mean_Y1_high_A = df_coded[df_coded['A'] == 1]['Y1_benefit'].mean()
            mean_Y1_low_B = df_coded[df_coded['B'] == -1]['Y1_benefit'].mean()
            mean_Y1_high_B = df_coded[df_coded['B'] == 1]['Y1_benefit'].mean()
            
            effect_A_Y1 = mean_Y1_high_A - mean_Y1_low_A
            effect_B_Y1 = mean_Y1_high_B - mean_Y1_low_B
            
            mean_Y2_low_A = df_coded[df_coded['A'] == -1]['Y2_waste'].mean()
            mean_Y2_high_A = df_coded[df_coded['A'] == 1]['Y2_waste'].mean()
            mean_Y2_low_B = df_coded[df_coded['B'] == -1]['Y2_waste'].mean()
            mean_Y2_high_B = df_coded[df_coded['B'] == 1]['Y2_waste'].mean()
            
            effect_A_Y2 = mean_Y2_high_A - mean_Y2_low_A
            effect_B_Y2 = mean_Y2_high_B - mean_Y2_low_B
            
            f.write("#### Table 2: Main Effects\n\n")
            f.write("| Factor | Effect on Y1 (Benefit) | Effect on Y2 (Waste) |\n")
            f.write("|--------|-------------------------|----------------------|\n")
            f.write(f"| Factor A (Trans. Cost) | ${effect_A_Y1:>22,.2f} | {effect_A_Y2:>20,.2f} kg |\n")
            f.write(f"| Factor B (Penalty)     | ${effect_B_Y1:>22,.2f} | {effect_B_Y2:>20,.2f} kg |\n")
            
            f.write("\n")
            
            f.write("### 5. Conclusions\n\n")
            
            if effect_A_Y1 < 0:
                f.write(f"1. **Effect of transport cost:** Increasing transport cost ")
                f.write(f"reduces net benefit by **${abs(effect_A_Y1):,.2f}**. ")
                f.write("This indicates that higher transport costs decrease system profitability.\n\n")
            
            if effect_B_Y1 < 0:
                f.write(f"2. **Effect of penalty:** Increasing the waste penalty ")
                f.write(f"reduces net benefit by **${abs(effect_B_Y1):,.2f}**. ")
                f.write("However, this incentivizes better waste management.\n\n")
            
            if effect_B_Y2 < 0:
                f.write(f"3. **Impact on waste:** A higher penalty reduces waste by ")
                f.write(f"**{abs(effect_B_Y2):,.2f} kg**, demonstrating that penalties are effective ")
                f.write("in minimizing food waste.\n\n")
            
            f.write("### 6. Recommendations\n\n")
            f.write("- Seek a balance between transport costs and penalties\n")
            f.write("- Consider investments in logistics to reduce transport costs\n")
            f.write("- Implement monitoring systems to minimize waste\n\n")
            
            f.write("---\n\n")
            f.write(f"*Report automatically generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        print(f"✓ Report generated at: {report_file}")


def main():
    """
    Main function to run the factorial experiment.
    """
    print("=" * 80)
    print("EXPERIMENTATION SYSTEM - 2² FACTORIAL DESIGN")
    print("Food Donation Optimization")
    print("=" * 80)
    
    # Experiment configuration
    instance_file = "instance_distance_exp.json"
    output_dir = "experiment_results"
    
    # Create experiment
    experiment = FactorialExperiment(instance_file, output_dir)
    
    try:
        # Run experiment
        # For quick tests: replications=1, time_limit=60
        # For robust results: replications=3, time_limit=300
        experiment.run_experiment(replications=1, time_limit=60)
        
        # Analyze results
        experiment.analyze_results()
        
        # Export results
        experiment.export_results()
        
        # Generate report
        experiment.generate_report()
        
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"\nAll results are in the directory: {output_dir}/")
        print("\nGenerated files:")
        print(f"  • experiment_results.csv - Complete data")
        print(f"  • experiment_results.json - Data in JSON format")
        print(f"  • statistical_summary.csv - Summary by treatment")
        print(f"  • effects_analysis.txt - Main effects analysis")
        print(f"  • experiment_report.md - Complete report in Markdown")
        
    except FileNotFoundError:
        print(f"\n❌ Error: File '{instance_file}' not found")
        print("Please ensure the instance file exists in the current directory.")
    except Exception as e:
        print(f"\n❌ Error during experiment execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
