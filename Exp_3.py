"""
3² FACTORIAL EXPERIMENT - INTERACTION BETWEEN TAX INCENTIVES AND TRANSPORT CAPACITY

This experiment analyzes the joint effect of:
- Factor 1: Tax benefit (β) - 3 levels
- Factor 2: Own transport capacity - 3 levels

Total: 9 experimental scenarios (3×3)

Response variables:
1. Total net benefit
2. Compliance rate
3. Outsourcing percentage

Analysis:
- Interaction plots
- Eta squared (η²) to quantify effect size
- Elasticity matrix
"""

import gurobipy as gp
from gurobipy import GRB
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
import time
import os


# ==============================================================================
# OPTIMIZER CLASS (COPIED FROM ORIGINAL MODEL)
# ==============================================================================

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
            else:  # Client
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
        """Helper function to get requirement data by ID (O(1) lookup)."""
        return self.requirements_dict.get(req_id)
    
    def build_model(self):
        """Build the optimization model."""
        # Create model
        self.model = gp.Model("FoodDonationOptimization", env=self.env)
        
        # Create variables
        self._create_variables()
        
        # Set objective
        self._set_objective()
        
        # Add constraints
        self._add_constraints()
        
        # Update model
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
                    lb=0, vtype=GRB.CONTINUOUS,
                    name=f"y_deliv_D_{r}_{t}"
                )
                
                self.x_out[r, t] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS,
                    name=f"x_out_{r}_{t}"
                )
        
        # Variables for INDIRECT requirements
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r):
                self.y_pickup[r, t] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS,
                    name=f"y_pickup_{r}_{t}"
                )
            
            for t in range(l_r + 1, e_r + 1):
                self.y_deliv[r, t] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS,
                    name=f"y_deliv_I_{r}_{t}"
                )
        
        # Waste variables
        for r in self.R_direct + self.R_indirect:
            self.w[r] = self.model.addVar(
                lb=0, vtype=GRB.CONTINUOUS,
                name=f"w_{r}"
            )
    
    def _set_objective(self):
        """Set the objective function to maximize net benefit."""
        product_benefit_terms = []
        
        # Direct requirements
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            benefit_coef = self.beta * c_prod
            
            for t in range(l_r, e_r + 1):
                product_benefit_terms.append(benefit_coef * self.y_deliv[r, t])
                product_benefit_terms.append(benefit_coef * self.x_out[r, t])
        
        # Indirect requirements
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            benefit_coef = self.beta * c_prod
            
            for t in range(l_r + 1, e_r + 1):
                product_benefit_terms.append(benefit_coef * self.y_deliv[r, t])
        
        # Transportation benefit terms
        transport_benefit_terms = []
        
        # Direct requirements
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            benefit_coef = self.beta * self.c_trans * dist
            
            for t in range(l_r, e_r + 1):
                transport_benefit_terms.append(benefit_coef * self.y_deliv[r, t])
                transport_benefit_terms.append(benefit_coef * self.alpha * self.x_out[r, t])
        
        # Indirect requirements
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            benefit_coef = self.beta * self.c_trans * dist_indir
            
            for t in range(l_r + 1, e_r + 1):
                transport_benefit_terms.append(benefit_coef * self.y_deliv[r, t])
        
        # Transportation cost terms
        transport_cost_terms = []
        
        # Direct requirements
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            cost_coef = self.c_trans * dist
            
            for t in range(l_r, e_r + 1):
                transport_cost_terms.append(cost_coef * self.y_deliv[r, t])
        
        # Indirect requirements
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            cost_coef = self.c_trans * dist_indir
            
            for t in range(l_r + 1, e_r + 1):
                transport_cost_terms.append(cost_coef * self.y_deliv[r, t])
        
        # Outsourcing cost terms
        outsource_cost_terms = []
        
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            cost_coef = self.alpha * self.c_trans * dist
            
            for t in range(l_r, e_r + 1):
                outsource_cost_terms.append(cost_coef * self.x_out[r, t])
        
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
                gp.quicksum(
                    self.y_deliv[r, t] + self.x_out[r, t]
                    for t in range(l_r, e_r + 1)
                ) + self.w[r] == q_r,
                name=f"flow_direct_{r}"
            )
        
        # Flow conservation for INDIRECT requirements
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            q_r = req['quantity']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            self.model.addConstr(
                gp.quicksum(
                    self.y_deliv[r, t]
                    for t in range(l_r + 1, e_r + 1)
                ) + self.w[r] == q_r,
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
    
    def get_metrics(self):
        """Calculate and return key metrics."""
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        
        if self.model.SolCount == 0:
            return None
        
        metrics = {
            'objetivo': self.model.ObjVal,
            'total_disponible': 0,
            'total_donado': 0,
            'total_desperdiciado': 0,
            'total_subcontratado': 0,
            'tasa_cumplimiento': 0,
            'pct_subcontratacion': 0
        }
        
        # Calculate totals
        for r in self.R_direct + self.R_indirect:
            req = self._get_requirement_by_id(r)
            metrics['total_disponible'] += req['quantity']
            metrics['total_desperdiciado'] += self.w[r].X
        
        # Direct deliveries
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                metrics['total_donado'] += self.y_deliv[r, t].X
                metrics['total_donado'] += self.x_out[r, t].X
                metrics['total_subcontratado'] += self.x_out[r, t].X
        
        # Indirect deliveries
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                metrics['total_donado'] += self.y_deliv[r, t].X
        
        # Calculate rates
        if metrics['total_disponible'] > 0:
            metrics['tasa_cumplimiento'] = (metrics['total_donado'] / metrics['total_disponible']) * 100
        
        if metrics['total_donado'] > 0:
            metrics['pct_subcontratacion'] = (metrics['total_subcontratado'] / metrics['total_donado']) * 100
        
        return metrics


# ==============================================================================
# 3² FACTORIAL EXPERIMENTAL DESIGN
# ==============================================================================

class FactorialExperiment:
    """
    3² factorial experiment to analyze the interaction between:
    - Tax benefit (β)
    - Own transport capacity
    """
    
    def __init__(self, instance_file='instance_distance_exp.json'):
        """
        Initialize experiment.
        
        Args:
            instance_file: JSON file with the instance data
        """
        self.instance_file = instance_file
        self.results = []
        
        # Load base data
        with open(instance_file, 'r', encoding='utf-8') as f:
            self.base_data = json.load(f)
        
        # Save original base values
        self.base_beta = self.base_data['parameters']['beta_1']
        self.base_capacity = self.base_data['parameters']['transport_capacity_per_period']
        
        print("="*80)
        print("3² FACTORIAL EXPERIMENT - INTERACTION β × TRANSPORT CAPACITY")
        print("="*80)
        print(f"\nBase values:")
        print(f"  β (tax benefit): {self.base_beta}")
        print(f"  Transport capacity: {self.base_capacity} kg/period")
    
    def define_factorial_levels(self):
        """
        Define the 3 levels for each factor.
        
        Factor 1 (β): Low (10%), Medium (100%), High (400%)
        Factor 2 (Capacity): Limited (10%), Standard (100%), Expanded (400%)
        """
        # Factor 1 levels: Tax benefit (β)
        self.beta_levels = {
            'Low': 0.10 * self.base_beta,
            'Medium': 1.00 * self.base_beta,
            'High': 4.00 * self.base_beta
        }
        
        # Factor 2 levels: Transport capacity
        self.capacity_levels = {
            'Limited': 0.10 * self.base_capacity,
            'Standard': 1.00 * self.base_capacity,
            'Expanded': 4.00 * self.base_capacity
        }
        
        print("\n" + "-"*80)
        print("FACTORIAL LEVELS DEFINED")
        print("-"*80)
        
        print("\nFactor 1 - Tax Benefit (β):")
        for level, value in self.beta_levels.items():
            pct = (value / self.base_beta) * 100
            print(f"  {level:10s}: {value:.4f} ({pct:.0f}% of base)")
        
        print("\nFactor 2 - Transport Capacity:")
        for level, value in self.capacity_levels.items():
            pct = (value / self.base_capacity) * 100
            print(f"  {level:10s}: {value:.2f} kg ({pct:.0f}% of base)")
    
    def generate_scenarios(self):
        """
        Generate the 9 factorial combinations (3×3).
        """
        self.scenarios = []
        
        for beta_name, beta_value in self.beta_levels.items():
            for cap_name, cap_value in self.capacity_levels.items():
                scenario = {
                    'id': len(self.scenarios) + 1,
                    'beta_level': beta_name,
                    'beta_value': beta_value,
                    'capacity_level': cap_name,
                    'capacity_value': cap_value
                }
                self.scenarios.append(scenario)
        
        print(f"\n✓ {len(self.scenarios)} experimental scenarios generated")
        
        return self.scenarios
    
    def run_experiment(self, verbose=True):
        """
        Execute the 9 scenarios of the factorial design.
        
        Args:
            verbose: If True, shows detailed progress
        """
        print("\n" + "="*80)
        print("RUNNING FACTORIAL EXPERIMENT")
        print("="*80)
        
        self.results = []
        
        # Create silent Gurobi environment
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)  # Suppress output
            env.setParam('TimeLimit', 300)  # 5 minutes per scenario
            env.start()
            
            for i, scenario in enumerate(self.scenarios, 1):
                if verbose:
                    print(f"\n[{i}/{len(self.scenarios)}] Scenario {scenario['id']}: " +
                          f"β={scenario['beta_level']}, Capacity={scenario['capacity_level']}")
                
                # Modify data for this scenario
                scenario_data = deepcopy(self.base_data)
                scenario_data['parameters']['beta_1'] = scenario['beta_value']
                scenario_data['parameters']['transport_capacity_per_period'] = scenario['capacity_value']
                
                # Save temporary data
                temp_file = f'temp_scenario_{scenario["id"]}.json'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(scenario_data, f, indent=2)
                
                # Run optimization
                try:
                    with FoodDonationOptimizer(env) as optimizer:
                        optimizer.load_data_from_json(temp_file)
                        optimizer.build_model()
                        status = optimizer.optimize()
                        
                        if status == GRB.OPTIMAL:
                            metrics = optimizer.get_metrics()
                            
                            result = {
                                **scenario,
                                **metrics,
                                'status': 'OPTIMAL'
                            }
                            
                            if verbose:
                                print(f"    ✓ Benefit: ${metrics['objetivo']:,.2f}")
                                print(f"    ✓ Compliance: {metrics['tasa_cumplimiento']:.2f}%")
                                print(f"    ✓ Outsourcing: {metrics['pct_subcontratacion']:.2f}%")
                        else:
                            result = {
                                **scenario,
                                'status': 'INFEASIBLE'
                            }
                            if verbose:
                                print(f"    ✗ Infeasible model")
                    
                    self.results.append(result)
                
                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    result = {
                        **scenario,
                        'status': 'ERROR',
                        'error': str(e)
                    }
                    self.results.append(result)
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED")
        print("="*80)
        print(f"✓ {len([r for r in self.results if r.get('status') == 'OPTIMAL'])} scenarios solved optimally")
        print(f"✗ {len([r for r in self.results if r.get('status') != 'OPTIMAL'])} scenarios with issues")
    
    def create_results_dataframe(self):
        """
        Convert results to DataFrame for analysis.
        """
        df = pd.DataFrame(self.results)
        
        # Filter only optimal results
        df_opt = df[df['status'] == 'OPTIMAL'].copy()
        
        # Order by levels
        beta_order = ['Low', 'Medium', 'High']
        cap_order = ['Limited', 'Standard', 'Expanded']
        
        df_opt['beta_level'] = pd.Categorical(df_opt['beta_level'], categories=beta_order, ordered=True)
        df_opt['capacity_level'] = pd.Categorical(df_opt['capacity_level'], categories=cap_order, ordered=True)
        
        df_opt = df_opt.sort_values(['beta_level', 'capacity_level'])
        
        self.results_df = df_opt
        return df_opt
    
    def calculate_sum_of_squares(self, response_var='objetivo'):
        """
        Calculate sum of squares decomposition for ANOVA.
        
        Args:
            response_var: Variable to analyze ('objetivo', 'tasa_cumplimiento', 'pct_subcontratacion')
        
        Returns:
            dict: Sum of squares and eta squared
        """
        df = self.results_df
        
        # Grand mean
        grand_mean = df[response_var].mean()
        
        # Total sum of squares
        SS_total = ((df[response_var] - grand_mean) ** 2).sum()
        
        # Main effect of β
        means_beta = df.groupby('beta_level')[response_var].mean()
        n_per_beta = df.groupby('beta_level').size()
        SS_beta = sum(n_per_beta * (means_beta - grand_mean) ** 2)
        
        # Main effect of Capacity
        means_cap = df.groupby('capacity_level')[response_var].mean()
        n_per_cap = df.groupby('capacity_level').size()
        SS_cap = sum(n_per_cap * (means_cap - grand_mean) ** 2)
        
        # Interaction effect
        means_interaction = df.groupby(['beta_level', 'capacity_level'])[response_var].mean()
        
        SS_interaction = 0
        for (beta_lvl, cap_lvl), cell_mean in means_interaction.items():
            expected_effect = means_beta[beta_lvl] + means_cap[cap_lvl] - grand_mean
            SS_interaction += (cell_mean - expected_effect) ** 2
        
        # Residual (in this deterministic case, it should be 0)
        SS_residual = SS_total - SS_beta - SS_cap - SS_interaction
        
        # Eta squared (effect size)
        eta2_beta = SS_beta / SS_total if SS_total > 0 else 0
        eta2_cap = SS_cap / SS_total if SS_total > 0 else 0
        eta2_interaction = SS_interaction / SS_total if SS_total > 0 else 0
        
        results = {
            'SS_total': SS_total,
            'SS_beta': SS_beta,
            'SS_capacity': SS_cap,
            'SS_interaction': SS_interaction,
            'SS_residual': SS_residual,
            'eta2_beta': eta2_beta,
            'eta2_capacity': eta2_cap,
            'eta2_interaction': eta2_interaction
        }
        
        return results
    
    def generate_interaction_plots(self, output_file='interaction_plots.png'):
        """
        Generate interaction plots for the three response variables.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        variables = [
            ('objetivo', 'Total Net Benefit ($)', 'objetivo'),
            ('tasa_cumplimiento', 'Compliance Rate (%)', 'tasa_cumplimiento'),
            ('pct_subcontratacion', 'Outsourcing Percentage (%)', 'pct_subcontratacion')
        ]
        
        colors = {'Low': 'blue', 'Medium': 'orange', 'High': 'red'}
        markers = {'Low': 'o', 'Medium': 's', 'High': '^'}
        
        for ax, (var_label, ylabel, var_col) in zip(axes, variables):
            # Prepare data for the plot
            for beta_level in ['Low', 'Medium', 'High']:
                data_beta = self.results_df[self.results_df['beta_level'] == beta_level]
                
                x_pos = {'Limited': 0, 'Standard': 1, 'Expanded': 2}
                x_vals = [x_pos[cap] for cap in data_beta['capacity_level']]
                y_vals = data_beta[var_col].values
                
                ax.plot(x_vals, y_vals, 
                       marker=markers[beta_level], 
                       color=colors[beta_level],
                       linewidth=2,
                       markersize=10,
                       label=f'β {beta_level}')
            
            ax.set_xlabel('Transport Capacity', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['Limited\n(10%)', 'Standard\n(100%)', 'Expanded\n(400%)'])
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{var_label}', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Interaction plots saved to: {output_file}")
        
        return fig
    
    def calculate_elasticity_matrix(self):
        """
        Calculate elasticity matrix: percentage increase in benefit
        when going from Standard to Expanded capacity, under different β levels.
        """
        print("\n" + "="*80)
        print("ELASTICITY / EFFECT MULTIPLIER MATRIX")
        print("="*80)
        print("\nPercentage increase in net benefit when expanding the fleet")
        print("(from Standard Capacity to Expanded Capacity)")
        print("-"*80)
        
        elasticities = {}
        
        for beta_level in ['Low', 'Medium', 'High']:
            # Benefit with Standard capacity
            obj_standard = self.results_df[
                (self.results_df['beta_level'] == beta_level) &
                (self.results_df['capacity_level'] == 'Standard')
            ]['objetivo'].values[0]
            
            # Benefit with Expanded capacity
            obj_expanded = self.results_df[
                (self.results_df['beta_level'] == beta_level) &
                (self.results_df['capacity_level'] == 'Expanded')
            ]['objetivo'].values[0]
            
            # Percentage increase
            if obj_standard > 0:
                increase_pct = ((obj_expanded - obj_standard) / obj_standard) * 100
            else:
                increase_pct = 0
            
            elasticities[beta_level] = {
                'obj_standard': obj_standard,
                'obj_expanded': obj_expanded,
                'absolute_increase': obj_expanded - obj_standard,
                'percentage_increase': increase_pct
            }
            
            print(f"\nβ {beta_level}:")
            print(f"  Benefit (Standard Capacity):  ${obj_standard:>15,.2f}")
            print(f"  Benefit (Expanded Capacity): ${obj_expanded:>15,.2f}")
            print(f"  Absolute increase:             ${elasticities[beta_level]['absolute_increase']:>15,.2f}")
            print(f"  Percentage increase:           {increase_pct:>15.2f}%")
        
        # Comparative analysis
        print("\n" + "="*80)
        print("COMPARATIVE ELASTICITY ANALYSIS")
        print("="*80)
        
        increases = [elasticities[level]['percentage_increase'] for level in ['Low', 'Medium', 'High']]
        
        if increases[2] > increases[0] * 1.5:  # If High is 50% larger than Low
            print("\n✓ INTERACTION DETECTED:")
            print("  The return on expanding the fleet is substantially higher when")
            print("  tax incentives are high. Tax benefits act as a CATALYST for")
            print("  investment in own capacity.")
        else:
            print("\n✓ ADDITIVE EFFECT:")
            print("  The return on expanding the fleet is relatively constant")
            print("  regardless of the level of tax incentive.")
        
        self.elasticities = elasticities
        return elasticities
    
    def generate_full_report(self, output_file='factorial_experiment_report.txt'):
        """
        Generate a full text report with all results.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("3² FACTORIAL EXPERIMENT REPORT - INTERACTION β × CAPACITY\n")
            f.write("="*80 + "\n\n")
            
            f.write("EXPERIMENTAL DESIGN\n")
            f.write("-"*80 + "\n")
            f.write(f"Factor 1: Tax Benefit (β) - 3 levels\n")
            f.write(f"Factor 2: Transport Capacity - 3 levels\n")
            f.write(f"Total scenarios: {len(self.scenarios)}\n\n")
            
            f.write("FACTORIAL LEVELS\n")
            f.write("-"*80 + "\n")
            f.write("Tax Benefit (β):\n")
            for level, value in self.beta_levels.items():
                f.write(f"  {level}: {value:.4f}\n")
            
            f.write("\nTransport Capacity:\n")
            for level, value in self.capacity_levels.items():
                f.write(f"  {level}: {value:.2f} kg\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RESULTS BY SCENARIO\n")
            f.write("="*80 + "\n\n")
            
            for _, row in self.results_df.iterrows():
                f.write(f"Scenario {row['id']}: β={row['beta_level']}, Cap={row['capacity_level']}\n")
                f.write(f"  Net Benefit:           ${row['objetivo']:>15,.2f}\n")
                f.write(f"  Compliance Rate:       {row['tasa_cumplimiento']:>15.2f}%\n")
                f.write(f"  Outsourcing %:         {row['pct_subcontratacion']:>15.2f}%\n")
                f.write(f"  Total Donated:         {row['total_donado']:>15,.2f} kg\n")
                f.write(f"  Total Wasted:          {row['total_desperdiciado']:>15,.2f} kg\n\n")
            
            f.write("="*80 + "\n")
            f.write("EFFECTS ANALYSIS (ETA SQUARED)\n")
            f.write("="*80 + "\n\n")
            
            for var in ['objetivo', 'tasa_cumplimiento', 'pct_subcontratacion']:
                ss = self.calculate_sum_of_squares(var)
                f.write(f"{var.upper().replace('_', ' ')}:\n")
                f.write(f"  η² (β):              {ss['eta2_beta']:>10.4f} ({ss['eta2_beta']*100:.2f}%)\n")
                f.write(f"  η² (Capacity):       {ss['eta2_capacity']:>10.4f} ({ss['eta2_capacity']*100:.2f}%)\n")
                f.write(f"  η² (Interaction):    {ss['eta2_interaction']:>10.4f} ({ss['eta2_interaction']*100:.2f}%)\n\n")
            
            f.write("="*80 + "\n")
            f.write("ELASTICITY MATRIX\n")
            f.write("="*80 + "\n\n")
            
            for beta_level, data in self.elasticities.items():
                f.write(f"β {beta_level}:\n")
                f.write(f"  Increase when expanding fleet: {data['percentage_increase']:.2f}%\n\n")
        
        print(f"\n✓ Full report saved to: {output_file}")
    
    def export_results_excel(self, output_file='factorial_experiment_results.xlsx'):
        """
        Export all results to an Excel file with multiple sheets.
        """
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: Complete results
            self.results_df.to_excel(writer, sheet_name='Results', index=False)
            
            # Sheet 2: Benefit matrix (pivot table)
            pivot_benefit = self.results_df.pivot(
                index='capacity_level',
                columns='beta_level',
                values='objetivo'
            )
            pivot_benefit.to_excel(writer, sheet_name='Net_Benefit')
            
            # Sheet 3: Compliance matrix
            pivot_compliance = self.results_df.pivot(
                index='capacity_level',
                columns='beta_level',
                values='tasa_cumplimiento'
            )
            pivot_compliance.to_excel(writer, sheet_name='Compliance_Rate')
            
            # Sheet 4: Outsourcing matrix
            pivot_outsource = self.results_df.pivot(
                index='capacity_level',
                columns='beta_level',
                values='pct_subcontratacion'
            )
            pivot_outsource.to_excel(writer, sheet_name='Outsourcing_Pct')
            
            # Sheet 5: Eta squared
            eta_data = []
            for var in ['objetivo', 'tasa_cumplimiento', 'pct_subcontratacion']:
                ss = self.calculate_sum_of_squares(var)
                eta_data.append({
                    'Variable': var,
                    'η² Beta': ss['eta2_beta'],
                    'η² Capacity': ss['eta2_capacity'],
                    'η² Interaction': ss['eta2_interaction']
                })
            
            df_eta = pd.DataFrame(eta_data)
            df_eta.to_excel(writer, sheet_name='Eta_Squared', index=False)
        
        print(f"\n✓ Results exported to Excel: {output_file}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function to run the complete factorial experiment.
    """
    print("\n" + "="*80)
    print(" " * 20 + "3² FACTORIAL EXPERIMENT")
    print(" " * 10 + "Interaction: Tax Incentives × Transport Capacity")
    print("="*80 + "\n")
    
    # Check if data file exists
    if not os.path.exists('instance_distance_exp.json'):
        print("❌ ERROR: File 'instance_distance_exp.json' not found")
        print("   Please ensure the data file is in the current directory.")
        return
    
    # Create experiment
    experiment = FactorialExperiment('instance_distance_exp.json')
    
    # Define factorial levels
    experiment.define_factorial_levels()
    
    # Generate scenarios
    experiment.generate_scenarios()
    
    # Run experiment
    print("\n⏳ Starting experiment execution...")
    print("   (This may take several minutes)")
    
    start_time = time.time()
    experiment.run_experiment(verbose=True)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Experiment completed in {elapsed/60:.2f} minutes")
    
    # Create results DataFrame
    df_results = experiment.create_results_dataframe()
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df_results[['beta_level', 'capacity_level', 'objetivo', 
                         'tasa_cumplimiento', 'pct_subcontratacion']].to_string(index=False))
    
    # Sum of squares and eta squared analysis
    print("\n" + "="*80)
    print("EFFECTS ANALYSIS - ETA SQUARED (η²)")
    print("="*80)
    
    for var_name, var_col in [('Net Benefit', 'objetivo'),
                               ('Compliance Rate', 'tasa_cumplimiento'),
                               ('Outsourcing %', 'pct_subcontratacion')]:
        print(f"\n{var_name}:")
        ss = experiment.calculate_sum_of_squares(var_col)
        print(f"  η² (Tax Benefit):           {ss['eta2_beta']:.4f} ({ss['eta2_beta']*100:.2f}%)")
        print(f"  η² (Transport Capacity):    {ss['eta2_capacity']:.4f} ({ss['eta2_capacity']*100:.2f}%)")
        print(f"  η² (Interaction β×Cap):     {ss['eta2_interaction']:.4f} ({ss['eta2_interaction']*100:.2f}%)")
        
        if ss['eta2_interaction'] > 0.10:
            print(f"  → SIGNIFICANT INTERACTION DETECTED (η² > 10%)")
    
    # Generate interaction plots (English labels already)
    print("\n" + "="*80)
    print("GENERATING INTERACTION PLOTS")
    print("="*80)
    experiment.generate_interaction_plots('interaction_plots.png')
    
    # Calculate elasticity matrix
    experiment.calculate_elasticity_matrix()
    
    # Generate full report
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)
    experiment.generate_full_report('factorial_experiment_report.txt')
    
    # Export to Excel
    experiment.export_results_excel('factorial_experiment_results.xlsx')
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT CONCLUSIONS")
    print("="*80)
    
    # Check for interaction
    ss_obj = experiment.calculate_sum_of_squares('objetivo')
    
    if ss_obj['eta2_interaction'] > 0.15:
        print("\n✓ ALTERNATIVE HYPOTHESIS CONFIRMED:")
        print("  There is a significant interaction between tax benefit and")
        print("  transport capacity. The economic value of expanding the fleet")
        print("  increases when tax incentives are higher.")
        print("\n  → Tax incentives act as a CATALYST for investment")
        print("    in own logistics capacity.")
    else:
        print("\n✓ NULL HYPOTHESIS:")
        print("  No strong interaction between factors was detected.")
        print("  The effects are mainly additive.")
    
    print("\n" + "="*80)
    print("GENERATED FILES:")
    print("="*80)
    print("  1. interaction_plots.png                - Interaction plots (English)")
    print("  2. factorial_experiment_report.txt      - Full text report")
    print("  3. factorial_experiment_results.xlsx    - Results in Excel")
    print("\n✓ Experiment completed successfully")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
