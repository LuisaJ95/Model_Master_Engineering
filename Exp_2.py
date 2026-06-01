# %%
"""
EXPERIMENT 2: RESPONSE SURFACE METHODOLOGY - CENTRAL COMPOSITE DESIGN (CCD)
=============================================================================

Objective: Determine optimal values for bank capacity and fleet capacity
          to maximize donation flow without overwhelming the system.

Factors:
  X₁: Bank capacity multiplier (0.6 to 1.5) - applies to ALL banks
  X₂: Fleet capacity (4000 to 12000 kg)

Response Variables:
  Y₁: Fulfillment rate (Total delivered / Total available)
  Y₂: Total outsourcing cost

Design: Central Composite Design with α = √2 for rotatability
"""

import gurobipy as gp
from gurobipy import GRB
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import itertools
import time
from copy import deepcopy
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# OPTIMIZATION MODEL (Simplified version for experiments)
# ==============================================================================

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
    
    def load_data_from_dict(self, data):
        """Load data from dictionary."""
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
        """Build optimization model."""
        self.model = gp.Model("FoodDonation", env=self.env)
        self.model.setParam('OutputFlag', 0)
        
        # Create variables
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                self.y_deliv[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS)
                self.x_out[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS)
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r):
                self.y_pickup[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS)
            
            for t in range(l_r + 1, e_r + 1):
                self.y_deliv[r, t] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS)
        
        for r in self.R_direct + self.R_indirect:
            self.w[r] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS)
        
        self.model.update()
        
        # Set objective
        self._set_objective()
        
        # Add constraints
        self._add_constraints()
        
        self.model.update()
    
    def _set_objective(self):
        """Set objective function."""
        obj_terms = []
        
        # Tax benefits and costs for direct requirements
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                y_val = self.y_deliv[r, t]
                x_val = self.x_out[r, t]
                
                # Tax benefit on product value
                obj_terms.append(self.beta * c_prod * (y_val + x_val))
                
                # Tax benefit on transport
                obj_terms.append(self.beta * self.c_trans * dist * (y_val + self.alpha * x_val))
                
                # Transport costs
                obj_terms.append(-self.c_trans * dist * y_val)
                
                # Outsourcing costs
                obj_terms.append(-self.alpha * self.c_trans * dist * x_val)
        
        # Tax benefits and costs for indirect requirements
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                y_val = self.y_deliv[r, t]
                
                # Tax benefit on product value
                obj_terms.append(self.beta * c_prod * y_val)
                
                # Tax benefit on transport
                obj_terms.append(self.beta * self.c_trans * dist_indir * y_val)
                
                # Transport costs
                obj_terms.append(-self.c_trans * dist_indir * y_val)
        
        # Waste penalties
        for r in self.R_direct + self.R_indirect:
            obj_terms.append(-self.pi * self.w[r])
        
        self.model.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)
    
    def _add_constraints(self):
        """Add constraints."""
        # Flow conservation - Direct
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            q_r = req['quantity']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            self.model.addConstr(
                gp.quicksum(self.y_deliv[r, t] + self.x_out[r, t] for t in range(l_r, e_r + 1)) + 
                self.w[r] == q_r
            )
        
        # Flow conservation - Indirect
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            q_r = req['quantity']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            self.model.addConstr(
                gp.quicksum(self.y_deliv[r, t] for t in range(l_r + 1, e_r + 1)) + 
                self.w[r] == q_r
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
                    self.model.addConstr(gp.quicksum(deliveries) <= cap_j)
        
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
                self.model.addConstr(gp.quicksum(fleet_usage) <= cap_trans)
        
        # Synchronization - Indirect
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r):
                self.model.addConstr(self.y_pickup[r, t] == self.y_deliv[r, t + 1])
    
    def optimize(self):
        """Solve the model."""
        self.model.optimize()
        return self.model.Status
    
    def get_metrics(self):
        """Extract metrics from solution."""
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        
        if self.model.SolCount == 0:
            return None
        
        metrics = {
            'objective_value': self.model.ObjVal,
            'total_donated': 0,
            'total_wasted': 0,
            'total_outsourced': 0,
            'total_available': 0,
            'fulfillment_rate': 0,
            'outsourcing_cost': 0
        }
        
        # Calculate totals
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            metrics['total_available'] += req['quantity']
            l_r = req['release_date']
            e_r = req['expiration_date']
            dist = req['distance']
            
            for t in range(l_r, e_r + 1):
                y_val = self.y_deliv[r, t].X
                x_val = self.x_out[r, t].X
                metrics['total_donated'] += (y_val + x_val)
                metrics['total_outsourced'] += x_val
                metrics['outsourcing_cost'] += self.alpha * self.c_trans * dist * x_val
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            metrics['total_available'] += req['quantity']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                y_val = self.y_deliv[r, t].X
                metrics['total_donated'] += y_val
        
        for r in self.R_direct + self.R_indirect:
            metrics['total_wasted'] += self.w[r].X
        
        # Calculate fulfillment rate
        if metrics['total_available'] > 0:
            metrics['fulfillment_rate'] = (metrics['total_donated'] / metrics['total_available']) * 100
        
        return metrics


# ==============================================================================
# EXPERIMENTAL DESIGN AND SIMULATION
# ==============================================================================

class ExperimentCCD:
    """Central Composite Design experiment manager."""
    
    def __init__(self, factors, num_replicates=3):
        """
        Initialize CCD experiment.
        
        Args:
            factors: Dictionary with factor names and (min, max) ranges
            num_replicates: Number of replicate runs at each design point
        """
        self.factors = factors
        self.factor_names = list(factors.keys())
        self.num_factors = len(factors)
        self.num_replicates = num_replicates
        
        # Design parameters
        self.alpha = np.sqrt(self.num_factors)  # For rotatability
        
        # Design points
        self.design_matrix = None
        self.design_matrix_coded = None
        
        # Results storage
        self.results = []
    
    def generate_design(self):
        """Generate Central Composite Design."""
        print("\n" + "="*80)
        print("GENERATING CENTRAL COMPOSITE DESIGN")
        print("="*80)
        
        # Factorial points (2^k corners of cube)
        factorial_coded = list(itertools.product([-1, 1], repeat=self.num_factors))
        factorial_points = np.array(factorial_coded)
        
        # Axial points (star points)
        axial_points = []
        for i in range(self.num_factors):
            point_plus = np.zeros(self.num_factors)
            point_plus[i] = self.alpha
            axial_points.append(point_plus)
            
            point_minus = np.zeros(self.num_factors)
            point_minus[i] = -self.alpha
            axial_points.append(point_minus)
        axial_points = np.array(axial_points)
        
        # Center points
        num_center = max(5, self.num_factors + 1)  # At least 5 center points
        center_points = np.zeros((num_center, self.num_factors))
        
        # Combine all points
        self.design_matrix_coded = np.vstack([
            factorial_points,
            axial_points,
            center_points
        ])
        
        # Convert coded values to natural units
        self.design_matrix = self._decode_design(self.design_matrix_coded)
        
        print(f"\n✓ Design generated:")
        print(f"  - Factorial points: {len(factorial_points)}")
        print(f"  - Axial points: {len(axial_points)}")
        print(f"  - Center points: {len(center_points)}")
        print(f"  - Total design points: {len(self.design_matrix)}")
        print(f"  - Alpha (for rotatability): {self.alpha:.4f}")
        print(f"  - Replicates per point: {self.num_replicates}")
        print(f"  - Total experimental runs: {len(self.design_matrix) * self.num_replicates}")
        
        return self.design_matrix
    
    def _decode_design(self, coded_matrix):
        """Convert coded design matrix to natural units."""
        natural_matrix = np.zeros_like(coded_matrix)
        
        for i, factor_name in enumerate(self.factor_names):
            low, high = self.factors[factor_name]
            center = (high + low) / 2
            radius = (high - low) / 2
            
            natural_matrix[:, i] = center + radius * coded_matrix[:, i]
        
        return natural_matrix
    
    def _encode_design(self, natural_matrix):
        """Convert natural units to coded design matrix."""
        coded_matrix = np.zeros_like(natural_matrix)
        
        for i, factor_name in enumerate(self.factor_names):
            low, high = self.factors[factor_name]
            center = (high + low) / 2
            radius = (high - low) / 2
            
            coded_matrix[:, i] = (natural_matrix[:, i] - center) / radius
        
        return coded_matrix
    
    def save_design(self, filename='experimental_design_ccd.csv'):
        """Save design matrix to CSV."""
        if self.design_matrix is None:
            print("⚠️ No design matrix to save. Generate design first.")
            return
        
        # Create DataFrame
        df_natural = pd.DataFrame(self.design_matrix, columns=self.factor_names)
        df_coded = pd.DataFrame(self.design_matrix_coded, 
                               columns=[f"{name}_coded" for name in self.factor_names])
        
        # Add design point type
        num_factorial = 2**self.num_factors
        num_axial = 2 * self.num_factors
        num_center = len(self.design_matrix) - num_factorial - num_axial
        
        point_types = (['Factorial'] * num_factorial + 
                      ['Axial'] * num_axial + 
                      ['Center'] * num_center)
        
        df = pd.concat([
            pd.DataFrame({'run': range(1, len(self.design_matrix) + 1),
                         'point_type': point_types}),
            df_natural,
            df_coded
        ], axis=1)
        
        df.to_csv(filename, index=False)
        print(f"\n✓ Design matrix saved to: {filename}")
        
        return df
    
    def visualize_design(self, filename='ccd_design_plot.png'):
        """Visualize the design in 2D."""
        if self.design_matrix_coded is None:
            print("⚠️ No design to visualize. Generate design first.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Identify point types
        num_factorial = 2**self.num_factors
        num_axial = 2 * self.num_factors
        
        factorial_idx = slice(0, num_factorial)
        axial_idx = slice(num_factorial, num_factorial + num_axial)
        center_idx = slice(num_factorial + num_axial, None)
        
        # Plot points
        ax.scatter(self.design_matrix_coded[factorial_idx, 0],
                  self.design_matrix_coded[factorial_idx, 1],
                  s=150, c='red', marker='s', label='Factorial', edgecolors='black', linewidth=2)
        
        ax.scatter(self.design_matrix_coded[axial_idx, 0],
                  self.design_matrix_coded[axial_idx, 1],
                  s=150, c='blue', marker='^', label='Axial', edgecolors='black', linewidth=2)
        
        ax.scatter(self.design_matrix_coded[center_idx, 0],
                  self.design_matrix_coded[center_idx, 1],
                  s=150, c='green', marker='o', label='Center', edgecolors='black', linewidth=2)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        # Labels and legend
        ax.set_xlabel(f'{self.factor_names[0]} (coded)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{self.factor_names[1]} (coded)', fontsize=12, fontweight='bold')
        ax.set_title('Central Composite Design (CCD) - Coded Units', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Design visualization saved to: {filename}")
        plt.close()


class OptimizationSimulator:
    """Run optimization simulations for experimental design."""
    
    def __init__(self, data_file, num_replicates=3):
        """
        Initialize simulator.
        
        Args:
            data_file: Path to JSON data file
            num_replicates: Number of replicate runs per design point
        """
        self.data_file = data_file
        self.num_replicates = num_replicates
        
        # Load base instance
        with open(data_file, 'r', encoding='utf-8') as f:
            self.base_instance = json.load(f)
        
        print(f"\n✓ Base instance loaded from: {data_file}")
        print(f"  - Total requirements: {len(self.base_instance['requirements'])}")
        print(f"  - Food banks: {len(self.base_instance['sets']['food_banks'])}")
        print(f"  - Base fleet capacity: {self.base_instance['parameters']['transport_capacity_per_period']} kg")
        
        # Show current bank capacities
        print(f"  - Bank capacities:")
        total_bank_cap = 0
        for bank, cap in self.base_instance['parameters']['food_bank_capacity'].items():
            print(f"      {bank}: {cap:,.0f} kg")
            total_bank_cap += cap
        print(f"      TOTAL: {total_bank_cap:,.0f} kg")
    
    def modify_instance(self, factor_values):
        """
        Modify instance with experimental factor values.
        
        Args:
            factor_values: Dictionary with factor names and values
                          e.g., {'X1_bank_capacity_multiplier': 1.2, 'X2_fleet_capacity': 10000}
        
        Returns:
            Modified instance dictionary
        """
        modified = deepcopy(self.base_instance)
        
        # Extract factor values
        bank_multiplier = factor_values.get('X1_bank_capacity_multiplier', 1.0)
        fleet_capacity = factor_values.get('X2_fleet_capacity', 8000)
        
        # Apply multiplier to ALL food bank capacities
        for bank_id in modified['parameters']['food_bank_capacity'].keys():
            original_capacity = modified['parameters']['food_bank_capacity'][bank_id]
            modified['parameters']['food_bank_capacity'][bank_id] = original_capacity * bank_multiplier
        
        # Set fleet capacity
        modified['parameters']['transport_capacity_per_period'] = fleet_capacity
        
        return modified
    
    def run_single_simulation(self, factor_values, run_id, env):
        """
        Run a single optimization simulation.
        
        Args:
            factor_values: Dictionary with factor names and values
            run_id: Run identifier
            env: Gurobi environment
        
        Returns:
            Dictionary with results
        """
        # Modify instance
        instance = self.modify_instance(factor_values)
        
        # Create and solve model
        optimizer = FoodDonationOptimizer(env)
        optimizer.load_data_from_dict(instance)
        optimizer.build_model()
        
        start_time = time.time()
        status = optimizer.optimize()
        solve_time = time.time() - start_time
        
        # Extract metrics
        if status == GRB.OPTIMAL:
            metrics = optimizer.get_metrics()
            
            result = {
                'run_id': run_id,
                'status': 'Optimal',
                'Y1_fulfillment_rate': metrics['fulfillment_rate'],
                'Y2_outsourcing_cost': metrics['outsourcing_cost'],
                'objective_value': metrics['objective_value'],
                'total_donated': metrics['total_donated'],
                'total_wasted': metrics['total_wasted'],
                'total_outsourced': metrics['total_outsourced'],
                'solve_time': solve_time
            }
            # Add factor values to result (this ensures correct column names)
            result.update(factor_values)
        else:
            result = {
                'run_id': run_id,
                'status': f'Status_{status}',
                'Y1_fulfillment_rate': np.nan,
                'Y2_outsourcing_cost': np.nan,
                'objective_value': np.nan,
                'total_donated': np.nan,
                'total_wasted': np.nan,
                'total_outsourced': np.nan,
                'solve_time': solve_time
            }
            # Add factor values to result
            result.update(factor_values)
        
        # Clean up
        optimizer.model.dispose()
        
        return result
    
    def run_experiment(self, design_matrix, factor_names):
        """
        Run full experimental design.
        
        Args:
            design_matrix: Matrix of design points (natural units)
            factor_names: List of factor names
        
        Returns:
            DataFrame with all results
        """
        print("\n" + "="*80)
        print("RUNNING EXPERIMENTAL SIMULATIONS")
        print("="*80)
        
        results = []
        total_runs = len(design_matrix) * self.num_replicates
        run_counter = 0
        
        # Create Gurobi environment
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('TimeLimit', 300)  # 5 minutes max per run
            env.setParam('MIPFocus', 1)
            env.setParam('Threads', 2)
            env.start()
            
            # Run each design point with replicates
            for point_idx, design_point in enumerate(design_matrix):
                # Create factor values dictionary
                factor_values = {factor_names[i]: design_point[i] for i in range(len(factor_names))}
                
                print(f"\n{'='*80}")
                print(f"Design Point {point_idx + 1}/{len(design_matrix)}")
                print(f"{'='*80}")
                for fname, fval in factor_values.items():
                    print(f"  {fname}: {fval:.4f}")
                
                # Run replicates
                for rep in range(self.num_replicates):
                    run_counter += 1
                    run_id = f"P{point_idx+1}_R{rep+1}"
                    
                    print(f"\n  Run {run_counter}/{total_runs} (Replicate {rep+1}/{self.num_replicates})...", end=' ')
                    
                    result = self.run_single_simulation(
                        factor_values, run_id, env
                    )
                    
                    results.append(result)
                    
                    if result['status'] == 'Optimal':
                        print(f"✓ Y₁={result['Y1_fulfillment_rate']:.2f}% Y₂=${result['Y2_outsourcing_cost']:.2f}")
                    else:
                        print(f"✗ {result['status']}")
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("EXPERIMENTAL RUNS COMPLETED")
        print("="*80)
        print(f"  Total runs: {len(df_results)}")
        print(f"  Successful: {(df_results['status'] == 'Optimal').sum()}")
        print(f"  Failed: {(df_results['status'] != 'Optimal').sum()}")
        
        return df_results


# ==============================================================================
# RESPONSE SURFACE ANALYSIS
# ==============================================================================

class ResponseSurfaceAnalysis:
    """Analyze experimental results and fit response surface."""
    
    def __init__(self, results_df, factors):
        """
        Initialize analysis.
        
        Args:
            results_df: DataFrame with experimental results
            factors: Dictionary with factor names and ranges
        """
        self.results_df = results_df
        self.factors = factors
        self.factor_names = list(factors.keys())
        
        # Models
        self.model_Y1 = None
        self.model_Y2 = None
        
        # Coefficients
        self.coef_Y1 = None
        self.coef_Y2 = None
        
        # Statistics
        self.stats_Y1 = {}
        self.stats_Y2 = {}
    
    def prepare_data(self):
        """Prepare data for regression analysis."""
        # Filter only successful runs
        df = self.results_df[self.results_df['status'] == 'Optimal'].copy()
        
        if len(df) == 0:
            raise ValueError("No successful optimization runs to analyze!")
        
        # Extract factors (natural units)
        X_natural = df[self.factor_names].values
        
        # Convert to coded units
        X_coded = self._encode_to_coded(X_natural)
        
        # Create design matrix for quadratic model
        # [1, X1, X2, X1², X2², X1*X2]
        X1 = X_coded[:, 0]
        X2 = X_coded[:, 1]
        
        X_design = np.column_stack([
            np.ones(len(X1)),  # Intercept
            X1,                # Linear X1
            X2,                # Linear X2
            X1**2,             # Quadratic X1
            X2**2,             # Quadratic X2
            X1*X2              # Interaction
        ])
        
        # Response variables
        Y1 = df['Y1_fulfillment_rate'].values
        Y2 = df['Y2_outsourcing_cost'].values
        
        return X_design, X_coded, X_natural, Y1, Y2
    
    def _encode_to_coded(self, X_natural):
        """Convert natural units to coded units."""
        X_coded = np.zeros_like(X_natural)
        
        for i, factor_name in enumerate(self.factor_names):
            low, high = self.factors[factor_name]
            center = (high + low) / 2
            radius = (high - low) / 2
            
            X_coded[:, i] = (X_natural[:, i] - center) / radius
        
        return X_coded
    
    def _decode_to_natural(self, X_coded):
        """Convert coded units to natural units."""
        X_natural = np.zeros_like(X_coded)
        
        for i, factor_name in enumerate(self.factor_names):
            low, high = self.factors[factor_name]
            center = (high + low) / 2
            radius = (high - low) / 2
            
            X_natural[:, i] = center + radius * X_coded[:, i]
        
        return X_natural
    
    def fit_models(self):
        """Fit quadratic response surface models."""
        print("\n" + "="*80)
        print("FITTING QUADRATIC RESPONSE SURFACE MODELS")
        print("="*80)
        
        X_design, X_coded, X_natural, Y1, Y2 = self.prepare_data()
        
        # Fit model for Y1 (Fulfillment Rate)
        self.model_Y1 = LinearRegression()
        self.model_Y1.fit(X_design, Y1)
        Y1_pred = self.model_Y1.predict(X_design)
        
        self.coef_Y1 = {
            'β0': self.model_Y1.intercept_,
            'β1': self.model_Y1.coef_[1],
            'β2': self.model_Y1.coef_[2],
            'β11': self.model_Y1.coef_[3],
            'β22': self.model_Y1.coef_[4],
            'β12': self.model_Y1.coef_[5]
        }
        
        self.stats_Y1 = {
            'R2': r2_score(Y1, Y1_pred),
            'R2_adj': 1 - (1 - r2_score(Y1, Y1_pred)) * (len(Y1) - 1) / (len(Y1) - 6),
            'RMSE': np.sqrt(mean_squared_error(Y1, Y1_pred))
        }
        
        # Fit model for Y2 (Outsourcing Cost)
        self.model_Y2 = LinearRegression()
        self.model_Y2.fit(X_design, Y2)
        Y2_pred = self.model_Y2.predict(X_design)
        
        self.coef_Y2 = {
            'β0': self.model_Y2.intercept_,
            'β1': self.model_Y2.coef_[1],
            'β2': self.model_Y2.coef_[2],
            'β11': self.model_Y2.coef_[3],
            'β22': self.model_Y2.coef_[4],
            'β12': self.model_Y2.coef_[5]
        }
        
        self.stats_Y2 = {
            'R2': r2_score(Y2, Y2_pred),
            'R2_adj': 1 - (1 - r2_score(Y2, Y2_pred)) * (len(Y2) - 1) / (len(Y2) - 6),
            'RMSE': np.sqrt(mean_squared_error(Y2, Y2_pred))
        }
        
        # Display results
        print("\n" + "─"*80)
        print("MODEL FOR Y₁ (FULFILLMENT RATE)")
        print("─"*80)
        print(f"\nRegression Equation (coded units):")
        print(f"  Y₁ = {self.coef_Y1['β0']:.4f}")
        print(f"       {self.coef_Y1['β1']:+.4f}·X₁")
        print(f"       {self.coef_Y1['β2']:+.4f}·X₂")
        print(f"       {self.coef_Y1['β11']:+.4f}·X₁²")
        print(f"       {self.coef_Y1['β22']:+.4f}·X₂²")
        print(f"       {self.coef_Y1['β12']:+.4f}·X₁·X₂")
        
        print(f"\nModel Statistics:")
        print(f"  R²:          {self.stats_Y1['R2']:.4f}")
        print(f"  R² adjusted: {self.stats_Y1['R2_adj']:.4f}")
        print(f"  RMSE:        {self.stats_Y1['RMSE']:.4f}")
        
        print("\n" + "─"*80)
        print("MODEL FOR Y₂ (OUTSOURCING COST)")
        print("─"*80)
        print(f"\nRegression Equation (coded units):")
        print(f"  Y₂ = {self.coef_Y2['β0']:.4f}")
        print(f"       {self.coef_Y2['β1']:+.4f}·X₁")
        print(f"       {self.coef_Y2['β2']:+.4f}·X₂")
        print(f"       {self.coef_Y2['β11']:+.4f}·X₁²")
        print(f"       {self.coef_Y2['β22']:+.4f}·X₂²")
        print(f"       {self.coef_Y2['β12']:+.4f}·X₁·X₂")
        
        print(f"\nModel Statistics:")
        print(f"  R²:          {self.stats_Y2['R2']:.4f}")
        print(f"  R² adjusted: {self.stats_Y2['R2_adj']:.4f}")
        print(f"  RMSE:        {self.stats_Y2['RMSE']:.4f}")
    
    def find_stationary_point(self, response='Y1'):
        """
        Find stationary point of response surface.
        
        Args:
            response: 'Y1' or 'Y2'
        
        Returns:
            Dictionary with stationary point analysis
        """
        print("\n" + "="*80)
        print(f"STATIONARY POINT ANALYSIS FOR {response}")
        print("="*80)
        
        # Select coefficients
        if response == 'Y1':
            coef = self.coef_Y1
        else:
            coef = self.coef_Y2
        
        # Compute stationary point (coded units)
        # Solve: ∂Y/∂X₁ = 0 and ∂Y/∂X₂ = 0
        
        # β₁ + 2β₁₁·X₁ + β₁₂·X₂ = 0
        # β₂ + 2β₂₂·X₂ + β₁₂·X₁ = 0
        
        # Matrix form: [2β₁₁  β₁₂ ] [X₁]   [-β₁]
        #              [β₁₂  2β₂₂] [X₂] = [-β₂]
        
        A = np.array([
            [2 * coef['β11'], coef['β12']],
            [coef['β12'], 2 * coef['β22']]
        ])
        
        b = np.array([-coef['β1'], -coef['β2']])
        
        try:
            X_stat_coded = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("⚠️ Singular matrix - no unique stationary point")
            return None
        
        # Convert to natural units
        X_stat_natural = self._decode_to_natural(X_stat_coded.reshape(1, -1))[0]
        
        # Predict response at stationary point
        X_design_stat = np.array([
            1,
            X_stat_coded[0],
            X_stat_coded[1],
            X_stat_coded[0]**2,
            X_stat_coded[1]**2,
            X_stat_coded[0] * X_stat_coded[1]
        ]).reshape(1, -1)
        
        if response == 'Y1':
            Y_stat = self.model_Y1.predict(X_design_stat)[0]
        else:
            Y_stat = self.model_Y2.predict(X_design_stat)[0]
        
        # Classify stationary point using eigenvalues of Hessian
        # Hessian = [2β₁₁  β₁₂ ]
        #           [β₁₂  2β₂₂]
        
        eigenvalues = np.linalg.eigvalsh(A)
        
        if all(eigenvalues > 0):
            point_type = "Minimum"
        elif all(eigenvalues < 0):
            point_type = "Maximum"
        else:
            point_type = "Saddle Point"
        
        # Display results
        print(f"\nStationary Point (coded units):")
        print(f"  X₁ = {X_stat_coded[0]:+.4f}")
        print(f"  X₂ = {X_stat_coded[1]:+.4f}")
        
        print(f"\nStationary Point (natural units):")
        print(f"  {self.factor_names[0]} = {X_stat_natural[0]:.4f}")
        print(f"  {self.factor_names[1]} = {X_stat_natural[1]:.4f}")
        
        print(f"\nPredicted Response:")
        print(f"  {response} = {Y_stat:.4f}")
        
        print(f"\nPoint Classification: {point_type}")
        print(f"  Eigenvalues: {eigenvalues}")
        
        result = {
            'X_coded': X_stat_coded,
            'X_natural': X_stat_natural,
            'Y_predicted': Y_stat,
            'point_type': point_type,
            'eigenvalues': eigenvalues
        }
        
        return result
    
    def plot_contour(self, response='Y1', filename=None, num_points=50):
        """
        Plot contour plot of response surface.
        
        Args:
            response: 'Y1' or 'Y2'
            filename: Output filename
            num_points: Resolution of grid
        """
        if filename is None:
            filename = f'contour_{response}.png'
        
        # Create grid
        low1, high1 = self.factors[self.factor_names[0]]
        low2, high2 = self.factors[self.factor_names[1]]
        
        x1_grid = np.linspace(low1, high1, num_points)
        x2_grid = np.linspace(low2, high2, num_points)
        X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
        
        # Convert to coded units
        X_natural_grid = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])
        X_coded_grid = self._encode_to_coded(X_natural_grid)
        
        # Create design matrix
        X_design_grid = np.column_stack([
            np.ones(len(X_coded_grid)),
            X_coded_grid[:, 0],
            X_coded_grid[:, 1],
            X_coded_grid[:, 0]**2,
            X_coded_grid[:, 1]**2,
            X_coded_grid[:, 0] * X_coded_grid[:, 1]
        ])
        
        # Predict response
        if response == 'Y1':
            Y_pred = self.model_Y1.predict(X_design_grid)
            title = 'Response Surface: Y₁ (Fulfillment Rate %)'
            cbar_label = 'Fulfillment Rate (%)'
        else:
            Y_pred = self.model_Y2.predict(X_design_grid)
            title = 'Response Surface: Y₂ (Outsourcing Cost $)'
            cbar_label = 'Outsourcing Cost ($)'
        
        Y_mesh = Y_pred.reshape(X1_mesh.shape)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 9))
        
        contour = ax.contourf(X1_mesh, X2_mesh, Y_mesh, levels=20, cmap='viridis')
        contour_lines = ax.contour(X1_mesh, X2_mesh, Y_mesh, levels=10, colors='white', 
                                   linewidths=0.5, alpha=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(cbar_label, fontsize=11, fontweight='bold')
        
        # Plot experimental points
        df_optimal = self.results_df[self.results_df['status'] == 'Optimal']
        ax.scatter(df_optimal[self.factor_names[0]], 
                  df_optimal[self.factor_names[1]],
                  c='red', s=50, marker='o', edgecolors='white', linewidth=1.5,
                  label='Experimental Points', zorder=5)
        
        # Labels
        ax.set_xlabel(self.factor_names[0], fontsize=12, fontweight='bold')
        ax.set_ylabel(self.factor_names[1], fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Contour plot saved to: {filename}")
        plt.close()
    
    def plot_surface_3d(self, response='Y1', filename=None, num_points=50):
        """
        Plot 3D surface of response.
        
        Args:
            response: 'Y1' or 'Y2'
            filename: Output filename
            num_points: Resolution of grid
        """
        if filename is None:
            filename = f'surface_3d_{response}.png'
        
        # Create grid
        low1, high1 = self.factors[self.factor_names[0]]
        low2, high2 = self.factors[self.factor_names[1]]
        
        x1_grid = np.linspace(low1, high1, num_points)
        x2_grid = np.linspace(low2, high2, num_points)
        X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
        
        # Convert to coded units and predict
        X_natural_grid = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])
        X_coded_grid = self._encode_to_coded(X_natural_grid)
        
        X_design_grid = np.column_stack([
            np.ones(len(X_coded_grid)),
            X_coded_grid[:, 0],
            X_coded_grid[:, 1],
            X_coded_grid[:, 0]**2,
            X_coded_grid[:, 1]**2,
            X_coded_grid[:, 0] * X_coded_grid[:, 1]
        ])
        
        if response == 'Y1':
            Y_pred = self.model_Y1.predict(X_design_grid)
            title = '3D Response Surface: Y₁ (Fulfillment Rate)'
            zlabel = 'Fulfillment Rate (%)'
        else:
            Y_pred = self.model_Y2.predict(X_design_grid)
            title = '3D Response Surface: Y₂ (Outsourcing Cost)'
            zlabel = 'Outsourcing Cost ($)'
        
        Y_mesh = Y_pred.reshape(X1_mesh.shape)
        
        # Plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X1_mesh, X2_mesh, Y_mesh, cmap='viridis',
                              alpha=0.8, edgecolor='none', antialiased=True)
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label(zlabel, fontsize=10, fontweight='bold')
        
        # Plot experimental points
        df_optimal = self.results_df[self.results_df['status'] == 'Optimal']
        if response == 'Y1':
            Y_exp = df_optimal['Y1_fulfillment_rate'].values
        else:
            Y_exp = df_optimal['Y2_outsourcing_cost'].values
        
        ax.scatter(df_optimal[self.factor_names[0]],
                  df_optimal[self.factor_names[1]],
                  Y_exp,
                  c='red', s=50, marker='o', edgecolors='black', linewidth=1.5,
                  label='Experimental Points', zorder=10)
        
        # Labels
        ax.set_xlabel(self.factor_names[0], fontsize=11, fontweight='bold', labelpad=10)
        ax.set_ylabel(self.factor_names[1], fontsize=11, fontweight='bold', labelpad=10)
        ax.set_zlabel(zlabel, fontsize=11, fontweight='bold', labelpad=10)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.legend(fontsize=9)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ 3D surface plot saved to: {filename}")
        plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    
    print("="*80)
    print("EXPERIMENT 2: RESPONSE SURFACE METHODOLOGY")
    print("Central Composite Design (CCD)")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # STEP 1: Define experimental factors
    # -------------------------------------------------------------------------
    
    factors = {
        'X1_bank_capacity_multiplier': (0.6, 1.5),  # 60% to 150% of base capacities
        'X2_fleet_capacity': (4000, 12000)  # Fleet capacity range in kg
    }
    
    num_replicates = 3  # Number of replicates per design point
    
    print("\n📋 EXPERIMENTAL SETUP:")
    print(f"\n  Factor X₁: {factors['X1_bank_capacity_multiplier']}")
    print(f"    Description: Multiplier for ALL bank capacities")
    print(f"    Range: {factors['X1_bank_capacity_multiplier'][0]} to {factors['X1_bank_capacity_multiplier'][1]}")
    print(f"    (1.0 = 100% of base capacities)")
    
    print(f"\n  Factor X₂: {factors['X2_fleet_capacity']}")
    print(f"    Description: Company-owned fleet capacity")
    print(f"    Range: {factors['X2_fleet_capacity'][0]} to {factors['X2_fleet_capacity'][1]} kg")
    print(f"    (Base: 8000 kg)")
    
    print(f"\n  Response Y₁: Fulfillment Rate (%)")
    print(f"    Total delivered / Total available")
    
    print(f"\n  Response Y₂: Total Outsourcing Cost ($)")
    
    print(f"\n  Replicates per design point: {num_replicates}")
    
    # -------------------------------------------------------------------------
    # STEP 2: Generate Central Composite Design
    # -------------------------------------------------------------------------
    
    experiment = ExperimentCCD(factors, num_replicates=num_replicates)
    design_matrix = experiment.generate_design()
    
    # Save and visualize design
    df_design = experiment.save_design('experimental_design_ccd.csv')
    experiment.visualize_design('ccd_design_plot.png')
    
    print("\n" + df_design.to_string())
    
    # -------------------------------------------------------------------------
    # STEP 3: Run simulations
    # -------------------------------------------------------------------------
    
    data_file = 'instance_distance_exp.json'
    
    if not os.path.exists(data_file):
        print(f"\n❌ ERROR: Data file '{data_file}' not found!")
        print("   Please ensure the instance file is in the current directory.")
        return
    
    simulator = OptimizationSimulator(data_file, num_replicates=num_replicates)
    
    print("\n⏳ Starting experimental runs...")
    print("   This may take several hours depending on instance size.")
    print("   Progress will be displayed for each run.")
    
    start_time = time.time()
    df_results = simulator.run_experiment(design_matrix, experiment.factor_names)
    total_time = time.time() - start_time
    
    print(f"\n⏱️ Total experimental time: {total_time/60:.2f} minutes")
    print(f"   Average time per run: {total_time/len(df_results):.2f} seconds")
    
    # Save results
    df_results.to_csv('experimental_results_ccd.csv', index=False)
    print(f"\n✓ Results saved to: experimental_results_ccd.csv")
    
    # -------------------------------------------------------------------------
    # STEP 4: Analyze results and fit response surfaces
    # -------------------------------------------------------------------------
    
    analysis = ResponseSurfaceAnalysis(df_results, factors)
    
    # Fit models
    analysis.fit_models()
    
    # Find stationary points
    stat_Y1 = analysis.find_stationary_point(response='Y1')
    stat_Y2 = analysis.find_stationary_point(response='Y2')
    
    # -------------------------------------------------------------------------
    # STEP 5: Generate visualizations
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Contour plots
    analysis.plot_contour(response='Y1', filename='contour_Y1_fulfillment.png')
    analysis.plot_contour(response='Y2', filename='contour_Y2_outsourcing.png')
    
    # 3D surface plots
    analysis.plot_surface_3d(response='Y1', filename='surface_3d_Y1_fulfillment.png')
    analysis.plot_surface_3d(response='Y2', filename='surface_3d_Y2_outsourcing.png')
    
    # -------------------------------------------------------------------------
    # STEP 6: Final summary
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    
    print("\n📁 Generated Files:")
    print("  1. experimental_design_ccd.csv - Design matrix")
    print("  2. experimental_results_ccd.csv - All simulation results")
    print("  3. ccd_design_plot.png - Design visualization")
    print("  4. contour_Y1_fulfillment.png - Y₁ contour plot")
    print("  5. contour_Y2_outsourcing.png - Y₂ contour plot")
    print("  6. surface_3d_Y1_fulfillment.png - Y₁ 3D surface")
    print("  7. surface_3d_Y2_outsourcing.png - Y₂ 3D surface")
    
    if stat_Y1:
        print("\n🎯 OPTIMAL CONFIGURATION (for maximizing Y₁):")
        print(f"  Bank Capacity Multiplier: {stat_Y1['X_natural'][0]:.4f}")
        print(f"  Fleet Capacity: {stat_Y1['X_natural'][1]:.2f} kg")
        print(f"  Predicted Fulfillment Rate: {stat_Y1['Y_predicted']:.2f}%")
        print(f"  Point Type: {stat_Y1['point_type']}")
    
    print("\n✓ Analysis complete! Review the generated files for detailed results.")


if __name__ == "__main__":
    main()



