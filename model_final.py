# %% [markdown]
# # 🚀 Food Donation Optimization

# %% [markdown]
# ## 1️⃣ Import

# %%
import gurobipy as gp
from gurobipy import GRB
import json
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML
import sys
from copy import deepcopy
import json
import time


print("✓ Libraries imported")

# %% [markdown]
# ## 2️⃣ Input DATA (JSON)

# %% [markdown]
# Test data instances can be loaded from drib with the following cell

# %%

file_path = "instance_distance_exp.json"

with open(file_path, "r") as f:
    data = json.load(f)

print(f"✓ Data loaded: {len(data['requirements'])} requirements")

# %% [markdown]
# ## 3️⃣ Build and Solve Model
# 
# This cell creates the optimization model and solves it:

# %%
"""
Food Donation Optimization Model - OPTIMIZED VERSION
Maximizes net benefit from food donations considering:
- Direct and indirect requirements
- Company-owned fleet and outsourcing options
- Tax benefits and operational costs
- Capacity constraints and time windows
"""


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
        self.requirements_dict = {}  # For O(1) lookup
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
        """
        Load instance data from JSON file.
        
        Args:
            json_file: Path to JSON file with instance data
        """
        print(f"\nLoading data from {json_file}...")
        start_time = time.time()
        
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
        
        # Transport capacity (same for all periods or can be customized)
        trans_cap = params['transport_capacity_per_period']
        self.transport_capacity = {t: trans_cap for t in self.T}
        
        elapsed = time.time() - start_time
        
        print(f"✓ Data loaded successfully in {elapsed:.2f}s")
        print(f"  - Total requirements: {len(self.requirements)}")
        print(f"  - Direct requirements: {len(self.R_direct)}")
        print(f"  - Indirect requirements: {len(self.R_indirect)}")
        print(f"  - Time periods: {len(self.T)}")
        print(f"  - Food banks: {len(self.B)}")
        print(f"  - Products: {len(self.P)}")
    
    def _get_requirement_by_id(self, req_id):
        """Helper function to get requirement data by ID (O(1) lookup)."""
        return self.requirements_dict.get(req_id)
    
    def build_model(self):
        """Build the optimization model."""
        print("\n" + "="*70)
        print("BUILDING OPTIMIZATION MODEL")
        print("="*70)
        
        # Create model
        self.model = gp.Model("FoodDonationOptimization", env=self.env)
        
        # Create variables
        self._create_variables()
        
        # Set objective
        self._set_objective()
        
        # Add constraints
        self._add_constraints()
        
        # Update model to process variables and constraints
        self.model.update()
        
        print("\n✓ Model built successfully")
        print(f"  - Variables: {self.model.NumVars}")
        print(f"  - Linear constraints: {self.model.NumConstrs}")
        print(f"  - Nonzeros: {self.model.NumNZs}")
        
        # Estimate expected variable count
        expected_vars = 0
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            time_window = req['expiration_date'] - req['release_date'] + 1
            expected_vars += 2 * time_window  # y_deliv + x_out
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            time_window = req['expiration_date'] - req['release_date'] + 1
            expected_vars += time_window  # y_pickup
            expected_vars += time_window  # y_deliv
        
        expected_vars += len(self.requirements)  # w variables
        
        print(f"  - Expected variables: ~{expected_vars}")
        
        if self.model.NumVars > 50000:
            print("  ⚠️ WARNING: Large model detected!")
            print("     This may take several minutes to solve")
    
    def _create_variables(self):
        """Create all decision variables."""
        print("\nCreating decision variables...")
        start_time = time.time()
        
        # Variables for DIRECT requirements
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                # Delivery by company-owned fleet
                self.y_deliv[r, t] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS,
                    name=f"y_deliv_D_{r}_{t}"
                )
                
                # Outsourcing
                self.x_out[r, t] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS,
                    name=f"x_out_{r}_{t}"
                )
        
        # Variables for INDIRECT requirements
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            # Pickup variables (from Client to CEDI)
            for t in range(l_r, e_r):  # Can't pickup on expiration date
                self.y_pickup[r, t] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS,
                    name=f"y_pickup_{r}_{t}"
                )
            
            # Delivery variables (from CEDI to Food Bank)
            for t in range(l_r + 1, e_r + 1):  # Can't deliver before l_r+1
                self.y_deliv[r, t] = self.model.addVar(
                    lb=0, vtype=GRB.CONTINUOUS,
                    name=f"y_deliv_I_{r}_{t}"
                )
        
        # Waste variables for ALL requirements
        for r in self.R_direct + self.R_indirect:
            self.w[r] = self.model.addVar(
                lb=0, vtype=GRB.CONTINUOUS,
                name=f"w_{r}"
            )
        
        elapsed = time.time() - start_time
        print(f"✓ Variables created in {elapsed:.2f}s")
    
    def _set_objective(self):
        """Set the objective function to maximize net benefit using efficient quicksum."""
        print("\nBuilding objective function...")
        start_time = time.time()
        
        # Build all objective terms as lists for efficient quicksum
        
        # Component 1: Tax benefit on product value
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
        
        # Component 2: Tax benefit on transportation costs
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
        
        # Component 3: Subtract actual transportation costs (company fleet)
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
        
        # Component 4: Subtract outsourcing costs
        outsource_cost_terms = []
        
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            cost_coef = self.alpha * self.c_trans * dist
            
            for t in range(l_r, e_r + 1):
                outsource_cost_terms.append(cost_coef * self.x_out[r, t])
        
        # Component 5: Waste penalties
        waste_terms = [self.pi * self.w[r] for r in self.R_direct + self.R_indirect]
        
        # Combine all components using efficient quicksum
        obj = (
            gp.quicksum(product_benefit_terms) +
            gp.quicksum(transport_benefit_terms) -
            gp.quicksum(transport_cost_terms) -
            gp.quicksum(outsource_cost_terms) -
            gp.quicksum(waste_terms)
        )
        
        # Set objective
        self.model.setObjective(obj, GRB.MAXIMIZE)
        
        elapsed = time.time() - start_time
        print(f"✓ Objective function built in {elapsed:.2f}s")
    
    def _add_constraints(self):
        """Add all constraints to the model."""
        print("\nAdding constraints...")
        start_time = time.time()
        
        # Constraint (2): Flow conservation for DIRECT requirements
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
        
        # Constraint (3): Flow conservation for INDIRECT requirements
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
        
        print(f"  ✓ Flow conservation: {len(self.requirements)} constraints")
        
        # Constraint (4): Food bank receiving capacity
        # Pre-build requirement lists per food bank for efficiency
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
        
        bank_cap_count = 0
        for j in self.B:
            cap_j = self.food_bank_capacity.get(j, 0)
            
            for t in self.T:
                deliveries = []
                
                # Direct requirements
                for r in direct_reqs_by_bank[j]:
                    req = self._get_requirement_by_id(r)
                    l_r = req['release_date']
                    e_r = req['expiration_date']
                    if l_r <= t <= e_r:
                        deliveries.append(self.y_deliv[r, t] + self.x_out[r, t])
                
                # Indirect requirements
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
                    bank_cap_count += 1
        
        print(f"  ✓ Food bank capacity: {bank_cap_count} constraints")
        
        # Constraint (5): Transportation capacity (company-owned fleet)
        trans_cap_count = 0
        for t in self.T:
            cap_trans = self.transport_capacity[t]
            
            fleet_usage = []
            
            # Direct requirements: delivery on day t
            for r in self.R_direct:
                req = self._get_requirement_by_id(r)
                l_r = req['release_date']
                e_r = req['expiration_date']
                if l_r <= t <= e_r:
                    fleet_usage.append(self.y_deliv[r, t])
            
            # Indirect requirements: pickup on day t
            for r in self.R_indirect:
                req = self._get_requirement_by_id(r)
                l_r = req['release_date']
                e_r = req['expiration_date']
                if l_r <= t < e_r:
                    fleet_usage.append(self.y_pickup[r, t])
            
            # Indirect requirements: delivery on day t
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
                trans_cap_count += 1
        
        print(f"  ✓ Transport capacity: {trans_cap_count} constraints")
        
        # Constraint (6): Synchronization for INDIRECT requirements
        sync_count = 0
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r):
                self.model.addConstr(
                    self.y_pickup[r, t] == self.y_deliv[r, t + 1],
                    name=f"sync_{r}_{t}"
                )
                sync_count += 1
        
        print(f"  ✓ Synchronization: {sync_count} constraints")
        
        elapsed = time.time() - start_time
        print(f"✓ All constraints added in {elapsed:.2f}s")
    
    def optimize(self):
        """Solve the optimization model."""
        print("\n" + "="*70)
        print("OPTIMIZING")
        print("="*70)
        
        start_time = time.time()
        self.model.optimize()
        elapsed = time.time() - start_time
        
        status = self.model.Status
        
        print("\n" + "="*70)
        if status == GRB.OPTIMAL:
            print(f"✓ OPTIMAL SOLUTION FOUND in {elapsed:.2f}s")
            print(f"  - Objective value: ${self.model.ObjVal:,.2f}")
            print(f"  - Simplex iterations: {self.model.IterCount}")
            print(f"  - Barrier iterations: {self.model.BarIterCount}")
            print(f"  - Nodes explored: {self.model.NodeCount}")
        elif status == GRB.INFEASIBLE:
            print("✗ MODEL IS INFEASIBLE")
            print("  Computing IIS...")
            self.model.computeIIS()
            self.model.write("food_donation_model.ilp")
            print("  IIS written to food_donation_model.ilp")
            print("\n  ⚠️ Please use the Explainer agent to analyze the ILP file")
        elif status == GRB.UNBOUNDED:
            print("✗ MODEL IS UNBOUNDED")
        elif status == GRB.TIME_LIMIT:
            print(f"⚠️ TIME LIMIT REACHED after {elapsed:.2f}s")
            if self.model.SolCount > 0:
                print(f"  - Best feasible solution found: ${self.model.ObjVal:,.2f}")
                print(f"  - Best bound: ${self.model.ObjBound:,.2f}")
                print(f"  - Gap: {self.model.MIPGap*100:.2f}%")
        else:
            print(f"✗ Optimization ended with status {status}")
        
        print("="*70)
        
        return status
    
    def _calculate_metrics(self):
        """
        Calculate detailed metrics for objective breakdown.
        
        Returns:
            dict: Dictionary with all objective components and operational metrics
        """
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        
        if self.model.SolCount == 0:
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
        
        # Calculate tax benefit on products (DIRECT)
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                y_val = self.y_deliv[r, t].X
                x_val = self.x_out[r, t].X
                metrics['tax_benefit_products'] += self.beta * c_prod * (y_val + x_val)
        
        # Calculate tax benefit on products (INDIRECT)
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                y_val = self.y_deliv[r, t].X
                metrics['tax_benefit_products'] += self.beta * c_prod * y_val
        
        # Calculate tax benefit on transportation (DIRECT)
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                y_val = self.y_deliv[r, t].X
                x_val = self.x_out[r, t].X
                metrics['tax_benefit_transport'] += self.beta * self.c_trans * dist * (y_val + self.alpha * x_val)
        
        # Calculate tax benefit on transportation (INDIRECT)
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                y_val = self.y_deliv[r, t].X
                metrics['tax_benefit_transport'] += self.beta * self.c_trans * dist_indir * y_val
        
        # Calculate own fleet costs (DIRECT)
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                y_val = self.y_deliv[r, t].X
                metrics['own_fleet_costs'] += self.c_trans * dist * y_val
                metrics['total_donated'] += y_val
        
        # Calculate own fleet costs (INDIRECT)
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                y_val = self.y_deliv[r, t].X
                metrics['own_fleet_costs'] += self.c_trans * dist_indir * y_val
                metrics['total_donated'] += y_val
        
        # Calculate outsourcing costs
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                x_val = self.x_out[r, t].X
                metrics['outsourcing_costs'] += self.alpha * self.c_trans * dist * x_val
                metrics['total_outsourced'] += x_val
                metrics['total_donated'] += x_val
        
        # Calculate waste penalty
        for r in self.R_direct + self.R_indirect:
            w_val = self.w[r].X
            metrics['waste_penalty'] += self.pi * w_val
            metrics['total_wasted'] += w_val
        
        # Calculate total available and donation rate
        for r in self.R_direct + self.R_indirect:
            req = self._get_requirement_by_id(r)
            metrics['total_available'] += req['quantity']
        
        if metrics['total_available'] > 0:
            metrics['donation_rate'] = (metrics['total_donated'] / metrics['total_available']) * 100
        
        return metrics
    
    def get_solution(self):
        """
        Extract and return solution details.
        
        Returns:
            dict: Solution summary with key metrics
        """
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        
        if self.model.SolCount == 0:
            return None
        
        solution = {
            'objective_value': self.model.ObjVal,
            'direct_deliveries': {},
            'indirect_deliveries': {},
            'outsourcing': {},
            'waste': {},
            'total_donated': 0,
            'total_wasted': 0,
            'total_outsourced': 0
        }
        
        # Extract direct deliveries
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                val = self.y_deliv[r, t].X
                if val > 0.01:
                    if r not in solution['direct_deliveries']:
                        solution['direct_deliveries'][r] = []
                    solution['direct_deliveries'][r].append((t, val))
                    solution['total_donated'] += val
        
        # Extract indirect deliveries
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                val = self.y_deliv[r, t].X
                if val > 0.01:
                    if r not in solution['indirect_deliveries']:
                        solution['indirect_deliveries'][r] = []
                    solution['indirect_deliveries'][r].append((t, val))
                    solution['total_donated'] += val
        
        # Extract outsourcing
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                val = self.x_out[r, t].X
                if val > 0.01:
                    if r not in solution['outsourcing']:
                        solution['outsourcing'][r] = []
                    solution['outsourcing'][r].append((t, val))
                    solution['total_outsourced'] += val
                    solution['total_donated'] += val
        
        # Extract waste
        for r in self.R_direct + self.R_indirect:
            val = self.w[r].X
            if val > 0.01:
                solution['waste'][r] = val
                solution['total_wasted'] += val
        
        return solution
    
    def display_results(self, detailed=False):
        """
        Display optimization results with detailed breakdown.
        
        Args:
            detailed: If True, show detailed variable values
        """
        if self.model.Status == GRB.OPTIMAL:
            print("\n" + "=" * 80)
            print("OPTIMIZATION RESULTS")
            print("=" * 80)
            
            metrics = self._calculate_metrics()
            
            print(f"\n💰 OBJECTIVE VALUE: ${self.model.ObjVal:,.2f}")
            
            print(f"\n📊 OBJECTIVE BREAKDOWN:")
            print(f"  + Tax benefit (products):     ${metrics['tax_benefit_products']:>15,.2f}")
            print(f"  + Tax benefit (transport):    ${metrics['tax_benefit_transport']:>15,.2f}")
            print(f"  - Own fleet costs:            ${metrics['own_fleet_costs']:>15,.2f}")
            print(f"  - Outsourcing costs:          ${metrics['outsourcing_costs']:>15,.2f}")
            print(f"  - Waste penalty:              ${metrics['waste_penalty']:>15,.2f}")
            print(f"  " + "-" * 60)
            print(f"  = NET BENEFIT:                ${self.model.ObjVal:>15,.2f}")
            
            print(f"\n📦 OPERATIONAL METRICS:")
            print(f"  Total food available:         {metrics['total_available']:>15,.2f} kg")
            print(f"  Total food donated:           {metrics['total_donated']:>15,.2f} kg")
            print(f"  Total food wasted:            {metrics['total_wasted']:>15,.2f} kg")
            print(f"  Donation rate:                {metrics['donation_rate']:>15.2f}%")
            
            if detailed:
                solution = self.get_solution()
                
                print("\n" + "-" * 80)
                print("DETAILED DELIVERIES")
                print("-" * 80)
                
                # Direct deliveries
                if solution['direct_deliveries']:
                    print("\nDirect Requirements (Company Fleet):")
                    for r, deliveries in sorted(solution['direct_deliveries'].items()):
                        req = self._get_requirement_by_id(r)
                        print(f"  Req {r} ({req['origin']} -> {req['destination']}):")
                        for t, qty in deliveries:
                            print(f"    Day {t}: {qty:.2f} kg")
                
                # Indirect deliveries
                if solution['indirect_deliveries']:
                    print("\nIndirect Requirements (via CEDI):")
                    for r, deliveries in sorted(solution['indirect_deliveries'].items()):
                        req = self._get_requirement_by_id(r)
                        print(f"  Req {r} ({req['origin']} -> CEDI -> {req['destination']}):")
                        for t, qty in deliveries:
                            print(f"    Delivery Day {t}: {qty:.2f} kg")
                
                # Outsourcing
                if solution['outsourcing']:
                    print("\nOutsourced Deliveries:")
                    for r, deliveries in sorted(solution['outsourcing'].items()):
                        req = self._get_requirement_by_id(r)
                        print(f"  Req {r} ({req['origin']} -> {req['destination']}):")
                        for t, qty in deliveries:
                            print(f"    Day {t}: {qty:.2f} kg")
                
                # Waste
                if solution['waste']:
                    print("\nWasted Food:")
                    for r, qty in sorted(solution['waste'].items()):
                        req = self._get_requirement_by_id(r)
                        print(f"  Req {r}: {qty:.2f} kg")
            
        else:
            print(f"\n⚠️  MODEL STATUS: {self.model.Status}")
    
    def export_solution(self, filename='solution.json'):
        """Export solution to JSON file."""
        solution = self.get_solution()
        if solution:
            # Convert tuple keys to serializable format
            export_data = {
                'objective_value': solution['objective_value'],
                'total_donated': solution['total_donated'],
                'total_wasted': solution['total_wasted'],
                'total_outsourced': solution['total_outsourced'],
                'direct_deliveries': {
                    str(k): [{'day': t, 'quantity': q} for t, q in v]
                    for k, v in solution['direct_deliveries'].items()
                },
                'indirect_deliveries': {
                    str(k): [{'day': t, 'quantity': q} for t, q in v]
                    for k, v in solution['indirect_deliveries'].items()
                },
                'outsourcing': {
                    str(k): [{'day': t, 'quantity': q} for t, q in v]
                    for k, v in solution['outsourcing'].items()
                },
                'waste': {str(k): v for k, v in solution['waste'].items()}
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Solution exported to '{filename}'")
        else:
            print("\n⚠️  No solution available to export")
    
    def write_model(self, filename='model.lp'):
        """Write model to file."""
        if self.model is not None:
            self.model.write(filename)
            print(f"\n✓ Model written to '{filename}'")
        else:
            print("\n⚠️  No model available to write")


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("FOOD DONATION OPTIMIZATION MODEL - FAST VERSION")
    print("=" * 70)
    
    # Create Gurobi environment with optimized parameters
    with gp.Env(empty=True) as env:
        # Configure environment parameters for better performance
        env.setParam('OutputFlag', 1)  # Show solver output
        env.setParam('TimeLimit', 600)  # 10 minutes max
        env.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions
        env.setParam('Presolve', 2)  # Aggressive presolve
        env.setParam('Threads', 0)  # Use all available threads
        
        # Performance tuning parameters
        env.setParam('NumericFocus', 0)  # Default numerical precision
        env.setParam('ScaleFlag', 1)  # Enable matrix scaling
        
        env.start()
        
        # Use optimizer with context manager
        with FoodDonationOptimizer(env) as optimizer:
            try:
                # Load data
                optimizer.load_data_from_json('instance_distance_exp.json')
                
                # Build model
                optimizer.build_model()
                
                # Optimize
                status = optimizer.optimize()
                
                # Display results if optimal or time limit with solution
                if status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                    if optimizer.model.SolCount > 0:
                        # Display detailed results
                        optimizer.display_results(detailed=False)
                        
                        # Export solution to JSON
                        optimizer.export_solution('solution.json')
                        
                        # Write model to LP file
                        optimizer.write_model('model.lp')
                
            except FileNotFoundError:
                print("\n❌ Error: instance_distance_exp.json not found")
                print("Please make sure the instance file exists in the current directory")
            except Exception as e:
                print(f"\n❌ Error during optimization: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()

