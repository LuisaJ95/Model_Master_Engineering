# %%
import gurobipy as gp
from gurobipy import GRB
import json
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML
import sys
from copy import deepcopy


print("✓ Libraries imported")

# %% [markdown]
# Validation #1

# %%
"""
Scenario 1 Validation Script
Tests model behavior with reduced infrastructure capacity
"""

import gurobipy as gp
from gurobipy import GRB
import json
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import sys

print("="*80)
print("SCENARIO 1 VALIDATION - INSUFFICIENT INFRASTRUCTURE")
print("="*80)


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
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model is not None:
            self.model.dispose()
        return False
    
    def load_data_from_json(self, json_file):
        """Load instance data from JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load requirements and classify them
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
        """Set the objective function."""
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
                # Product benefit
                product_benefit_terms.append(self.beta * c_prod * self.y_deliv[r, t])
                product_benefit_terms.append(self.beta * c_prod * self.x_out[r, t])
                
                # Transport benefit
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
                # Product benefit
                product_benefit_terms.append(self.beta * c_prod * self.y_deliv[r, t])
                
                # Transport benefit
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
    
    def get_metrics(self):
        """Calculate and return solution metrics."""
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        
        if self.model.SolCount == 0:
            return None
        
        metrics = {
            'objective_value': self.model.ObjVal,
            'total_waste': 0,
            'total_outsourced': 0,
            'total_own_fleet': 0,
            'total_donated': 0,
            'total_available': 0,
            'max_fleet_usage_pct': 0
        }
        
        # Calculate waste
        for r in self.R_direct + self.R_indirect:
            metrics['total_waste'] += self.w[r].X
        
        # Calculate outsourcing
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                metrics['total_outsourced'] += self.x_out[r, t].X
        
        # Calculate own fleet usage and max percentage
        for t in self.T:
            fleet_usage = 0
            
            for r in self.R_direct:
                req = self._get_requirement_by_id(r)
                l_r = req['release_date']
                e_r = req['expiration_date']
                if l_r <= t <= e_r:
                    fleet_usage += self.y_deliv[r, t].X
            
            for r in self.R_indirect:
                req = self._get_requirement_by_id(r)
                l_r = req['release_date']
                e_r = req['expiration_date']
                if l_r <= t < e_r:
                    fleet_usage += self.y_pickup[r, t].X
                if l_r + 1 <= t <= e_r:
                    fleet_usage += self.y_deliv[r, t].X
            
            metrics['total_own_fleet'] += fleet_usage
            
            cap_trans = self.transport_capacity[t]
            if cap_trans > 0:
                usage_pct = (fleet_usage / cap_trans) * 100
                metrics['max_fleet_usage_pct'] = max(metrics['max_fleet_usage_pct'], usage_pct)
        
        # Calculate total donated and available
        metrics['total_donated'] = metrics['total_own_fleet'] + metrics['total_outsourced']
        
        for r in self.R_direct + self.R_indirect:
            req = self._get_requirement_by_id(r)
            metrics['total_available'] += req['quantity']
        
        return metrics


def run_scenario(env, json_file, scenario_name, modifications=None):
    """
    Run optimization for a specific scenario.
    
    Args:
        env: Gurobi environment
        json_file: Path to JSON data file
        scenario_name: Name of the scenario
        modifications: Dict with parameter modifications
    
    Returns:
        dict: Scenario results
    """
    print(f"\n{'='*80}")
    print(f"RUNNING: {scenario_name}")
    print(f"{'='*80}")
    
    with FoodDonationOptimizer(env) as optimizer:
        # Load data
        optimizer.load_data_from_json(json_file)
        
        # Apply modifications if provided
        if modifications:
            if 'food_bank_capacity_factor' in modifications:
                factor = modifications['food_bank_capacity_factor']
                print(f"\n  Modifying food bank capacity: {factor*100:.0f}% of original")
                for j in optimizer.B:
                    optimizer.food_bank_capacity[j] *= factor
            
            if 'transport_capacity_factor' in modifications:
                factor = modifications['transport_capacity_factor']
                print(f"  Modifying transport capacity: {factor*100:.0f}% of original")
                for t in optimizer.T:
                    optimizer.transport_capacity[t] *= factor
            
            if 'alpha_multiplier' in modifications:
                multiplier = modifications['alpha_multiplier']
                print(f"  Modifying alpha (outsourcing cost): {multiplier}x")
                optimizer.alpha *= multiplier
        
        # Build and solve
        print("\nBuilding model...")
        optimizer.build_model()
        
        print("Optimizing...")
        status = optimizer.optimize()
        
        # Get results
        if status == GRB.OPTIMAL:
            print(f"\n✓ OPTIMAL solution found")
            metrics = optimizer.get_metrics()
            
            print(f"\nKey Metrics:")
            print(f"  Objective Value:      ${metrics['objective_value']:>15,.2f}")
            print(f"  Total Waste:          {metrics['total_waste']:>15,.2f} kg")
            print(f"  Total Outsourced:     {metrics['total_outsourced']:>15,.2f} kg")
            print(f"  Total Own Fleet:      {metrics['total_own_fleet']:>15,.2f} kg")
            print(f"  Max Fleet Usage:      {metrics['max_fleet_usage_pct']:>15,.1f} %")
            
            return {
                'status': 'optimal',
                'metrics': metrics
            }
        else:
            print(f"\n✗ Optimization failed with status: {status}")
            return {
                'status': 'failed',
                'metrics': None
            }


def validate_scenario_1(baseline_metrics, scenario_metrics):
    """
    Validate Scenario 1 expected behaviors.
    
    Returns:
        dict: Validation results
    """
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS - SCENARIO 1")
    print(f"{'='*80}")
    
    validation = {
        'waste_increased': False,
        'outsourcing_minimal': False,
        'fleet_at_capacity': False,
        'objective_reduced': False
    }
    
    # Check 1: Waste should increase
    waste_increase = scenario_metrics['total_waste'] > baseline_metrics['total_waste']
    waste_pct_change = ((scenario_metrics['total_waste'] - baseline_metrics['total_waste']) / 
                        max(baseline_metrics['total_waste'], 1)) * 100
    
    print(f"\n1. WASTE INCREASE:")
    print(f"   Baseline waste:  {baseline_metrics['total_waste']:,.2f} kg")
    print(f"   Scenario waste:  {scenario_metrics['total_waste']:,.2f} kg")
    print(f"   Change:          {waste_pct_change:+.2f}%")
    
    if waste_increase:
        print("   ✓ PASS: Waste increased as expected")
        validation['waste_increased'] = True
    else:
        print("   ✗ FAIL: Waste did not increase")
    
    # Check 2: Outsourcing should be minimal
    outsource_threshold = baseline_metrics['total_available'] * 0.05  # Less than 5%
    outsourcing_minimal = scenario_metrics['total_outsourced'] < outsource_threshold
    outsource_pct = (scenario_metrics['total_outsourced'] / baseline_metrics['total_available']) * 100
    
    print(f"\n2. OUTSOURCING USAGE:")
    print(f"   Baseline outsourced: {baseline_metrics['total_outsourced']:,.2f} kg")
    print(f"   Scenario outsourced: {scenario_metrics['total_outsourced']:,.2f} kg")
    print(f"   Percentage of total: {outsource_pct:.2f}%")
    
    if outsourcing_minimal:
        print("   ✓ PASS: Outsourcing is minimal (<5%) due to high cost")
        validation['outsourcing_minimal'] = True
    else:
        print("   ✗ FAIL: Outsourcing is not minimal")
    
    # Check 3: Fleet should be at or near capacity
    fleet_at_capacity = scenario_metrics['max_fleet_usage_pct'] >= 95.0
    
    print(f"\n3. OWN FLEET UTILIZATION:")
    print(f"   Baseline max usage:  {baseline_metrics['max_fleet_usage_pct']:.1f}%")
    print(f"   Scenario max usage:  {scenario_metrics['max_fleet_usage_pct']:.1f}%")
    
    if fleet_at_capacity:
        print("   ✓ PASS: Own fleet utilized at full capacity (≥95%)")
        validation['fleet_at_capacity'] = True
    else:
        print("   ✗ FAIL: Own fleet not at full capacity")
    
    # Check 4: Objective should be significantly reduced
    obj_reduced = scenario_metrics['objective_value'] < baseline_metrics['objective_value']
    obj_pct_change = ((scenario_metrics['objective_value'] - baseline_metrics['objective_value']) / 
                      abs(baseline_metrics['objective_value'])) * 100
    
    print(f"\n4. OBJECTIVE FUNCTION:")
    print(f"   Baseline objective:  ${baseline_metrics['objective_value']:,.2f}")
    print(f"   Scenario objective:  ${scenario_metrics['objective_value']:,.2f}")
    print(f"   Change:              {obj_pct_change:+.2f}%")
    
    if obj_reduced:
        print("   ✓ PASS: Objective reduced as expected")
        validation['objective_reduced'] = True
    else:
        print("   ✗ FAIL: Objective did not reduce")
    
    # Overall validation
    all_passed = all(validation.values())
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
    else:
        failed_count = sum(1 for v in validation.values() if not v)
        print(f"⚠️ {failed_count}/4 VALIDATIONS FAILED")
    print(f"{'='*80}")
    
    return validation


def create_comparison_plots(baseline_metrics, scenario_metrics):
    """Create visualization comparing baseline and scenario 1."""
    print("\nGenerating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Scenario 1 vs Baseline Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Waste Comparison
    ax1 = axes[0, 0]
    scenarios = ['Baseline', 'Scenario 1']
    waste_values = [baseline_metrics['total_waste'], scenario_metrics['total_waste']]
    bars1 = ax1.bar(scenarios, waste_values, color=['green', 'red'], alpha=0.7)
    ax1.set_ylabel('Waste (kg)', fontweight='bold')
    ax1.set_title('Total Waste', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, waste_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(waste_values)*0.02,
                f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Outsourcing Comparison
    ax2 = axes[0, 1]
    outsource_values = [baseline_metrics['total_outsourced'], scenario_metrics['total_outsourced']]
    bars2 = ax2.bar(scenarios, outsource_values, color=['blue', 'orange'], alpha=0.7)
    ax2.set_ylabel('Quantity (kg)', fontweight='bold')
    ax2.set_title('Outsourced Deliveries', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, outsource_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(outsource_values + [1])*0.02,
                f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Fleet Utilization
    ax3 = axes[1, 0]
    fleet_usage = [baseline_metrics['max_fleet_usage_pct'], scenario_metrics['max_fleet_usage_pct']]
    bars3 = ax3.bar(scenarios, fleet_usage, color=['cyan', 'purple'], alpha=0.7)
    ax3.set_ylabel('Utilization (%)', fontweight='bold')
    ax3.set_title('Maximum Fleet Utilization', fontweight='bold')
    ax3.set_ylim([0, 105])
    ax3.axhline(y=100, color='red', linestyle='--', linewidth=1, label='Capacity')
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend()
    
    for bar, val in zip(bars3, fleet_usage):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Objective Value
    ax4 = axes[1, 1]
    obj_values = [baseline_metrics['objective_value'], scenario_metrics['objective_value']]
    colors = ['green' if v >= 0 else 'red' for v in obj_values]
    bars4 = ax4.bar(scenarios, obj_values, color=colors, alpha=0.7)
    ax4.set_ylabel('Net Benefit ($)', fontweight='bold')
    ax4.set_title('Objective Function Value', fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars4, obj_values):
        y_pos = val + (abs(val)*0.05 if val >= 0 else -abs(val)*0.05)
        ax4.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'${val:,.0f}', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('scenario1_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Plots saved to 'scenario1_comparison.png'")
    
    return fig


def main():
    """Main execution function."""
    
    # Check if data file exists
    json_file = 'instance_distance_exp.json'
    try:
        with open(json_file, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"\n❌ ERROR: File '{json_file}' not found!")
        print("Please ensure the data file is in the same directory.")
        sys.exit(1)
    
    # Create Gurobi environment
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)  # Suppress solver output for cleaner results
        env.setParam('TimeLimit', 300)
        env.start()
        
        # Run BASELINE scenario
        baseline_result = run_scenario(
            env=env,
            json_file=json_file,
            scenario_name="BASELINE (Original Parameters)"
        )
        
        if baseline_result['status'] != 'optimal':
            print("\n❌ ERROR: Baseline scenario failed to solve optimally")
            sys.exit(1)
        
        # Run SCENARIO 1 with modifications
        scenario1_result = run_scenario(
            env=env,
            json_file=json_file,
            scenario_name="SCENARIO 1 (Insufficient Infrastructure)",
            modifications={
                'food_bank_capacity_factor': 0.10,      # 10% of original
                'transport_capacity_factor': 0.15,       # 15% of original
                'alpha_multiplier': 5.0                  # 5x outsourcing cost
            }
        )
        
        if scenario1_result['status'] != 'optimal':
            print("\n❌ ERROR: Scenario 1 failed to solve optimally")
            sys.exit(1)
        
        # Validate results
        validation_results = validate_scenario_1(
            baseline_result['metrics'],
            scenario1_result['metrics']
        )
        
        # Create comparison plots
        create_comparison_plots(
            baseline_result['metrics'],
            scenario1_result['metrics']
        )
        
        # Save detailed results to JSON
        results_summary = {
            'baseline': baseline_result['metrics'],
            'scenario_1': scenario1_result['metrics'],
            'validation': validation_results
        }
        
        with open('scenario1_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("\n✓ Results saved to 'scenario1_results.json'")
        print("\n" + "="*80)
        print("SCENARIO 1 VALIDATION COMPLETE")
        print("="*80)


if __name__ == "__main__":
    main()


# %% [markdown]
# Validation #2

# %%
"""
Sensitivity Analysis for Food Donation Optimization Model
Analysis of 12 scenarios varying:
- Fiscal Incentive (β): Low (0.05), Baseline, High (0.95)
- Waste Penalty (π): Normal, High (10x)
- Transportation Cost (c_trans): Normal, High (5x)
"""

import gurobipy as gp
from gurobipy import GRB
import json
import pandas as pd
import time
from copy import deepcopy
import sys


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
        """Helper function to get requirement data by ID."""
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
        # Component 1: Tax benefit on products
        product_benefit_terms = []
        
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
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            benefit_coef = self.beta * c_prod
            
            for t in range(l_r + 1, e_r + 1):
                product_benefit_terms.append(benefit_coef * self.y_deliv[r, t])
        
        # Component 2: Tax benefit on transportation
        transport_benefit_terms = []
        
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            benefit_coef = self.beta * self.c_trans * dist
            
            for t in range(l_r, e_r + 1):
                transport_benefit_terms.append(benefit_coef * self.y_deliv[r, t])
                transport_benefit_terms.append(benefit_coef * self.alpha * self.x_out[r, t])
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            benefit_coef = self.beta * self.c_trans * dist_indir
            
            for t in range(l_r + 1, e_r + 1):
                transport_benefit_terms.append(benefit_coef * self.y_deliv[r, t])
        
        # Component 3: Transportation costs (company fleet)
        transport_cost_terms = []
        
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            cost_coef = self.c_trans * dist
            
            for t in range(l_r, e_r + 1):
                transport_cost_terms.append(cost_coef * self.y_deliv[r, t])
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            cost_coef = self.c_trans * dist_indir
            
            for t in range(l_r + 1, e_r + 1):
                transport_cost_terms.append(cost_coef * self.y_deliv[r, t])
        
        # Component 4: Outsourcing costs
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
        
        # Constraint (4): Food bank receiving capacity
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
        
        # Constraint (5): Transportation capacity
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
        
        # Constraint (6): Synchronization for INDIRECT requirements
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r):
                self.model.addConstr(
                    self.y_pickup[r, t] == self.y_deliv[r, t + 1],
                    name=f"sync_{r}_{t}"
                )
    
    def optimize(self, verbose=False):
        """Solve the optimization model."""
        if verbose:
            self.model.setParam('OutputFlag', 1)
        else:
            self.model.setParam('OutputFlag', 0)
        
        self.model.optimize()
        return self.model.Status
    
    def get_metrics(self):
        """Calculate detailed metrics for the solution."""
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        
        if self.model.SolCount == 0:
            return None
        
        metrics = {
            'objective_value': self.model.ObjVal,
            'tax_benefit_products': 0,
            'tax_benefit_transport': 0,
            'own_fleet_costs': 0,
            'outsourcing_costs': 0,
            'waste_penalty': 0,
            'total_donated': 0,
            'total_wasted': 0,
            'total_outsourced': 0,
            'own_fleet_used': 0,
            'total_available': 0,
            'donation_rate': 0
        }
        
        # Calculate tax benefit on products
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
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                y_val = self.y_deliv[r, t].X
                metrics['tax_benefit_products'] += self.beta * c_prod * y_val
        
        # Calculate tax benefit on transportation
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                y_val = self.y_deliv[r, t].X
                x_val = self.x_out[r, t].X
                metrics['tax_benefit_transport'] += self.beta * self.c_trans * dist * (y_val + self.alpha * x_val)
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                y_val = self.y_deliv[r, t].X
                metrics['tax_benefit_transport'] += self.beta * self.c_trans * dist_indir * y_val
        
        # Calculate costs and quantities
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                y_val = self.y_deliv[r, t].X
                x_val = self.x_out[r, t].X
                metrics['own_fleet_costs'] += self.c_trans * dist * y_val
                metrics['outsourcing_costs'] += self.alpha * self.c_trans * dist * x_val
                metrics['own_fleet_used'] += y_val
                metrics['total_outsourced'] += x_val
                metrics['total_donated'] += y_val + x_val
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                y_val = self.y_deliv[r, t].X
                metrics['own_fleet_costs'] += self.c_trans * dist_indir * y_val
                metrics['own_fleet_used'] += y_val
                metrics['total_donated'] += y_val
        
        # Calculate waste
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


class SensitivityAnalyzer:
    """
    Manages sensitivity analysis across multiple scenarios.
    """
    
    def __init__(self, json_file):
        """
        Initialize analyzer with data file.
        
        Args:
            json_file: Path to JSON file with instance data
        """
        self.json_file = json_file
        self.baseline_data = None
        self.scenarios = []
        self.results = []
        
        # Load baseline data
        with open(json_file, 'r', encoding='utf-8') as f:
            self.baseline_data = json.load(f)
        
        # Store baseline parameters
        self.beta_baseline = self.baseline_data['parameters']['beta_1']
        self.pi_baseline = self.baseline_data['parameters']['pi_penalty']
        self.c_trans_baseline = self.baseline_data['parameters']['c_trans']
    
    def define_scenarios(self):
        """
        Define the 12 scenarios for sensitivity analysis.
        
        Dimensions:
        - Incentive (β): Low (0.05), Baseline, High (0.95)
        - Penalty (π): Normal, High (10x)
        - Transport Cost (c_trans): Normal, High (5x)
        """
        self.scenarios = [
            # Run 1: Low incentive, standard penalty, standard cost
            {
                'run': 1,
                'incentive': 'Low',
                'penalty': 'Standard',
                'cost_trans': 'Standard',
                'beta': 0.05,
                'pi': self.pi_baseline,
                'c_trans': self.c_trans_baseline,
                'expected': 'Moderate waste, average fleet, minimal outsourcing'
            },
            # Run 2: Low incentive, standard penalty, high cost
            {
                'run': 2,
                'incentive': 'Low',
                'penalty': 'Standard',
                'cost_trans': 'High',
                'beta': 0.05,
                'pi': self.pi_baseline,
                'c_trans': self.c_trans_baseline * 5,
                'expected': 'Maximum waste, minimal fleet, zero outsourcing'
            },
            # Run 3: Low incentive, high penalty, standard cost
            {
                'run': 3,
                'incentive': 'Low',
                'penalty': 'High',
                'cost_trans': 'Standard',
                'beta': 0.05,
                'pi': self.pi_baseline * 10,
                'c_trans': self.c_trans_baseline,
                'expected': 'Zero/min waste, fleet saturation, moderate outsourcing'
            },
            # Run 4: Low incentive, high penalty, high cost
            {
                'run': 4,
                'incentive': 'Low',
                'penalty': 'High',
                'cost_trans': 'High',
                'beta': 0.05,
                'pi': self.pi_baseline * 10,
                'c_trans': self.c_trans_baseline * 5,
                'expected': 'Zero/min waste, fleet saturation, maximum outsourcing'
            },
            # Run 5: Standard incentive, standard penalty, standard cost (BASELINE)
            {
                'run': 5,
                'incentive': 'Standard',
                'penalty': 'Standard',
                'cost_trans': 'Standard',
                'beta': self.beta_baseline,
                'pi': self.pi_baseline,
                'c_trans': self.c_trans_baseline,
                'expected': 'Min waste, optimal fleet, moderate outsourcing'
            },
            # Run 6: Standard incentive, standard penalty, high cost
            {
                'run': 6,
                'incentive': 'Standard',
                'penalty': 'Standard',
                'cost_trans': 'High',
                'beta': self.beta_baseline,
                'pi': self.pi_baseline,
                'c_trans': self.c_trans_baseline * 5,
                'expected': 'Moderate waste, fleet saturation, minimal outsourcing'
            },
            # Run 7: Standard incentive, high penalty, standard cost
            {
                'run': 7,
                'incentive': 'Standard',
                'penalty': 'High',
                'cost_trans': 'Standard',
                'beta': self.beta_baseline,
                'pi': self.pi_baseline * 10,
                'c_trans': self.c_trans_baseline,
                'expected': 'Zero waste, fleet saturation, moderate outsourcing'
            },
            # Run 8: Standard incentive, high penalty, high cost
            {
                'run': 8,
                'incentive': 'Standard',
                'penalty': 'High',
                'cost_trans': 'High',
                'beta': self.beta_baseline,
                'pi': self.pi_baseline * 10,
                'c_trans': self.c_trans_baseline * 5,
                'expected': 'Zero waste, fleet saturation, high outsourcing'
            },
            # Run 9: High incentive, standard penalty, standard cost
            {
                'run': 9,
                'incentive': 'High',
                'penalty': 'Standard',
                'cost_trans': 'Standard',
                'beta': 0.95,
                'pi': self.pi_baseline,
                'c_trans': self.c_trans_baseline,
                'expected': 'Zero waste, fleet saturation, high outsourcing'
            },
            # Run 10: High incentive, standard penalty, high cost
            {
                'run': 10,
                'incentive': 'High',
                'penalty': 'Standard',
                'cost_trans': 'High',
                'beta': 0.95,
                'pi': self.pi_baseline,
                'c_trans': self.c_trans_baseline * 5,
                'expected': 'Zero waste, fleet saturation, maximum outsourcing'
            },
            # Run 11: High incentive, high penalty, standard cost
            {
                'run': 11,
                'incentive': 'High',
                'penalty': 'High',
                'cost_trans': 'Standard',
                'beta': 0.95,
                'pi': self.pi_baseline * 10,
                'c_trans': self.c_trans_baseline,
                'expected': 'Zero waste, fleet saturation, high outsourcing'
            },
            # Run 12: High incentive, high penalty, high cost
            {
                'run': 12,
                'incentive': 'High',
                'penalty': 'High',
                'cost_trans': 'High',
                'beta': 0.95,
                'pi': self.pi_baseline * 10,
                'c_trans': self.c_trans_baseline * 5,
                'expected': 'Zero waste, fleet saturation, maximum outsourcing'
            }
        ]
        
        print(f"✓ Defined {len(self.scenarios)} scenarios for analysis")
    
    def run_analysis(self):
        """Execute all scenarios and collect results."""
        print("\n" + "=" * 80)
        print("STARTING SENSITIVITY ANALYSIS")
        print("=" * 80)
        
        total_start = time.time()
        
        for scenario in self.scenarios:
            print(f"\n{'='*80}")
            print(f"RUN {scenario['run']}: β={scenario['incentive']}, "
                  f"π={scenario['penalty']}, c_trans={scenario['cost_trans']}")
            print(f"{'='*80}")
            
            scenario_start = time.time()
            
            # Create temporary modified data
            temp_data = deepcopy(self.baseline_data)
            temp_data['parameters']['beta_1'] = scenario['beta']
            temp_data['parameters']['pi_penalty'] = scenario['pi']
            temp_data['parameters']['c_trans'] = scenario['c_trans']
            
            # Write temporary data file
            temp_file = 'temp_scenario.json'
            with open(temp_file, 'w') as f:
                json.dump(temp_data, f)
            
            # Create environment and solve
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.setParam('TimeLimit', 300)
                env.start()
                
                with FoodDonationOptimizer(env) as optimizer:
                    try:
                        optimizer.load_data_from_json(temp_file)
                        optimizer.build_model()
                        status = optimizer.optimize(verbose=False)
                        
                        if status == GRB.OPTIMAL:
                            metrics = optimizer.get_metrics()
                            
                            # Store results
                            result = {
                                'run': scenario['run'],
                                'incentive': scenario['incentive'],
                                'penalty': scenario['penalty'],
                                'cost_trans': scenario['cost_trans'],
                                'beta': scenario['beta'],
                                'pi': scenario['pi'],
                                'c_trans': scenario['c_trans'],
                                'status': 'Optimal',
                                'objective_value': metrics['objective_value'],
                                'total_wasted': metrics['total_wasted'],
                                'own_fleet_used': metrics['own_fleet_used'],
                                'total_outsourced': metrics['total_outsourced'],
                                'donation_rate': metrics['donation_rate'],
                                'tax_benefit_products': metrics['tax_benefit_products'],
                                'tax_benefit_transport': metrics['tax_benefit_transport'],
                                'own_fleet_costs': metrics['own_fleet_costs'],
                                'outsourcing_costs': metrics['outsourcing_costs'],
                                'waste_penalty': metrics['waste_penalty'],
                                'expected': scenario['expected']
                            }
                            
                            self.results.append(result)
                            
                            elapsed = time.time() - scenario_start
                            print(f"\n✓ Scenario completed in {elapsed:.2f}s")
                            print(f"  Objective: ${metrics['objective_value']:,.2f}")
                            print(f"  Waste: {metrics['total_wasted']:.2f} kg "
                                  f"({100 - metrics['donation_rate']:.1f}%)")
                            print(f"  Own Fleet: {metrics['own_fleet_used']:.2f} kg")
                            print(f"  Outsourcing: {metrics['total_outsourced']:.2f} kg")
                        else:
                            print(f"\n✗ Optimization failed with status: {status}")
                            self.results.append({
                                'run': scenario['run'],
                                'status': f'Failed (Status: {status})',
                                'objective_value': None
                            })
                    
                    except Exception as e:
                        print(f"\n✗ Error in scenario: {e}")
                        self.results.append({
                            'run': scenario['run'],
                            'status': f'Error: {str(e)}',
                            'objective_value': None
                        })
        
        total_elapsed = time.time() - total_start
        print(f"\n{'='*80}")
        print(f"✓ SENSITIVITY ANALYSIS COMPLETED in {total_elapsed:.2f}s")
        print(f"{'='*80}")
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "=" * 100)
        print("SENSITIVITY ANALYSIS REPORT")
        print("=" * 100)
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Print summary table
        print("\n" + "-" * 100)
        print("SCENARIO SUMMARY")
        print("-" * 100)
        
        summary_cols = ['run', 'incentive', 'penalty', 'cost_trans', 
                       'objective_value', 'total_wasted', 'own_fleet_used', 
                       'total_outsourced', 'donation_rate']
        
        print(df[summary_cols].to_string(index=False))
        
        # Detailed breakdown
        print("\n" + "-" * 100)
        print("OBJECTIVE FUNCTION BREAKDOWN")
        print("-" * 100)
        
        breakdown_cols = ['run', 'tax_benefit_products', 'tax_benefit_transport',
                         'own_fleet_costs', 'outsourcing_costs', 'waste_penalty',
                         'objective_value']
        
        print(df[breakdown_cols].to_string(index=False, float_format='%.2f'))
        
        # Key insights
        print("\n" + "-" * 100)
        print("KEY INSIGHTS")
        print("-" * 100)
        
        # Find best and worst scenarios
        best_idx = df['objective_value'].idxmax()
        worst_idx = df['objective_value'].idxmin()
        
        print(f"\n✓ BEST SCENARIO: Run {df.loc[best_idx, 'run']}")
        print(f"  Configuration: β={df.loc[best_idx, 'incentive']}, "
              f"π={df.loc[best_idx, 'penalty']}, "
              f"c_trans={df.loc[best_idx, 'cost_trans']}")
        print(f"  Objective: ${df.loc[best_idx, 'objective_value']:,.2f}")
        print(f"  Donation Rate: {df.loc[best_idx, 'donation_rate']:.2f}%")
        
        print(f"\n✗ WORST SCENARIO: Run {df.loc[worst_idx, 'run']}")
        print(f"  Configuration: β={df.loc[worst_idx, 'incentive']}, "
              f"π={df.loc[worst_idx, 'penalty']}, "
              f"c_trans={df.loc[worst_idx, 'cost_trans']}")
        print(f"  Objective: ${df.loc[worst_idx, 'objective_value']:,.2f}")
        print(f"  Donation Rate: {df.loc[worst_idx, 'donation_rate']:.2f}%")
        
        # Impact analysis
        print("\n" + "-" * 100)
        print("IMPACT ANALYSIS BY PARAMETER")
        print("-" * 100)
        
        # Group by incentive level
        print("\nBy INCENTIVE LEVEL (β):")
        incentive_impact = df.groupby('incentive')['objective_value'].agg(['mean', 'min', 'max'])
        print(incentive_impact.to_string())
        
        # Group by penalty level
        print("\nBy PENALTY LEVEL (π):")
        penalty_impact = df.groupby('penalty')['total_wasted'].agg(['mean', 'min', 'max'])
        print(penalty_impact.to_string())
        
        # Group by transport cost
        print("\nBy TRANSPORT COST (c_trans):")
        cost_impact = df.groupby('cost_trans')['total_outsourced'].agg(['mean', 'min', 'max'])
        print(cost_impact.to_string())
        
        # Expected vs Actual comparison
        print("\n" + "-" * 100)
        print("EXPECTED BEHAVIOR VALIDATION")
        print("-" * 100)
        
        for _, row in df.iterrows():
            print(f"\nRun {int(row['run'])}:")
            print(f"  Expected: {row['expected']}")
            print(f"  Actual: Waste={row['total_wasted']:.1f} kg, "
                  f"Fleet={row['own_fleet_used']:.1f} kg, "
                  f"Outsource={row['total_outsourced']:.1f} kg, "
                  f"OF=${row['objective_value']:,.2f}")
        
        return df
    
    def export_results(self, filename='sensitivity_results.csv'):
        """Export results to CSV file."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            print(f"\n✓ Results exported to '{filename}'")
        else:
            print("\n⚠️  No results available to export")


def main():
    """Main execution function."""
    print("=" * 100)
    print("SENSITIVITY ANALYSIS - FOOD DONATION OPTIMIZATION MODEL")
    print("=" * 100)
    print("\nAnalyzing 12 scenarios with varying:")
    print("  - Fiscal Incentive (β): Low (0.05), Baseline, High (0.95)")
    print("  - Waste Penalty (π): Normal, High (10x)")
    print("  - Transportation Cost (c_trans): Normal, High (5x)")
    
    try:
        # Create analyzer
        analyzer = SensitivityAnalyzer('instance_distance_exp.json')
        
        # Define scenarios
        analyzer.define_scenarios()
        
        # Run analysis
        analyzer.run_analysis()
        
        # Generate report
        results_df = analyzer.generate_report()
        
        # Export results
        analyzer.export_results('sensitivity_results.csv')
        
        print("\n" + "=" * 100)
        print("ANALYSIS COMPLETE")
        print("=" * 100)
        
    except FileNotFoundError:
        print("\n❌ Error: instance_distance_exp.json not found")
        print("Please make sure the instance file exists in the current directory")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



