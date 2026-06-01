# %%
"""
Multi-Instance Food Donation Optimization Analysis
Runs optimization model on 12 monthly instances and generates comparative results
"""

import json
import gurobipy as gp
from gurobipy import GRB
import time
import pandas as pd
from datetime import datetime
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
        
        # Instance name for tracking
        self.instance_name = ""
    
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
        self.instance_name = json_file
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load requirements and classify them
        self.requirements = data['requirements']
        self.requirements_dict = {req['id']: req for req in self.requirements}
        
        self.R_direct = []
        self.R_indirect = []
        
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
            self.w[r] = self.model.addVar(
                lb=0, vtype=GRB.CONTINUOUS, name=f"w_{r}"
            )
    
    def _set_objective(self):
        """Set the objective function."""
        product_benefit_terms = []
        transport_benefit_terms = []
        transport_cost_terms = []
        outsource_cost_terms = []
        waste_terms = []
        
        # Direct requirements - benefits and costs
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                y_var = self.y_deliv[r, t]
                x_var = self.x_out[r, t]
                
                # Tax benefits
                product_benefit_terms.append(self.beta * c_prod * (y_var + x_var))
                transport_benefit_terms.append(
                    self.beta * self.c_trans * dist * (y_var + self.alpha * x_var)
                )
                
                # Costs
                transport_cost_terms.append(self.c_trans * dist * y_var)
                outsource_cost_terms.append(self.alpha * self.c_trans * dist * x_var)
        
        # Indirect requirements - benefits and costs
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                y_var = self.y_deliv[r, t]
                
                # Tax benefits
                product_benefit_terms.append(self.beta * c_prod * y_var)
                transport_benefit_terms.append(
                    self.beta * self.c_trans * dist_indir * y_var
                )
                
                # Costs
                transport_cost_terms.append(self.c_trans * dist_indir * y_var)
        
        # Waste penalties
        waste_terms = [self.pi * self.w[r] for r in self.R_direct + self.R_indirect]
        
        # Objective function
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
    
    def calculate_metrics(self):
        """Calculate comprehensive metrics from the solution."""
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        
        if self.model.SolCount == 0:
            return None
        
        metrics = {
            'instance_name': self.instance_name,
            'status': self.model.Status,
            'objective_value': self.model.ObjVal,
            'solve_time': self.model.Runtime,
            'mip_gap': self.model.MIPGap if self.model.Status == GRB.TIME_LIMIT else 0,
            'num_variables': self.model.NumVars,
            'num_constraints': self.model.NumConstrs,
            'tax_benefit_products': 0,
            'tax_benefit_transport': 0,
            'own_fleet_costs': 0,
            'outsourcing_costs': 0,
            'waste_penalty': 0,
            'total_donated': 0,
            'total_wasted': 0,
            'total_outsourced': 0,
            'total_available': 0,
            'total_direct_fleet': 0,
            'total_indirect_fleet': 0,
            'donation_rate': 0,
            'waste_rate': 0,
            'outsource_rate': 0,
            'num_direct_req': len(self.R_direct),
            'num_indirect_req': len(self.R_indirect),
            'num_food_banks': len(self.B),
            'num_periods': len(self.T)
        }
        
        # Calculate total available
        for r in self.R_direct + self.R_indirect:
            req = self._get_requirement_by_id(r)
            metrics['total_available'] += req['quantity']
        
        # Calculate metrics from DIRECT requirements
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r + 1):
                y_val = self.y_deliv[r, t].X
                x_val = self.x_out[r, t].X
                
                # Benefits
                metrics['tax_benefit_products'] += self.beta * c_prod * (y_val + x_val)
                metrics['tax_benefit_transport'] += self.beta * self.c_trans * dist * (y_val + self.alpha * x_val)
                
                # Costs
                metrics['own_fleet_costs'] += self.c_trans * dist * y_val
                metrics['outsourcing_costs'] += self.alpha * self.c_trans * dist * x_val
                
                # Quantities
                metrics['total_direct_fleet'] += y_val
                metrics['total_outsourced'] += x_val
                metrics['total_donated'] += y_val + x_val
        
        # Calculate metrics from INDIRECT requirements
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            product = req['product']
            c_prod = self.product_costs.get(product, 0)
            dist_indir = req['dist_indirect']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r + 1, e_r + 1):
                y_val = self.y_deliv[r, t].X
                
                # Benefits
                metrics['tax_benefit_products'] += self.beta * c_prod * y_val
                metrics['tax_benefit_transport'] += self.beta * self.c_trans * dist_indir * y_val
                
                # Costs
                metrics['own_fleet_costs'] += self.c_trans * dist_indir * y_val
                
                # Quantities
                metrics['total_indirect_fleet'] += y_val
                metrics['total_donated'] += y_val
        
        # Calculate waste
        for r in self.R_direct + self.R_indirect:
            w_val = self.w[r].X
            metrics['waste_penalty'] += self.pi * w_val
            metrics['total_wasted'] += w_val
        
        # Calculate rates
        if metrics['total_available'] > 0:
            metrics['donation_rate'] = (metrics['total_donated'] / metrics['total_available']) * 100
            metrics['waste_rate'] = (metrics['total_wasted'] / metrics['total_available']) * 100
            metrics['outsource_rate'] = (metrics['total_outsourced'] / metrics['total_available']) * 100
        
        return metrics


def run_multi_instance_analysis():
    """Run optimization on all 12 monthly instances and generate comparative analysis."""
    
    # Define instances
    instances = [
        'instance_distance_2024_08.json',
        'instance_distance_2024_09.json',
        'instance_distance_2024_10.json',
        'instance_distance_2024_11.json',
        'instance_distance_2024_12.json',
        'instance_distance_2025_01.json',
        'instance_distance_2025_02.json',
        'instance_distance_2025_03.json',
        'instance_distance_2025_04.json',
        'instance_distance_2025_05.json',
        'instance_distance_2025_06.json',
        'instance_distance_2025_07.json'
    ]
    
    print("="*80)
    print("MULTI-INSTANCE FOOD DONATION OPTIMIZATION ANALYSIS")
    print("="*80)
    print(f"\nTotal instances to process: {len(instances)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Store results
    all_results = []
    failed_instances = []
    
    # Create Gurobi environment
    with gp.Env(empty=True) as env:
        # Configure environment
        env.setParam('OutputFlag', 0)  # Suppress Gurobi output
        env.setParam('TimeLimit', 600)  # 10 minutes per instance
        env.setParam('MIPFocus', 1)
        env.setParam('Presolve', 2)
        env.setParam('Threads', 0)
        env.start()
        
        # Process each instance
        for idx, instance_file in enumerate(instances, 1):
            print(f"\n{'='*80}")
            print(f"[{idx}/{len(instances)}] Processing: {instance_file}")
            print(f"{'='*80}")
            
            # Check if file exists
            if not os.path.exists(instance_file):
                print(f"  ❌ ERROR: File not found - {instance_file}")
                failed_instances.append({
                    'instance': instance_file,
                    'error': 'File not found'
                })
                continue
            
            try:
                start_time = time.time()
                
                # Create optimizer
                with FoodDonationOptimizer(env) as optimizer:
                    # Load data
                    print(f"  Loading data...")
                    optimizer.load_data_from_json(instance_file)
                    print(f"    - Requirements: {len(optimizer.requirements)} "
                          f"(Direct: {len(optimizer.R_direct)}, Indirect: {len(optimizer.R_indirect)})")
                    print(f"    - Time periods: {len(optimizer.T)}")
                    print(f"    - Food banks: {len(optimizer.B)}")
                    
                    # Build model
                    print(f"  Building model...")
                    optimizer.build_model()
                    print(f"    - Variables: {optimizer.model.NumVars}")
                    print(f"    - Constraints: {optimizer.model.NumConstrs}")
                    
                    # Optimize
                    print(f"  Optimizing...")
                    status = optimizer.optimize()
                    
                    elapsed = time.time() - start_time
                    
                    # Get results
                    if status == GRB.OPTIMAL:
                        print(f"  ✓ OPTIMAL solution found in {elapsed:.2f}s")
                        metrics = optimizer.calculate_metrics()
                        
                        if metrics:
                            metrics['elapsed_time'] = elapsed
                            all_results.append(metrics)
                            
                            print(f"    - Objective value: ${metrics['objective_value']:,.2f}")
                            print(f"    - Donation rate: {metrics['donation_rate']:.2f}%")
                            print(f"    - Waste rate: {metrics['waste_rate']:.2f}%")
                    
                    elif status == GRB.TIME_LIMIT:
                        print(f"  ⚠️  TIME LIMIT reached after {elapsed:.2f}s")
                        if optimizer.model.SolCount > 0:
                            metrics = optimizer.calculate_metrics()
                            if metrics:
                                metrics['elapsed_time'] = elapsed
                                all_results.append(metrics)
                                print(f"    - Best solution: ${metrics['objective_value']:,.2f}")
                                print(f"    - Gap: {metrics['mip_gap']*100:.2f}%")
                        else:
                            failed_instances.append({
                                'instance': instance_file,
                                'error': 'Time limit - no solution found'
                            })
                    
                    elif status == GRB.INFEASIBLE:
                        print(f"  ❌ MODEL IS INFEASIBLE")
                        failed_instances.append({
                            'instance': instance_file,
                            'error': 'Infeasible'
                        })
                    
                    else:
                        print(f"  ❌ Optimization failed with status {status}")
                        failed_instances.append({
                            'instance': instance_file,
                            'error': f'Status {status}'
                        })
            
            except Exception as e:
                print(f"  ❌ ERROR: {str(e)}")
                failed_instances.append({
                    'instance': instance_file,
                    'error': str(e)
                })
                continue
    
    # Generate comprehensive report
    print(f"\n\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successfully processed: {len(all_results)}/{len(instances)} instances")
    
    if failed_instances:
        print(f"\n⚠️  Failed instances: {len(failed_instances)}")
        for fail in failed_instances:
            print(f"  - {fail['instance']}: {fail['error']}")
    
    # Create DataFrame and save results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Extract month from instance name
        df['month'] = df['instance_name'].str.extract(r'(\d{4}_\d{2})')[0]
        
        # Save detailed results
        csv_filename = 'multi_instance_results.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\n✓ Detailed results saved to: {csv_filename}")
        
        # Save summary JSON
        json_filename = 'multi_instance_summary.json'
        summary = {
            'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_instances': len(instances),
            'successful': len(all_results),
            'failed': len(failed_instances),
            'results': all_results,
            'failed_instances': failed_instances
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✓ Summary saved to: {json_filename}")
        
        # Print comparative analysis
        print_comparative_analysis(df)
        
        # Generate markdown report
        generate_markdown_report(df, failed_instances)
    
    else:
        print("\n❌ No results to analyze - all instances failed")
    
    return all_results, failed_instances


def print_comparative_analysis(df):
    """Print comparative analysis of results."""
    print(f"\n\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}\n")
    
    # Key metrics summary
    print("SUMMARY STATISTICS")
    print("-" * 80)
    
    summary_cols = [
        'objective_value', 'total_donated', 'total_wasted', 
        'donation_rate', 'waste_rate', 'outsource_rate', 
        'solve_time', 'total_direct_req', 'total_indirect_req'
    ]
    
    # Filter columns that exist
    available_cols = [col for col in summary_cols if col in df.columns]
    
    print(df[available_cols].describe().round(2).to_string())
    
    # Monthly comparison table
    print(f"\n\nMONTHLY COMPARISON")
    print("-" * 80)
    
    comparison_cols = ['month', 'objective_value', 'total_donated', 'total_wasted', 
                       'donation_rate', 'waste_rate', 'solve_time']
    
    # Filter columns that exist
    comparison_cols = [col for col in comparison_cols if col in df.columns]
    
    if comparison_cols:
        comparison_df = df[comparison_cols].copy()
        
        # Format numeric columns
        for col in comparison_df.columns:
            if col == 'month':
                continue
            elif col == 'objective_value':
                comparison_df[col] = comparison_df[col].apply(lambda x: f"${x:,.0f}")
            elif 'rate' in col:
                comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2f}%")
            elif 'time' in col:
                comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2f}s")
            else:
                comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:,.0f}")
        
        print(comparison_df.to_string(index=False))
    
    # Best and worst months
    if 'objective_value' in df.columns and 'month' in df.columns:
        print(f"\n\nKEY INSIGHTS")
        print("-" * 80)
        
        best_month = df.loc[df['objective_value'].idxmax()]
        worst_month = df.loc[df['objective_value'].idxmin()]
        
        print(f"Best performing month: {best_month['month']}")
        print(f"  - Objective value: ${best_month['objective_value']:,.2f}")
        print(f"  - Donation rate: {best_month['donation_rate']:.2f}%")
        print(f"  - Waste rate: {best_month['waste_rate']:.2f}%")
        
        print(f"\nWorst performing month: {worst_month['month']}")
        print(f"  - Objective value: ${worst_month['objective_value']:,.2f}")
        print(f"  - Donation rate: {worst_month['donation_rate']:.2f}%")
        print(f"  - Waste rate: {worst_month['waste_rate']:.2f}%")
        
        # Average metrics
        print(f"\nAverage across all months:")
        print(f"  - Objective value: ${df['objective_value'].mean():,.2f}")
        print(f"  - Donation rate: {df['donation_rate'].mean():.2f}%")
        print(f"  - Waste rate: {df['waste_rate'].mean():.2f}%")
        print(f"  - Total donated: {df['total_donated'].sum():,.0f} kg")
        print(f"  - Total wasted: {df['total_wasted'].sum():,.0f} kg")


def generate_markdown_report(df, failed_instances):
    """Generate a comprehensive markdown report."""
    report_filename = 'analysis_report.md'
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("# Food Donation Optimization - Multi-Instance Analysis Report\n\n")
        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Instances Analyzed:** {len(df) + len(failed_instances)}\n")
        f.write(f"- **Successfully Optimized:** {len(df)}\n")
        f.write(f"- **Failed Instances:** {len(failed_instances)}\n\n")
        
        if len(df) > 0:
            f.write(f"- **Total Food Donated:** {df['total_donated'].sum():,.0f} kg\n")
            f.write(f"- **Total Food Wasted:** {df['total_wasted'].sum():,.0f} kg\n")
            f.write(f"- **Average Donation Rate:** {df['donation_rate'].mean():.2f}%\n")
            f.write(f"- **Average Waste Rate:** {df['waste_rate'].mean():.2f}%\n")
            f.write(f"- **Total Net Benefit:** ${df['objective_value'].sum():,.2f}\n\n")
        
        # Monthly Results Table
        f.write("## Monthly Results\n\n")
        
        if len(df) > 0:
            # Create a clean table
            table_df = df[['month', 'objective_value', 'total_donated', 'total_wasted', 
                          'donation_rate', 'waste_rate', 'solve_time']].copy()
            
            table_df.columns = ['Month', 'Objective Value ($)', 'Donated (kg)', 
                               'Wasted (kg)', 'Donation Rate (%)', 'Waste Rate (%)', 
                               'Solve Time (s)']
            
            f.write(table_df.to_markdown(index=False, floatfmt='.2f'))
            f.write("\n\n")
        
        # Statistical Analysis
        f.write("## Statistical Analysis\n\n")
        
        if len(df) > 0:
            f.write("### Key Performance Metrics\n\n")
            
            metrics = {
                'Objective Value': df['objective_value'],
                'Donation Rate (%)': df['donation_rate'],
                'Waste Rate (%)': df['waste_rate'],
                'Total Donated (kg)': df['total_donated'],
                'Solve Time (s)': df['solve_time']
            }
            
            f.write("| Metric | Mean | Std Dev | Min | Max |\n")
            f.write("|--------|------|---------|-----|-----|\n")
            
            for metric_name, metric_data in metrics.items():
                f.write(f"| {metric_name} | {metric_data.mean():.2f} | "
                       f"{metric_data.std():.2f} | {metric_data.min():.2f} | "
                       f"{metric_data.max():.2f} |\n")
            
            f.write("\n")
        
        # Insights and Recommendations
        f.write("## Insights and Recommendations\n\n")
        
        if len(df) > 0:
            # Best performing month
            best_idx = df['objective_value'].idxmax()
            best_month = df.loc[best_idx]
            
            f.write(f"### Best Performing Month: {best_month['month']}\n\n")
            f.write(f"- Objective Value: ${best_month['objective_value']:,.2f}\n")
            f.write(f"- Donation Rate: {best_month['donation_rate']:.2f}%\n")
            f.write(f"- Waste Rate: {best_month['waste_rate']:.2f}%\n")
            f.write(f"- Total Donated: {best_month['total_donated']:,.0f} kg\n\n")
            
            # Worst performing month
            worst_idx = df['objective_value'].idxmin()
            worst_month = df.loc[worst_idx]
            
            f.write(f"### Worst Performing Month: {worst_month['month']}\n\n")
            f.write(f"- Objective Value: ${worst_month['objective_value']:,.2f}\n")
            f.write(f"- Donation Rate: {worst_month['donation_rate']:.2f}%\n")
            f.write(f"- Waste Rate: {worst_month['waste_rate']:.2f}%\n")
            f.write(f"- Total Donated: {worst_month['total_donated']:,.0f} kg\n\n")
            
            # Recommendations
            f.write("### Recommendations\n\n")
            
            avg_waste = df['waste_rate'].mean()
            if avg_waste > 10:
                f.write(f"1. **High Waste Alert:** Average waste rate is {avg_waste:.2f}%. "
                       "Consider increasing transportation or food bank capacity.\n")
            
            avg_outsource = df['outsource_rate'].mean()
            if avg_outsource > 20:
                f.write(f"2. **Outsourcing Analysis:** {avg_outsource:.2f}% of donations use outsourcing. "
                       "Consider expanding fleet capacity for cost reduction.\n")
            
            if df['solve_time'].max() > 300:
                f.write(f"3. **Computational Performance:** Some instances take long to solve (max: {df['solve_time'].max():.1f}s). "
                       "Consider model simplification for real-time use.\n")
        
        # Failed Instances
        if failed_instances:
            f.write("\n## Failed Instances\n\n")
            f.write("| Instance | Error |\n")
            f.write("|----------|-------|\n")
            for fail in failed_instances:
                f.write(f"| {fail['instance']} | {fail['error']} |\n")
        
        f.write("\n---\n\n")
        f.write("*Report generated by Food Donation Optimization System*\n")
    
    print(f"✓ Comprehensive report saved to: {report_filename}")


if __name__ == "__main__":
    results, failures = run_multi_instance_analysis()


# %%
"""
Statistical Analysis and Visualization Script
Performs deeper statistical analysis on multi-instance results
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime


def load_results(csv_file='multi_instance_results.csv'):
    """Load results from CSV file."""
    try:
        df = pd.DataFrame(pd.read_csv(csv_file))
        print(f"✓ Loaded {len(df)} results from {csv_file}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        print("Please run the multi-instance analysis first.")
        return None


def correlation_analysis(df):
    """Perform correlation analysis on key metrics."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80 + "\n")
    
    # Select numeric columns for correlation
    numeric_cols = [
        'objective_value', 'total_donated', 'total_wasted', 'total_outsourced',
        'donation_rate', 'waste_rate', 'outsource_rate',
        'own_fleet_costs', 'outsourcing_costs', 'waste_penalty',
        'num_direct_req', 'num_indirect_req', 'solve_time'
    ]
    
    # Filter only existing columns
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print("Insufficient numeric columns for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[available_cols].corr()
    
    print("Correlation Matrix (showing strong correlations > 0.7 or < -0.7):\n")
    
    # Find strong correlations
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corrs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if strong_corrs:
        for corr in sorted(strong_corrs, key=lambda x: abs(x['correlation']), reverse=True):
            direction = "positive" if corr['correlation'] > 0 else "negative"
            print(f"  {corr['var1']} <-> {corr['var2']}")
            print(f"    Correlation: {corr['correlation']:.3f} ({direction})\n")
    else:
        print("  No strong correlations found (|r| > 0.7)\n")
    
    # Save full correlation matrix
    corr_matrix.to_csv('correlation_matrix.csv')
    print("✓ Full correlation matrix saved to: correlation_matrix.csv")


def trend_analysis(df):
    """Analyze trends across months."""
    print("\n" + "="*80)
    print("TREND ANALYSIS")
    print("="*80 + "\n")
    
    if 'month' not in df.columns:
        print("Month column not found - cannot perform trend analysis")
        return
    
    # Sort by month
    df_sorted = df.sort_values('month').copy()
    
    # Analyze key metrics trends
    metrics = ['objective_value', 'donation_rate', 'waste_rate', 'total_donated']
    
    print("Month-over-Month Changes:\n")
    
    for metric in metrics:
        if metric not in df_sorted.columns:
            continue
        
        values = df_sorted[metric].values
        
        if len(values) < 2:
            continue
        
        # Calculate changes
        changes = np.diff(values)
        pct_changes = (changes / values[:-1]) * 100
        
        avg_change = np.mean(changes)
        avg_pct_change = np.mean(pct_changes)
        
        trend = "increasing" if avg_change > 0 else "decreasing"
        
        print(f"{metric}:")
        print(f"  Average change: {avg_change:.2f} ({avg_pct_change:+.2f}%)")
        print(f"  Trend: {trend}")
        print(f"  Min value: {values.min():.2f} (month: {df_sorted.iloc[values.argmin()]['month']})")
        print(f"  Max value: {values.max():.2f} (month: {df_sorted.iloc[values.argmax()]['month']})")
        print()


def efficiency_analysis(df):
    """Analyze operational efficiency metrics."""
    print("\n" + "="*80)
    print("EFFICIENCY ANALYSIS")
    print("="*80 + "\n")
    
    # Calculate efficiency metrics
    if all(col in df.columns for col in ['total_donated', 'total_available']):
        df['efficiency'] = (df['total_donated'] / df['total_available']) * 100
        
        print("Donation Efficiency:")
        print(f"  Average: {df['efficiency'].mean():.2f}%")
        print(f"  Best month: {df.loc[df['efficiency'].idxmax(), 'month']} ({df['efficiency'].max():.2f}%)")
        print(f"  Worst month: {df.loc[df['efficiency'].idxmin(), 'month']} ({df['efficiency'].min():.2f}%)")
        print()
    
    # Cost per kg donated
    if all(col in df.columns for col in ['own_fleet_costs', 'outsourcing_costs', 'total_donated']):
        df['cost_per_kg'] = (df['own_fleet_costs'] + df['outsourcing_costs']) / df['total_donated']
        
        print("Cost Efficiency (Cost per kg donated):")
        print(f"  Average: ${df['cost_per_kg'].mean():.2f}/kg")
        print(f"  Most efficient: {df.loc[df['cost_per_kg'].idxmin(), 'month']} (${df['cost_per_kg'].min():.2f}/kg)")
        print(f"  Least efficient: {df.loc[df['cost_per_kg'].idxmax(), 'month']} (${df['cost_per_kg'].max():.2f}/kg)")
        print()
    
    # Fleet utilization
    if all(col in df.columns for col in ['total_direct_fleet', 'total_indirect_fleet', 'total_donated']):
        df['fleet_usage_rate'] = ((df['total_direct_fleet'] + df['total_indirect_fleet']) / df['total_donated']) * 100
        
        print("Fleet vs Outsourcing:")
        print(f"  Average fleet usage: {df['fleet_usage_rate'].mean():.2f}%")
        print(f"  Average outsourcing: {df['outsource_rate'].mean():.2f}%")
        print()


def generate_insights_report(df):
    """Generate insights and recommendations based on data analysis."""
    print("\n" + "="*80)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("="*80 + "\n")
    
    insights = []
    recommendations = []
    
    # Check waste rate
    if 'waste_rate' in df.columns:
        avg_waste = df['waste_rate'].mean()
        max_waste = df['waste_rate'].max()
        
        if avg_waste > 15:
            insights.append(f"High average waste rate: {avg_waste:.2f}%")
            recommendations.append(
                "Consider increasing transportation capacity or extending food bank receiving hours"
            )
        
        if max_waste > 25:
            worst_month = df.loc[df['waste_rate'].idxmax(), 'month']
            insights.append(f"Critical waste in {worst_month}: {max_waste:.2f}%")
            recommendations.append(
                f"Investigate specific constraints in {worst_month} - "
                "may need temporary capacity expansion"
            )
    
    # Check outsourcing
    if 'outsource_rate' in df.columns:
        avg_outsource = df['outsource_rate'].mean()
        
        if avg_outsource > 30:
            insights.append(f"High outsourcing rate: {avg_outsource:.2f}%")
            recommendations.append(
                "Heavy reliance on outsourcing. ROI analysis recommended for fleet expansion"
            )
    
    # Check computational performance
    if 'solve_time' in df.columns:
        avg_time = df['solve_time'].mean()
        max_time = df['solve_time'].max()
        
        if avg_time > 180:
            insights.append(f"Long average solve time: {avg_time:.1f}s")
            recommendations.append(
                "Model complexity may be high. Consider heuristic approaches for real-time decisions"
            )
    
    # Check variability
    if 'objective_value' in df.columns:
        cv = (df['objective_value'].std() / df['objective_value'].mean()) * 100
        
        if cv > 20:
            insights.append(f"High month-to-month variability: CV = {cv:.1f}%")
            recommendations.append(
                "Significant seasonal effects detected. Develop month-specific strategies"
            )
    
    # Print insights
    if insights:
        print("INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")
        print()
    
    if recommendations:
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        print()
    
    # Save to file
    report = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'insights': insights,
        'recommendations': recommendations
    }
    
    with open('insights_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("✓ Insights report saved to: insights_report.json")


def compare_scenarios(df):
    """Compare different operational scenarios."""
    print("\n" + "="*80)
    print("SCENARIO COMPARISON")
    print("="*80 + "\n")
    
    if len(df) < 2:
        print("Insufficient data for scenario comparison")
        return
    
    # Group by characteristics
    if 'num_direct_req' in df.columns and 'num_indirect_req' in df.columns:
        df['total_req'] = df['num_direct_req'] + df['num_indirect_req']
        df['indirect_ratio'] = df['num_indirect_req'] / df['total_req']
        
        # Split into high/low indirect requirement months
        median_indirect = df['indirect_ratio'].median()
        
        high_indirect = df[df['indirect_ratio'] >= median_indirect]
        low_indirect = df[df['indirect_ratio'] < median_indirect]
        
        print("Impact of Indirect Requirements:\n")
        print(f"Months with HIGH indirect requirements (>= {median_indirect:.2%}):")
        print(f"  Count: {len(high_indirect)}")
        print(f"  Avg objective value: ${high_indirect['objective_value'].mean():,.2f}")
        print(f"  Avg donation rate: {high_indirect['donation_rate'].mean():.2f}%")
        print(f"  Avg waste rate: {high_indirect['waste_rate'].mean():.2f}%")
        print()
        
        print(f"Months with LOW indirect requirements (< {median_indirect:.2%}):")
        print(f"  Count: {len(low_indirect)}")
        print(f"  Avg objective value: ${low_indirect['objective_value'].mean():,.2f}")
        print(f"  Avg donation rate: {low_indirect['donation_rate'].mean():.2f}%")
        print(f"  Avg waste rate: {low_indirect['waste_rate'].mean():.2f}%")
        print()


def export_summary_for_presentation(df):
    """Export a clean summary table for presentations."""
    print("\n" + "="*80)
    print("EXPORTING PRESENTATION SUMMARY")
    print("="*80 + "\n")
    
    # Create summary table
    summary_cols = [
        'month', 'objective_value', 'total_donated', 'total_wasted',
        'donation_rate', 'waste_rate', 'outsource_rate'
    ]
    
    available_cols = [col for col in summary_cols if col in df.columns]
    
    if not available_cols:
        print("No data available for summary export")
        return
    
    summary_df = df[available_cols].copy()
    
    # Rename columns for clarity
    column_names = {
        'month': 'Month',
        'objective_value': 'Net Benefit ($)',
        'total_donated': 'Donated (kg)',
        'total_wasted': 'Wasted (kg)',
        'donation_rate': 'Donation Rate (%)',
        'waste_rate': 'Waste Rate (%)',
        'outsource_rate': 'Outsource Rate (%)'
    }
    
    summary_df = summary_df.rename(columns=column_names)
    
    # Round values
    for col in summary_df.columns:
        if col != 'Month' and col in summary_df.columns:
            summary_df[col] = summary_df[col].round(2)
    
    # Export
    summary_df.to_csv('presentation_summary.csv', index=False)
    summary_df.to_excel('presentation_summary.xlsx', index=False, engine='openpyxl')
    
    print("✓ Summary exported to:")
    print("  - presentation_summary.csv")
    print("  - presentation_summary.xlsx")
    
    # Create totals row
    if len(summary_df) > 0:
        print("\nOverall Totals:")
        if 'Net Benefit ($)' in summary_df.columns:
            print(f"  Total Net Benefit: ${summary_df['Net Benefit ($)'].sum():,.2f}")
        if 'Donated (kg)' in summary_df.columns:
            print(f"  Total Donated: {summary_df['Donated (kg)'].sum():,.0f} kg")
        if 'Wasted (kg)' in summary_df.columns:
            print(f"  Total Wasted: {summary_df['Wasted (kg)'].sum():,.0f} kg")


def main():
    """Main analysis function."""
    print("="*80)
    print("STATISTICAL ANALYSIS OF MULTI-INSTANCE RESULTS")
    print("="*80)
    
    # Load results
    df = load_results()
    
    if df is None or len(df) == 0:
        print("\n❌ No data available for analysis")
        return
    
    print(f"\nAnalyzing {len(df)} monthly instances...")
    
    try:
        # Perform analyses
        correlation_analysis(df)
        trend_analysis(df)
        efficiency_analysis(df)
        compare_scenarios(df)
        generate_insights_report(df)
        export_summary_for_presentation(df)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        print("  - correlation_matrix.csv")
        print("  - insights_report.json")
        print("  - presentation_summary.csv")
        print("  - presentation_summary.xlsx")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



