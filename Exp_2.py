# %% [markdown]
# # 🧪 Experiment 2: Route Mix and Fleet Capacity Optimization
# 
# Full 3² Factorial Design with 3 replicas (27 total runs)

# %% [markdown]
# ## 1️⃣ Import

# %%
# %%
import gurobipy as gp
from gurobipy import GRB
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time
from copy import deepcopy
import warnings
import os

warnings.filterwarnings('ignore')

# Import the model class (assuming it is in the same directory)
from model_final import FoodDonationOptimizer

print("✓ Libraries imported")

# %% [markdown]
# ## 2️⃣ Experimental Design Configuration

# %%
class ExperimentDesign:
    """
    Factorial Design 3² for Food Donation Optimization
    
    Factors:
    - X1: Fleet capacity (5,000 / 10,000 / 15,000 kg)
    - X2: % Indirect requirements (20% / 50% / 80%)
    
    Response Variables:
    - Y1: Net benefit (USD)
    - Y2: Fulfillment rate (%)
    - Y3: Outsourcing cost (USD)
    - Y4: Waste rate (%)
    - Y5: Days with fleet saturation (days)
    """
    
    def __init__(self, base_instance_file):
        """
        Initialize experimental design.
        
        Args:
            base_instance_file: Path to base JSON instance file
        """
        self.base_instance_file = base_instance_file
        
        # Load base instance
        with open(base_instance_file, 'r', encoding='utf-8') as f:
            self.base_data = json.load(f)
        
        # Define factor levels
        self.factor_levels = {
            'fleet_capacity': [5000, 10000, 15000],
            'indirect_percentage': [20, 50, 80]
        }
        
        # Generate experimental matrix
        self.n_replicas = 3
        self.experimental_runs = self._generate_experimental_matrix()
        
        # Storage for results
        self.results = []
        
        print(f"✓ Experimental design initialized")
        print(f"  - Total runs: {len(self.experimental_runs)}")
        print(f"  - Factors: 2")
        print(f"  - Levels per factor: 3")
        print(f"  - Replicas: {self.n_replicas}")
    
    def _generate_experimental_matrix(self):
        """Generate the full factorial design with replicas."""
        runs = []
        run_id = 1
        
        for replica in range(1, self.n_replicas + 1):
            for fleet_cap in self.factor_levels['fleet_capacity']:
                for indirect_pct in self.factor_levels['indirect_percentage']:
                    runs.append({
                        'run_id': run_id,
                        'replica': replica,
                        'fleet_capacity': fleet_cap,
                        'indirect_percentage': indirect_pct,
                        'fleet_level': self._encode_level(fleet_cap, 'fleet_capacity'),
                        'indirect_level': self._encode_level(indirect_pct, 'indirect_percentage')
                    })
                    run_id += 1
        
        return runs
    
    def _encode_level(self, value, factor):
        """Encode factor level as -1, 0, +1."""
        levels = self.factor_levels[factor]
        if value == levels[0]:
            return -1
        elif value == levels[1]:
            return 0
        else:
            return 1
    
    def _ensure_required_keys(self, data):
        """
        Ensure all requirements have 'distance' (for direct) and 'dist_indirect' (for indirect).
        If missing, calculate from the other or set default values.
        """
        for req in data['requirements']:
            origin = req['origin_type']
            
            # For direct: must have 'distance'
            if origin in ['Manufacturing', 'DistributionCenter']:
                if 'distance' not in req:
                    if 'dist_indirect' in req:
                        req['distance'] = req['dist_indirect'] / 1.4
                    else:
                        req['distance'] = 100.0  # default value (adjustable)
                # If it has dist_indirect, remove it to avoid confusion
                if 'dist_indirect' in req:
                    del req['dist_indirect']
            
            # For indirect (Client): must have 'dist_indirect'
            elif origin == 'Client':
                if 'dist_indirect' not in req:
                    if 'distance' in req:
                        req['dist_indirect'] = req['distance'] * 1.4
                    else:
                        req['dist_indirect'] = 140.0  # default value
                # Ensure they also have 'distance' (just in case)
                if 'distance' not in req:
                    req['distance'] = req['dist_indirect'] / 1.4
        
        return data
    
    def _create_instance_for_run(self, run_config):
        """
        Create a modified instance for a specific experimental run.
        
        Args:
            run_config: Dictionary with run configuration
            
        Returns:
            Modified data dictionary with all required keys
        """
        # Deep copy base data
        data = deepcopy(self.base_data)
        
        # 1. Modify fleet capacity
        fleet_capacity = run_config['fleet_capacity']
        data['parameters']['transport_capacity_per_period'] = fleet_capacity
        
        # 2. Reclassify requirements according to indirect percentage
        indirect_pct = run_config['indirect_percentage']
        data = self._reclassify_requirements(data, indirect_pct, run_config['replica'])
        
        # 3. Ensure all required keys exist
        data = self._ensure_required_keys(data)
        
        return data
    
    def _reclassify_requirements(self, data, indirect_pct, seed):
        """
        Reclassify Client requirements as direct or indirect based on target percentage.
        
        IMPORTANT: This method MODIFIES 'origin_type' and ensures proper keys.
        """
        np.random.seed(seed * 1000 + indirect_pct)
        requirements = data['requirements']
        
        # First, ensure all Clients have 'distance' (to use later)
        for req in requirements:
            if req['origin_type'] == 'Client' and 'distance' not in req:
                if 'dist_indirect' in req:
                    req['distance'] = req['dist_indirect'] / 1.4
                else:
                    req['distance'] = 100.0  # default
        
        # Select clients
        clients = [r for r in requirements if r['origin_type'] == 'Client']
        n_total = len(clients)
        n_indirect = int(n_total * indirect_pct / 100)
        
        # Randomly choose which ones will be indirect
        np.random.shuffle(clients)
        indirect_ids = {req['id'] for req in clients[:n_indirect]}
        
        # Reclassify
        for req in requirements:
            if req['origin_type'] == 'Client':
                if req['id'] in indirect_ids:
                    # Keep as indirect: ensure dist_indirect
                    if 'dist_indirect' not in req:
                        req['dist_indirect'] = req['distance'] * 1.4
                else:
                    # Convert to direct
                    req['origin_type'] = 'DistributionCenter'  # or 'Manufacturing'
                    # Ensure it has 'distance' (it should already)
                    if 'distance' not in req:
                        req['distance'] = req.get('dist_indirect', 100.0) / 1.4
                    # Remove dist_indirect if it exists
                    if 'dist_indirect' in req:
                        del req['dist_indirect']
        
        return data
    
    def _calculate_response_variables(self, optimizer, run_config):
        """
        Calculate all response variables from optimized model.
        """
        if optimizer.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        
        if optimizer.model.SolCount == 0:
            return None
        
        metrics = optimizer._calculate_metrics()
        
        Y1 = optimizer.model.ObjVal
        Y2 = metrics['donation_rate']
        Y3 = metrics['outsourcing_costs']
        Y4 = (metrics['total_wasted'] / metrics['total_available'] * 100) if metrics['total_available'] > 0 else 0
        
        # Y5: Days with Fleet Saturation
        Y5 = 0
        fleet_capacity = run_config['fleet_capacity']
        
        for t in optimizer.T:
            fleet_usage = 0
            
            # Direct deliveries
            for r in optimizer.R_direct:
                req = optimizer._get_requirement_by_id(r)
                l_r = req['release_date']
                e_r = req['expiration_date']
                if l_r <= t <= e_r and (r, t) in optimizer.y_deliv:
                    fleet_usage += optimizer.y_deliv[r, t].X
            
            # Indirect pickups
            for r in optimizer.R_indirect:
                req = optimizer._get_requirement_by_id(r)
                l_r = req['release_date']
                e_r = req['expiration_date']
                if l_r <= t < e_r and (r, t) in optimizer.y_pickup:
                    fleet_usage += optimizer.y_pickup[r, t].X
            
            # Indirect deliveries
            for r in optimizer.R_indirect:
                req = optimizer._get_requirement_by_id(r)
                l_r = req['release_date']
                e_r = req['expiration_date']
                if l_r + 1 <= t <= e_r and (r, t) in optimizer.y_deliv:
                    fleet_usage += optimizer.y_deliv[r, t].X
            
            if fleet_usage >= 0.95 * fleet_capacity:
                Y5 += 1
        
        return {
            'Y1_net_benefit': Y1,
            'Y2_fulfillment_rate': Y2,
            'Y3_outsourcing_cost': Y3,
            'Y4_waste_rate': Y4,
            'Y5_saturation_days': Y5,
            'total_donated': metrics['total_donated'],
            'total_wasted': metrics['total_wasted'],
            'total_available': metrics['total_available']
        }
    
    def run_experiment(self, output_file='experiment2_results.csv'):
        """
        Execute all experimental runs.
        """
        print("\n" + "="*80)
        print("EXECUTING EXPERIMENT 2: FLEET CAPACITY & ROUTING STRATEGY")
        print("="*80)
        
        total_runs = len(self.experimental_runs)
        temp_files = []  # for cleanup later
        
        for idx, run_config in enumerate(self.experimental_runs, 1):
            print(f"\n{'='*80}")
            print(f"RUN {idx}/{total_runs}")
            print(f"  Fleet Capacity: {run_config['fleet_capacity']:,} kg")
            print(f"  Indirect %: {run_config['indirect_percentage']}%")
            print(f"  Replica: {run_config['replica']}")
            print(f"{'='*80}")
            
            start_time = time.time()
            temp_file = f"temp_instance_run_{run_config['run_id']}.json"
            temp_files.append(temp_file)
            
            try:
                # Create instance
                instance_data = self._create_instance_for_run(run_config)
                
                # Save temporary file
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(instance_data, f, indent=2)
                
                # Solve with Gurobi
                with gp.Env(empty=True) as env:
                    env.setParam('OutputFlag', 0)
                    env.setParam('TimeLimit', 300)
                    env.setParam('MIPFocus', 1)
                    env.setParam('Threads', 4)
                    env.start()
                    
                    with FoodDonationOptimizer(env) as optimizer:
                        optimizer.load_data_from_json(temp_file)
                        optimizer.build_model()
                        status = optimizer.optimize()
                        
                        if status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and optimizer.model.SolCount > 0:
                            responses = self._calculate_response_variables(optimizer, run_config)
                            if responses:
                                result = {
                                    **run_config,
                                    **responses,
                                    'solve_time': time.time() - start_time,
                                    'status': 'OPTIMAL' if status == GRB.OPTIMAL else 'TIME_LIMIT',
                                    'gap': optimizer.model.MIPGap if status == GRB.TIME_LIMIT else 0
                                }
                                self.results.append(result)
                                
                                print(f"\n✓ Run completed successfully")
                                print(f"  Y1 (Net Benefit): ${responses['Y1_net_benefit']:,.2f}")
                                print(f"  Y2 (Fulfillment): {responses['Y2_fulfillment_rate']:.2f}%")
                                print(f"  Y3 (Outsourcing): ${responses['Y3_outsourcing_cost']:,.2f}")
                                print(f"  Y4 (Waste Rate): {responses['Y4_waste_rate']:.2f}%")
                                print(f"  Y5 (Saturation): {responses['Y5_saturation_days']} days")
                                print(f"  Solve time: {result['solve_time']:.2f}s")
                            else:
                                print(f"\n⚠️ Could not calculate response variables")
                        else:
                            print(f"\n⚠️ No solution found (Status: {status})")
                
            except Exception as e:
                print(f"\n❌ Error in run {idx}: {e}")
                import traceback
                traceback.print_exc()
                # Optional: save the problematic JSON for debugging
                # with open(f"error_run_{run_config['run_id']}.json", 'w') as f:
                #     json.dump(instance_data, f, indent=2)
        
        # Clean up temporary files
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        
        # Save results
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(output_file, index=False)
            print(f"\n✓ Results saved to '{output_file}'")
            print(f"  Total successful runs: {len(self.results)}/{total_runs}")
        else:
            print(f"\n⚠️ No results to save")
    
    def analyze_results(self, results_file='experiment2_results.csv'):
        """
        Perform statistical analysis and generate visualizations.
        """
        # Load results
        df = pd.read_csv(results_file)
        
        print("\n" + "="*80)
        print("EXPERIMENTAL ANALYSIS")
        print("="*80)
        
        self._analyze_main_effects(df)
        self._analyze_interactions(df)
        self._compare_scenarios(df)
    
    def _analyze_main_effects(self, df):
        """Analyze and plot main effects."""
        print("\n" + "-"*80)
        print("1. MAIN EFFECTS ANALYSIS")
        print("-"*80)
        
        response_vars = ['Y1_net_benefit', 'Y2_fulfillment_rate', 'Y3_outsourcing_cost', 
                        'Y4_waste_rate', 'Y5_saturation_days']
        
        for response in response_vars:
            print(f"\n{response}:")
            fleet_means = df.groupby('fleet_capacity')[response].mean()
            print(f"  Fleet Capacity Effect:")
            for level, value in fleet_means.items():
                print(f"    {level:,} kg: {value:.2f}")
            
            indirect_means = df.groupby('indirect_percentage')[response].mean()
            print(f"  Indirect % Effect:")
            for level, value in indirect_means.items():
                print(f"    {level}%: {value:.2f}")
        
        # Plots
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Main Effects Plot', fontsize=16, fontweight='bold')
        
        for idx, response in enumerate(response_vars):
            ax1 = axes[0, idx]
            fleet_means = df.groupby('fleet_capacity')[response].mean()
            ax1.plot(fleet_means.index, fleet_means.values, marker='o', linewidth=2, markersize=8)
            ax1.set_xlabel('Fleet Capacity (kg)')
            ax1.set_ylabel(response.replace('_', ' ').title())
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f'Fleet Effect on {response.split("_")[0]}')
            
            ax2 = axes[1, idx]
            indirect_means = df.groupby('indirect_percentage')[response].mean()
            ax2.plot(indirect_means.index, indirect_means.values, marker='s', linewidth=2, markersize=8, color='orange')
            ax2.set_xlabel('Indirect Routes (%)')
            ax2.set_ylabel(response.replace('_', ' ').title())
            ax2.grid(True, alpha=0.3)
            ax2.set_title(f'Routing Effect on {response.split("_")[0]}')
        
        plt.tight_layout()
        plt.savefig('experiment2_main_effects.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Main effects plot saved to 'experiment2_main_effects.png'")
        plt.show()
    
    def _analyze_interactions(self, df):
        """Analyze and plot interaction effects."""
        print("\n" + "-"*80)
        print("2. INTERACTION ANALYSIS")
        print("-"*80)
        
        response_vars = ['Y1_net_benefit', 'Y4_waste_rate', 'Y3_outsourcing_cost']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Interaction Effects: Fleet Capacity × Indirect %', fontsize=16, fontweight='bold')
        
        for idx, response in enumerate(response_vars):
            ax = axes[idx]
            interaction_data = df.groupby(['fleet_capacity', 'indirect_percentage'])[response].mean().unstack()
            for col in interaction_data.columns:
                ax.plot(interaction_data.index, interaction_data[col], 
                       marker='o', linewidth=2, markersize=8, label=f'{col}% Indirect')
            ax.set_xlabel('Fleet Capacity (kg)', fontsize=11)
            ax.set_ylabel(response.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'Interaction on {response.split("_")[0]}', fontsize=12, fontweight='bold')
            ax.legend(title='Indirect Routes', loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiment2_interactions.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Interaction plot saved to 'experiment2_interactions.png'")
        plt.show()
        
        print("\nInteraction Assessment:")
        print("  (Non-parallel lines indicate interaction effects)")
    
    def _compare_scenarios(self, df):
        """Compare scenarios and identify optimal configuration."""
        print("\n" + "-"*80)
        print("3. SCENARIO COMPARISON")
        print("-"*80)
        
        config_summary = df.groupby(['fleet_capacity', 'indirect_percentage']).agg({
            'Y1_net_benefit': ['mean', 'std'],
            'Y2_fulfillment_rate': ['mean', 'std'],
            'Y3_outsourcing_cost': ['mean', 'std'],
            'Y4_waste_rate': ['mean', 'std'],
            'Y5_saturation_days': ['mean', 'std']
        }).round(2)
        
        config_summary.columns = ['_'.join(col).strip() for col in config_summary.columns.values]
        config_summary = config_summary.reset_index()
        config_summary['rank'] = config_summary['Y1_net_benefit_mean'].rank(ascending=False)
        config_summary = config_summary.sort_values('rank')
        
        print("\nConfiguration Ranking (by Net Benefit):")
        print(config_summary[['fleet_capacity', 'indirect_percentage', 'Y1_net_benefit_mean', 
                              'Y2_fulfillment_rate_mean', 'Y4_waste_rate_mean', 'rank']].to_string(index=False))
        
        best_config = config_summary.iloc[0]
        print(f"\n{'='*80}")
        print("OPTIMAL CONFIGURATION:")
        print(f"  Fleet Capacity: {best_config['fleet_capacity']:,.0f} kg")
        print(f"  Indirect Routes: {best_config['indirect_percentage']:.0f}%")
        print(f"  Net Benefit: ${best_config['Y1_net_benefit_mean']:,.2f} ± ${best_config['Y1_net_benefit_std']:,.2f}")
        print(f"  Fulfillment Rate: {best_config['Y2_fulfillment_rate_mean']:.2f}% ± {best_config['Y2_fulfillment_rate_std']:.2f}%")
        print(f"  Waste Rate: {best_config['Y4_waste_rate_mean']:.2f}% ± {best_config['Y4_waste_rate_std']:.2f}%")
        print(f"{'='*80}")
        
        # 3D and contour plots
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        X = config_summary['fleet_capacity'].values
        Y = config_summary['indirect_percentage'].values
        Z = config_summary['Y1_net_benefit_mean'].values
        ax1.scatter(X, Y, Z, c=Z, cmap='viridis', s=200, edgecolors='black', linewidth=1.5)
        ax1.set_xlabel('Fleet Capacity (kg)', fontsize=10)
        ax1.set_ylabel('Indirect Routes (%)', fontsize=10)
        ax1.set_zlabel('Net Benefit (USD)', fontsize=10)
        ax1.set_title('3D Response Surface: Net Benefit', fontsize=12, fontweight='bold')
        
        ax2 = fig.add_subplot(122)
        fleet_levels = sorted(config_summary['fleet_capacity'].unique())
        indirect_levels = sorted(config_summary['indirect_percentage'].unique())
        Z_matrix = config_summary.pivot(index='indirect_percentage', 
                                        columns='fleet_capacity', 
                                        values='Y1_net_benefit_mean').values
        X_grid, Y_grid = np.meshgrid(fleet_levels, indirect_levels)
        contour = ax2.contourf(X_grid, Y_grid, Z_matrix, levels=10, cmap='viridis')
        ax2.scatter(config_summary['fleet_capacity'], 
                   config_summary['indirect_percentage'], 
                   c='red', s=100, edgecolors='white', linewidth=2, zorder=5)
        ax2.scatter(best_config['fleet_capacity'], 
                   best_config['indirect_percentage'],
                   c='lime', s=300, marker='*', edgecolors='black', linewidth=2, zorder=6,
                   label='Optimal')
        ax2.set_xlabel('Fleet Capacity (kg)', fontsize=11)
        ax2.set_ylabel('Indirect Routes (%)', fontsize=11)
        ax2.set_title('Contour Plot: Net Benefit', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        plt.colorbar(contour, ax=ax2, label='Net Benefit (USD)')
        plt.tight_layout()
        plt.savefig('experiment2_response_surface.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Response surface plot saved to 'experiment2_response_surface.png'")
        plt.show()
        
        print("\nTrade-off Analysis (Top 3 Configurations):")
        top3 = config_summary.head(3)
        for idx, row in top3.iterrows():
            print(f"\n  Rank #{int(row['rank'])}:")
            print(f"    Configuration: Fleet={row['fleet_capacity']:,.0f} kg, Indirect={row['indirect_percentage']:.0f}%")
            print(f"    Net Benefit: ${row['Y1_net_benefit_mean']:,.2f}")
            print(f"    Fulfillment: {row['Y2_fulfillment_rate_mean']:.2f}%")
            print(f"    Outsourcing Cost: ${row['Y3_outsourcing_cost_mean']:,.2f}")
            print(f"    Waste Rate: {row['Y4_waste_rate_mean']:.2f}%")
            print(f"    Saturation Days: {row['Y5_saturation_days_mean']:.1f}")


# %% [markdown]
# ## 3️⃣ Execute Experiment

# %%
def main():
    """Main execution function for Experiment 2."""
    
    print("="*80)
    print("EXPERIMENT 2: FLEET CAPACITY & ROUTING STRATEGY OPTIMIZATION")
    print("="*80)
    
    # Initialize experiment
    experiment = ExperimentDesign('instance_distance_exp.json')
    
    # Display experimental matrix
    print("\nExperimental Matrix:")
    df_matrix = pd.DataFrame(experiment.experimental_runs)
    print(df_matrix.head(10))
    
    # Run all experiments
    experiment.run_experiment(output_file='experiment2_results.csv')
    
    # Analyze results
    if experiment.results:
        experiment.analyze_results(results_file='experiment2_results.csv')
    else:
        print("\n⚠️ No results to analyze")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## 4️⃣ Optional: Load and Re-analyze Existing Results

# %%
# If you already have results and just want to re-run the analysis:
#
# experiment = ExperimentDesign('instance_distance_exp.json')
# experiment.analyze_results('experiment2_results.csv')

print("\n✓ Experiment 2 code ready to execute")
