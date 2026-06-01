# %%
"""
Sensitivity Analysis for Food Donation Optimization Model
Analyzes the impact of parameter variations on the objective function
"""

import gurobipy as gp
from gurobipy import GRB
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
from copy import deepcopy


# Import the optimizer class from your model


# AFTER (Replace 'your_model_file' with the actual filename without .py):
from model_final import FoodDonationOptimizer


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on the food donation optimization model.
    Analyzes 6 key parameters and their impact on the objective function.
    """
    
    def __init__(self, base_data_file):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            base_data_file: Path to the base instance JSON file
        """
        self.base_data_file = base_data_file
        self.base_data = None
        self.base_objective = None
        
        # Define parameters to analyze
        self.parameters = {
            'beta': {'name': 'Incentive (β)', 'unit': 'Dimensionless', 'path': ['parameters', 'beta_1']},
            'pi': {'name': 'Penalty (π)', 'unit': '$/kg', 'path': ['parameters', 'pi_penalty']},
            'c_trans': {'name': 'Transportation Cost (c_trans)', 'unit': '$/km·kg', 'path': ['parameters', 'c_trans']},
            'alpha': {'name': 'Outsourcing (α)', 'unit': 'Dimensionless', 'path': ['parameters', 'alpha_outsource']},
            'cap_bank': {'name': 'Bank Capacity (cap_ban)', 'unit': 'kg/day', 'path': ['parameters', 'food_bank_capacity']},
            'cap_trans': {'name': 'Transportation Capacity (cap_trans)', 'unit': 'kg', 'path': ['parameters', 'transport_capacity_per_period']}
        }
        
        # Variation levels (multipliers)
        self.variation_levels = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
        
        # Results storage
        self.results = {param: {} for param in self.parameters.keys()}
        self.elasticities = {param: {} for param in self.parameters.keys()}
        self.regression_results = {}
        
    def load_base_data(self):
        """Load base instance data."""
        print("\n" + "="*80)
        print("LOADING BASE INSTANCE")
        print("="*80)
        
        with open(self.base_data_file, 'r', encoding='utf-8') as f:
            self.base_data = json.load(f)
        
        print(f"✓ Base data loaded from {self.base_data_file}")
        
    def solve_base_case(self):
        """Solve the base case to get reference objective value."""
        print("\n" + "="*80)
        print("STEP 1: SOLVING BASE CASE")
        print("="*80)
        
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)  # Suppress output
            env.setParam('TimeLimit', 300)
            env.start()
            
            with FoodDonationOptimizer(env) as optimizer:
                # Save base data to temporary file
                with open('temp_base.json', 'w') as f:
                    json.dump(self.base_data, f)
                
                optimizer.load_data_from_json('temp_base.json')
                optimizer.build_model()
                status = optimizer.optimize()
                
                if status == GRB.OPTIMAL:
                    self.base_objective = optimizer.model.ObjVal
                    print(f"\n✓ Base case solved successfully")
                    print(f"  Base objective value (Z₀): ${self.base_objective:,.2f}")
                else:
                    raise Exception(f"Base case optimization failed with status {status}")
    
    def get_parameter_value(self, data, param_key):
        """Extract parameter value from data dictionary."""
        path = self.parameters[param_key]['path']
        value = data
        for key in path:
            value = value[key]
        return value
    

    
    def set_parameter_value(self, data, param_key, new_value):
        """Establece el valor del parámetro en el diccionario de datos."""
        path = self.parameters[param_key]['path']
        
        # Navegar hasta el padre de la clave objetivo
        target = data
        for key in path[:-1]:
            target = target[key]
        
        # Asignar directamente el valor preparado (funciona para números y diccionarios)
        target[path[-1]] = new_value
    
    def solve_with_parameter(self, param_key, multiplier):
        """
        Solve model with modified parameter value.
        
        Args:
            param_key: Parameter to modify
            multiplier: Multiplier for the base value
            
        Returns:
            Objective value or None if infeasible
        """
        # Create modified data
        modified_data = deepcopy(self.base_data)
        
        # Get base value
        base_value = self.get_parameter_value(self.base_data, param_key)
        
        # Calculate new value
        if param_key == 'cap_bank':
            # For bank capacity, we need to scale each bank's capacity
            new_value = {}
            for bank, capacity in base_value.items():
                new_value[bank] = capacity * multiplier
        else:
            new_value = base_value * multiplier
        
        # Set new value
        self.set_parameter_value(modified_data, param_key, new_value)
        
        # Solve model
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('TimeLimit', 300)
            env.start()
            
            with FoodDonationOptimizer(env) as optimizer:
                # Save modified data
                with open('temp_modified.json', 'w') as f:
                    json.dump(modified_data, f)
                
                optimizer.load_data_from_json('temp_modified.json')
                optimizer.build_model()
                status = optimizer.optimize()
                
                if status == GRB.OPTIMAL:
                    return optimizer.model.ObjVal
                else:
                    print(f"  ⚠️ Warning: {param_key} with multiplier {multiplier} did not find optimal solution")
                    return None
    
    def run_univariate_analysis(self):
        """
        STEP 2: Run univariate sensitivity analysis for all parameters.
        Vary each parameter while keeping others constant.
        """
        print("\n" + "="*80)
        print("STEP 2: UNIVARIATE SENSITIVITY ANALYSIS")
        print("="*80)
        
        total_scenarios = len(self.parameters) * (len(self.variation_levels) - 1)  # -1 because base is already solved
        scenario_count = 0
        
        for param_key, param_info in self.parameters.items():
            print(f"\n--- Analyzing: {param_info['name']} ---")
            
            base_value = self.get_parameter_value(self.base_data, param_key)
            
            if param_key == 'cap_bank':
                # For display, show average capacity
                base_value_display = np.mean(list(base_value.values()))
                print(f"Base value (average): {base_value_display:.2f} {param_info['unit']}")
            else:
                print(f"Base value: {base_value:.4f} {param_info['unit']}")
            
            for multiplier in self.variation_levels:
                if multiplier == 1.0:
                    # Use base case result
                    self.results[param_key][multiplier] = self.base_objective
                    print(f"  Multiplier {multiplier:.2f}: ${self.base_objective:,.2f} (base case)")
                else:
                    scenario_count += 1
                    print(f"  Solving scenario {scenario_count}/{total_scenarios}: multiplier {multiplier:.2f}...", end='')
                    
                    obj_value = self.solve_with_parameter(param_key, multiplier)
                    
                    if obj_value is not None:
                        self.results[param_key][multiplier] = obj_value
                        change_pct = ((obj_value - self.base_objective) / self.base_objective) * 100
                        print(f" ${obj_value:,.2f} ({change_pct:+.2f}%)")
                    else:
                        self.results[param_key][multiplier] = None
                        print(" INFEASIBLE/NO SOLUTION")
        
        print("\n✓ Univariate analysis completed")
    
    def calculate_elasticities(self):
        """
        STEP 3: Calculate point-to-point elasticities.
        Elasticity = (ΔZ/Z₀) / (Δp/p₀)
        """
        print("\n" + "="*80)
        print("STEP 3: CALCULATING ELASTICITIES")
        print("="*80)
        
        for param_key in self.parameters.keys():
            param_elasticities = []
            
            for multiplier in self.variation_levels:
                if multiplier == 1.0:
                    continue  # Skip base case
                
                obj_value = self.results[param_key].get(multiplier)
                
                if obj_value is not None:
                    # Calculate changes
                    delta_Z = obj_value - self.base_objective
                    delta_p = multiplier - 1.0  # Since multiplier = p/p₀
                    
                    # Calculate elasticity
                    if delta_p != 0:
                        elasticity = (delta_Z / self.base_objective) / delta_p
                        self.elasticities[param_key][multiplier] = elasticity
                        param_elasticities.append(abs(elasticity))
            
            # Calculate average absolute elasticity
            if param_elasticities:
                avg_elasticity = np.mean(param_elasticities)
                self.elasticities[param_key]['average'] = avg_elasticity
                print(f"{self.parameters[param_key]['name']:40s}: ε_avg = {avg_elasticity:.4f}")
            else:
                self.elasticities[param_key]['average'] = 0
                print(f"{self.parameters[param_key]['name']:40s}: No valid elasticity calculated")
        
        print("\n✓ Elasticities calculated")
    
    def perform_regression_analysis(self):
        """
        STEP 4: Fit linear regression models for each parameter.
        Model: Z = β₀ + β₁·p + ε
        """
        print("\n" + "="*80)
        print("STEP 4: REGRESSION ANALYSIS")
        print("="*80)
        
        for param_key, param_info in self.parameters.items():
            # Prepare data for regression
            multipliers = []
            objectives = []
            
            for mult, obj_val in self.results[param_key].items():
                if obj_val is not None:
                    multipliers.append(mult)
                    objectives.append(obj_val)
            
            if len(multipliers) >= 2:
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(multipliers, objectives)
                r_squared = r_value ** 2
                
                self.regression_results[param_key] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    'std_err': std_err
                }
                
                print(f"\n{param_info['name']}:")
                print(f"  Z = {intercept:,.2f} + {slope:,.2f}·p")
                print(f"  R² = {r_squared:.4f}")
                print(f"  p-value = {p_value:.6f}")
            else:
                print(f"\n{param_info['name']}: Insufficient data for regression")
                self.regression_results[param_key] = None
        
        print("\n✓ Regression analysis completed")
    
    def classify_parameters(self):
        """Classify parameters based on average elasticity."""
        print("\n" + "="*80)
        print("PARAMETER CLASSIFICATION")
        print("="*80)
        
        classifications = {
            'High (>1)': [],
            'Medium ([0.5, 1])': [],
            'Low (<0.5)': []
        }
        
        for param_key, param_info in self.parameters.items():
            avg_elasticity = self.elasticities[param_key].get('average', 0)
            
            if avg_elasticity > 1:
                classification = 'High (>1)'
                implication = 'Critical parameter: requires precise estimate'
            elif avg_elasticity >= 0.5:
                classification = 'Medium ([0.5, 1])'
                implication = 'Important parameter: requires careful estimation'
            else:
                classification = 'Low (<0.5)'
                implication = 'Robust parameter: approximate estimate is sufficient'
            
            classifications[classification].append({
                'name': param_info['name'],
                'elasticity': avg_elasticity,
                'implication': implication
            })
        
        # Display classification
        for category, params in classifications.items():
            if params:
                print(f"\n{category}:")
                for p in params:
                    print(f"  • {p['name']:40s} ε = {p['elasticity']:.4f}")
                    print(f"    → {p['implication']}")
        
        return classifications
    
    def generate_plots(self):
        """Generate visualization plots for sensitivity analysis."""
        print("\n" + "="*80)
        print("GENERATING PLOTS")
        print("="*80)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sensitivity Analysis: Impact of Parameter Variations on Objective Function', 
                     fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (param_key, param_info) in enumerate(self.parameters.items()):
            ax = axes[idx]
            
            # Extract data
            multipliers = []
            objectives = []
            
            for mult in sorted(self.results[param_key].keys()):
                obj_val = self.results[param_key][mult]
                if obj_val is not None:
                    multipliers.append(mult)
                    objectives.append(obj_val)
            
            if len(multipliers) >= 2:
                # Plot actual values
                ax.plot(multipliers, objectives, 'o-', linewidth=2, markersize=8, label='Actual')
                
                # Plot regression line if available
                if param_key in self.regression_results and self.regression_results[param_key]:
                    reg = self.regression_results[param_key]
                    x_line = np.array(multipliers)
                    y_line = reg['intercept'] + reg['slope'] * x_line
                    ax.plot(x_line, y_line, '--', linewidth=2, alpha=0.7, 
                           label=f'Linear fit (R²={reg["r_squared"]:.3f})')
                
                # Mark base case
                ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.5, label='Base case')
                ax.axhline(y=self.base_objective, color='red', linestyle=':', alpha=0.5)
                
                # Formatting
                ax.set_xlabel(f'Parameter Multiplier', fontsize=10)
                ax.set_ylabel('Objective Value ($)', fontsize=10)
                ax.set_title(f'{param_info["name"]}\n(ε_avg = {self.elasticities[param_key].get("average", 0):.3f})', 
                            fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                # Format y-axis
                ax.ticklabel_format(style='plain', axis='y')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        plt.tight_layout()
        plt.savefig('sensitivity_analysis_plots.png', dpi=300, bbox_inches='tight')
        print("✓ Plots saved to 'sensitivity_analysis_plots.png'")
        
        # Create elasticity bar chart
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        param_names = [self.parameters[k]['name'] for k in self.parameters.keys()]
        elasticities = [self.elasticities[k].get('average', 0) for k in self.parameters.keys()]
        
        colors = ['red' if e > 1 else 'orange' if e >= 0.5 else 'green' for e in elasticities]
        
        bars = ax2.barh(param_names, elasticities, color=colors, alpha=0.7, edgecolor='black')
        
        # Add threshold lines
        ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='High threshold (ε=1)')
        ax2.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Medium threshold (ε=0.5)')
        
        ax2.set_xlabel('Average Absolute Elasticity |ε|', fontsize=12, fontweight='bold')
        ax2.set_title('Parameter Sensitivity Ranking', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('elasticity_ranking.png', dpi=300, bbox_inches='tight')
        print("✓ Elasticity ranking saved to 'elasticity_ranking.png'")
    
    def export_results(self):
        """Export results to Excel file."""
        print("\n" + "="*80)
        print("EXPORTING RESULTS")
        print("="*80)
        
        with pd.ExcelWriter('sensitivity_analysis_results.xlsx', engine='openpyxl') as writer:
            # Sheet 1: Objective values
            obj_data = []
            for param_key in self.parameters.keys():
                for mult in sorted(self.results[param_key].keys()):
                    obj_data.append({
                        'Parameter': self.parameters[param_key]['name'],
                        'Multiplier': mult,
                        'Parameter_Value_Ratio': mult,
                        'Objective_Value': self.results[param_key][mult],
                        'Change_from_Base_%': ((self.results[param_key][mult] - self.base_objective) / self.base_objective * 100) 
                                               if self.results[param_key][mult] else None
                    })
            
            df_obj = pd.DataFrame(obj_data)
            df_obj.to_excel(writer, sheet_name='Objective_Values', index=False)
            
            # Sheet 2: Elasticities
            elast_data = []
            for param_key in self.parameters.keys():
                for mult, elast in self.elasticities[param_key].items():
                    if mult != 'average':
                        elast_data.append({
                            'Parameter': self.parameters[param_key]['name'],
                            'Multiplier': mult,
                            'Elasticity': elast,
                            'Absolute_Elasticity': abs(elast)
                        })
            
            df_elast = pd.DataFrame(elast_data)
            df_elast.to_excel(writer, sheet_name='Elasticities', index=False)
            
            # Sheet 3: Summary
            summary_data = []
            for param_key in self.parameters.keys():
                avg_elast = self.elasticities[param_key].get('average', 0)
                
                if avg_elast > 1:
                    classification = 'High'
                elif avg_elast >= 0.5:
                    classification = 'Medium'
                else:
                    classification = 'Low'
                
                reg = self.regression_results.get(param_key)
                
                summary_data.append({
                    'Parameter': self.parameters[param_key]['name'],
                    'Unit': self.parameters[param_key]['unit'],
                    'Average_Elasticity': avg_elast,
                    'Classification': classification,
                    'R_Squared': reg['r_squared'] if reg else None,
                    'Regression_Slope': reg['slope'] if reg else None,
                    'Regression_Intercept': reg['intercept'] if reg else None
                })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary = df_summary.sort_values('Average_Elasticity', ascending=False)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 4: Base case info
            base_info = pd.DataFrame([{
                'Metric': 'Base Objective Value',
                'Value': self.base_objective,
                'Unit': '$'
            }])
            base_info.to_excel(writer, sheet_name='Base_Case', index=False)
        
        print("✓ Results exported to 'sensitivity_analysis_results.xlsx'")
    
    def run_complete_analysis(self):
        """Run the complete sensitivity analysis workflow."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("FOOD DONATION MODEL - SENSITIVITY ANALYSIS")
        print("="*80)
        print(f"Instance: {self.base_data_file}")
        print(f"Parameters analyzed: {len(self.parameters)}")
        print(f"Variation levels: {self.variation_levels}")
        print(f"Total scenarios: {len(self.parameters) * len(self.variation_levels)}")
        
        # Execute analysis steps
        self.load_base_data()
        self.solve_base_case()
        self.run_univariate_analysis()
        self.calculate_elasticities()
        self.perform_regression_analysis()
        classifications = self.classify_parameters()
        self.generate_plots()
        self.export_results()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total execution time: {elapsed/60:.2f} minutes")
        print("\nGenerated files:")
        print("  • sensitivity_analysis_results.xlsx - Detailed numerical results")
        print("  • sensitivity_analysis_plots.png - Parameter response curves")
        print("  • elasticity_ranking.png - Parameter sensitivity ranking")
        print("\n" + "="*80)


# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SensitivityAnalyzer('instance_distance_exp.json')
    
    # Run complete analysis
    analyzer.run_complete_analysis()



