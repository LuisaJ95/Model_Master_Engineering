# %%
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import itertools
import time
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import multiprocessing as mp
from functools import partial
from tqdm.auto import tqdm


class FoodDonationOptimizer:
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
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.load_data_from_dict(data)

    def load_data_from_dict(self, data):
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
    
    def update_parameters(self, beta=None, pi=None, c_trans=None, alpha=None, 
                         bank_capacity_multiplier=None, transport_capacity_multiplier=None):
        if beta is not None:
            self.beta = beta
        if pi is not None:
            self.pi = pi
        if c_trans is not None:
            self.c_trans = c_trans
        if alpha is not None:
            self.alpha = alpha
        if bank_capacity_multiplier is not None:
            for j in self.B:
                self.food_bank_capacity[j] *= bank_capacity_multiplier
        if transport_capacity_multiplier is not None:
            for t in self.T:
                self.transport_capacity[t] *= transport_capacity_multiplier
    
    def build_model(self):
        if self.model is not None:
            self.model.dispose()
        
        self.model = gp.Model("FoodDonationOptimization", env=self.env)
        self.model.setParam('OutputFlag', 0)
        
        self._create_variables()
        self._set_objective()
        self._add_constraints()
        self.model.update()
    
    def _create_variables(self):
        self.y_deliv = {}
        self.x_out = {}
        self.y_pickup = {}
        self.w = {}
        
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
        
        outsource_cost_terms = []
        
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            dist = req['distance']
            l_r = req['release_date']
            e_r = req['expiration_date']
            cost_coef = self.alpha * self.c_trans * dist
            
            for t in range(l_r, e_r + 1):
                outsource_cost_terms.append(cost_coef * self.x_out[r, t])
        
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
        for r in self.R_direct:
            req = self._get_requirement_by_id(r)
            q_r = req['quantity']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            self.model.addConstr(
                gp.quicksum(self.y_deliv[r, t] + self.x_out[r, t] for t in range(l_r, e_r + 1)) + self.w[r] == q_r,
                name=f"flow_direct_{r}"
            )
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            q_r = req['quantity']
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            self.model.addConstr(
                gp.quicksum(self.y_deliv[r, t] for t in range(l_r + 1, e_r + 1)) + self.w[r] == q_r,
                name=f"flow_indirect_{r}"
            )
        
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
                self.model.addConstr(gp.quicksum(fleet_usage) <= cap_trans, name=f"transport_cap_{t}")
        
        for r in self.R_indirect:
            req = self._get_requirement_by_id(r)
            l_r = req['release_date']
            e_r = req['expiration_date']
            
            for t in range(l_r, e_r):
                self.model.addConstr(self.y_pickup[r, t] == self.y_deliv[r, t + 1], name=f"sync_{r}_{t}")
    
    def optimize(self, time_limit=300):
        self.model.setParam('TimeLimit', time_limit)
        self.model.optimize()
        return self.model.Status
    
    def get_objective_value(self):
        if self.model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and self.model.SolCount > 0:
            return self.model.ObjVal
        return None


# Module-level worker state: initialised once per worker process, reused across tasks.
_worker_env = None
_worker_json_data = None


def _worker_initializer(json_file):
    """Called once when each worker process starts.
    Creates the Gurobi env and parses the JSON so every task in this worker
    can skip those expensive operations."""
    global _worker_env, _worker_json_data
    with open(json_file, 'r', encoding='utf-8') as f:
        _worker_json_data = json.load(f)
    _worker_env = gp.Env(empty=True)
    _worker_env.setParam('OutputFlag', 0)
    _worker_env.setParam('LogToConsole', 0)
    # One thread per worker process avoids oversubscribing the CPU when many
    # workers run in parallel (default Threads=0 would use all cores each).
    _worker_env.setParam('Threads', 1)
    _worker_env.start()


def solve_single_scenario(scenario_data, time_limit):
    """Solve one scenario using the worker-local Gurobi env and pre-parsed data."""
    global _worker_env, _worker_json_data
    _, scenario = scenario_data
    scenario_id = scenario['scenario_id']

    try:
        with FoodDonationOptimizer(_worker_env) as optimizer:
            optimizer.load_data_from_dict(_worker_json_data)

            optimizer.update_parameters(
                beta=scenario['beta_actual'],
                pi=scenario['pi_actual'],
                c_trans=scenario['c_trans_actual'],
                alpha=scenario['alpha_actual'],
                bank_capacity_multiplier=scenario['bank_capacity'],
                transport_capacity_multiplier=scenario['transport_capacity']
            )

            optimizer.build_model()

            solve_start = time.time()
            status = optimizer.optimize(time_limit=time_limit)
            solve_time = time.time() - solve_start

            obj_value = optimizer.get_objective_value()

            return {
                'scenario_id': scenario_id,
                'beta': scenario['beta'],
                'pi': scenario['pi'],
                'c_trans': scenario['c_trans'],
                'alpha': scenario['alpha'],
                'bank_capacity': scenario['bank_capacity'],
                'transport_capacity': scenario['transport_capacity'],
                'objective_value': obj_value,
                'status': status,
                'solve_time': solve_time
            }

    except Exception as e:
        print(f"Error solving scenario {scenario_id}: {str(e)}")
        return {
            'scenario_id': scenario_id,
            'beta': scenario['beta'],
            'pi': scenario['pi'],
            'c_trans': scenario['c_trans'],
            'alpha': scenario['alpha'],
            'bank_capacity': scenario['bank_capacity'],
            'transport_capacity': scenario['transport_capacity'],
            'objective_value': None,
            'status': -1,
            'solve_time': 0,
            'error': str(e)
        }


class SensitivityAnalyzer:
    def __init__(self, json_file, n_processes=4):
        self.json_file = json_file
        self.base_params = {}
        self.param_names = ['beta', 'pi', 'c_trans', 'alpha', 'bank_capacity', 'transport_capacity']
        # OPTIMIZED: Reduced from 7 levels to 5 levels
        self.levels = [0.5, 0.85, 1.0, 1.3, 1.5]
        self.results_dir = Path('sensitivity_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Set number of parallel processes (default to CPU count - 1)
        if n_processes is None:
            self.n_processes = max(1, mp.cpu_count() - 1)
        else:
            self.n_processes = n_processes
        
        print(f"Initialized with {self.n_processes} parallel processes")
        
        self._load_base_parameters()
    
    def _load_base_parameters(self):
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        params = data['parameters']
        self.base_params = {
            'beta': params['beta_1'],
            'pi': params['pi_penalty'],
            'c_trans': params['c_trans'],
            'alpha': params['alpha_outsource'],
            'bank_capacity': 1.0,
            'transport_capacity': 1.0
        }
    
    def generate_design_matrix(self):
        all_combinations = list(itertools.product(self.levels, repeat=len(self.param_names)))
        
        design_matrix = pd.DataFrame(all_combinations, columns=self.param_names)
        
        for param in self.param_names:
            design_matrix[f'{param}_actual'] = design_matrix[param] * self.base_params[param]
        
        design_matrix['scenario_id'] = range(len(design_matrix))
        
        design_matrix.to_csv(self.results_dir / 'design_matrix.csv', index=False)
        
        print(f"Design matrix generated: {len(design_matrix)} scenarios")
        print(f"Reduction from original: {7**6} -> {len(design_matrix)} scenarios ({100*(1-len(design_matrix)/(7**6)):.1f}% reduction)")
        
        return design_matrix
    
    def run_full_factorial_parallel(self, design_matrix, start_idx=0, end_idx=None, time_limit=60):
        if end_idx is None:
            end_idx = len(design_matrix)

        results_file = self.results_dir / f'results_{start_idx}_{end_idx}.pkl'

        # Convert rows to plain dicts so multiprocessing can pickle them cleanly.
        scenarios_to_solve = [
            (idx, design_matrix.iloc[idx].to_dict())
            for idx in range(start_idx, end_idx)
        ]

        total_scenarios = len(scenarios_to_solve)
        start_time = time.time()

        print(f"Processing {total_scenarios} scenarios using {self.n_processes} parallel processes...")

        solve_func = partial(solve_single_scenario, time_limit=time_limit)

        results = []
        with mp.Pool(
            processes=self.n_processes,
            initializer=_worker_initializer,
            initargs=(self.json_file,)
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(solve_func, scenarios_to_solve),
                total=total_scenarios,
                desc="Solving scenarios",
                unit="scenario"
            ):
                results.append(result)

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        elapsed_total = time.time() - start_time
        print(f"\nResults saved to {results_file}")
        print(f"Total time: {elapsed_total/60:.1f} minutes")
        print(f"Average time per scenario: {elapsed_total/total_scenarios:.2f} seconds")

        return results
    
    def load_all_results(self):
        all_results = []
        
        for results_file in self.results_dir.glob('results_*.pkl'):
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
                all_results.extend(results)
        
        df = pd.DataFrame(all_results)
        df.to_csv(self.results_dir / 'all_results.csv', index=False)
        
        print(f"Loaded {len(df)} results")
        
        return df
    
    def calculate_main_effects(self, df):
        df_optimal = df[df['objective_value'].notna()].copy()
        
        base_scenario = df_optimal[
            (df_optimal['beta'] == 1.0) &
            (df_optimal['pi'] == 1.0) &
            (df_optimal['c_trans'] == 1.0) &
            (df_optimal['alpha'] == 1.0) &
            (df_optimal['bank_capacity'] == 1.0) &
            (df_optimal['transport_capacity'] == 1.0)
        ]
        
        if len(base_scenario) == 0:
            print("Warning: Base scenario not found")
            Z0 = df_optimal['objective_value'].median()
        else:
            Z0 = base_scenario['objective_value'].iloc[0]
        
        main_effects = {}
        elasticities = {}
        
        for param in self.param_names:
            effects = []
            elast = []
            
            for level in self.levels:
                subset = df_optimal[df_optimal[param] == level]
                avg_obj = subset['objective_value'].mean()
                
                effects.append({
                    'parameter': param,
                    'level': level,
                    'avg_objective': avg_obj,
                    'count': len(subset)
                })
                
                if level != 1.0:
                    delta_Z = avg_obj - Z0
                    delta_p = level - 1.0
                    elasticity = (delta_Z / Z0) / delta_p if delta_p != 0 else 0
                    
                    elast.append({
                        'parameter': param,
                        'level': level,
                        'elasticity': elasticity
                    })
            
            main_effects[param] = pd.DataFrame(effects)
            elasticities[param] = pd.DataFrame(elast)
        
        main_effects_df = pd.concat(main_effects.values(), ignore_index=True)
        elasticities_df = pd.concat(elasticities.values(), ignore_index=True)
        
        main_effects_df.to_csv(self.results_dir / 'main_effects.csv', index=False)
        elasticities_df.to_csv(self.results_dir / 'elasticities.csv', index=False)
        
        return main_effects_df, elasticities_df, Z0
    
    def calculate_two_way_interactions(self, df):
        df_optimal = df[df['objective_value'].notna()].copy()
        
        interactions = []
        
        param_pairs = list(itertools.combinations(self.param_names, 2))
        
        for param_i, param_j in tqdm(param_pairs, desc="Two-way interactions", unit="pair"):
            for level_i in self.levels:
                for level_j in self.levels:
                    subset = df_optimal[
                        (df_optimal[param_i] == level_i) &
                        (df_optimal[param_j] == level_j)
                    ]
                    
                    if len(subset) > 0:
                        avg_obj = subset['objective_value'].mean()
                        
                        interactions.append({
                            'param_i': param_i,
                            'param_j': param_j,
                            'level_i': level_i,
                            'level_j': level_j,
                            'avg_objective': avg_obj,
                            'count': len(subset)
                        })
        
        interactions_df = pd.DataFrame(interactions)
        interactions_df.to_csv(self.results_dir / 'two_way_interactions.csv', index=False)
        
        return interactions_df
    
    def perform_anova(self, df):
        df_optimal = df[df['objective_value'].notna()].copy()
        
        df_optimal['beta_cat'] = df_optimal['beta'].astype(str)
        df_optimal['pi_cat'] = df_optimal['pi'].astype(str)
        df_optimal['c_trans_cat'] = df_optimal['c_trans'].astype(str)
        df_optimal['alpha_cat'] = df_optimal['alpha'].astype(str)
        df_optimal['bank_capacity_cat'] = df_optimal['bank_capacity'].astype(str)
        df_optimal['transport_capacity_cat'] = df_optimal['transport_capacity'].astype(str)
        
        grand_mean = df_optimal['objective_value'].mean()
        ss_total = np.sum((df_optimal['objective_value'] - grand_mean) ** 2)
        
        anova_results = {}
        
        for param in self.param_names:
            param_cat = f'{param}_cat'
            group_means = df_optimal.groupby(param_cat)['objective_value'].mean()
            group_counts = df_optimal.groupby(param_cat).size()
            
            ss_param = np.sum(group_counts * (group_means - grand_mean) ** 2)
            eta_squared = (ss_param / ss_total) * 100 if ss_total > 0 else 0
            
            anova_results[param] = {
                'SS': ss_param,
                'eta_squared': eta_squared
            }
        
        anova_df = pd.DataFrame(anova_results).T
        anova_df['SS_total'] = ss_total
        
        anova_df.to_csv(self.results_dir / 'anova_results.csv')
        
        return anova_df
    
    def fit_metamodel(self, df):
        df_optimal = df[df['objective_value'].notna()].copy()
        
        X = df_optimal[self.param_names].values
        y = df_optimal['objective_value'].values
        
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        y_pred = model.predict(X_poly)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        n = len(y)
        p = X_poly.shape[1] - 1
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        feature_names = poly.get_feature_names_out(self.param_names)
        
        coefficients = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_
        })
        
        coefficients.to_csv(self.results_dir / 'metamodel_coefficients.csv', index=False)
        
        metamodel_metrics = {
            'R2': r2,
            'R2_adjusted': r2_adj,
            'RMSE': rmse,
            'n_samples': n,
            'n_features': p
        }
        
        with open(self.results_dir / 'metamodel_metrics.json', 'w') as f:
            json.dump(metamodel_metrics, f, indent=2)
        
        with open(self.results_dir / 'metamodel.pkl', 'wb') as f:
            pickle.dump({'model': model, 'poly': poly}, f)
        
        return model, poly, metamodel_metrics
    
    def plot_main_effects(self, main_effects_df):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, param in enumerate(self.param_names):
            subset = main_effects_df[main_effects_df['parameter'] == param]
            
            axes[idx].plot(subset['level'], subset['avg_objective'], marker='o', linewidth=2)
            axes[idx].set_xlabel(f'{param} multiplier', fontsize=12)
            axes[idx].set_ylabel('Average Objective Value', fontsize=12)
            axes[idx].set_title(f'Main Effect: {param}', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'main_effects_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_elasticities(self, elasticities_df):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for param in self.param_names:
            subset = elasticities_df[elasticities_df['parameter'] == param]
            ax.plot(subset['level'], subset['elasticity'], marker='o', label=param, linewidth=2)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Parameter Multiplier', fontsize=12)
        ax.set_ylabel('Elasticity', fontsize=12)
        ax.set_title('Parameter Elasticities', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'elasticities_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pareto_chart(self, anova_df):
        anova_sorted = anova_df.sort_values('eta_squared', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(anova_sorted)), anova_sorted['eta_squared'])
        ax.set_xticks(range(len(anova_sorted)))
        ax.set_xticklabels(anova_sorted.index, rotation=45, ha='right')
        ax.set_ylabel('Variance Contribution (%)', fontsize=12)
        ax.set_title('Pareto Chart: Parameter Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (param, row) in enumerate(anova_sorted.iterrows()):
            ax.text(i, row['eta_squared'] + 0.5, f"{row['eta_squared']:.1f}%", 
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'pareto_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_interaction_heatmap(self, interactions_df, param_i, param_j):
        subset = interactions_df[
            (interactions_df['param_i'] == param_i) &
            (interactions_df['param_j'] == param_j)
        ]
        
        pivot = subset.pivot(index='level_i', columns='level_j', values='avg_objective')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Objective Value'})
        
        ax.set_xlabel(f'{param_j} multiplier', fontsize=12)
        ax.set_ylabel(f'{param_i} multiplier', fontsize=12)
        ax.set_title(f'Two-Way Interaction: {param_i} × {param_j}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'interaction_{param_i}_{param_j}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, df, main_effects_df, elasticities_df, anova_df, Z0, metamodel_metrics):
        report = []
        
        report.append("="*80)
        report.append("OPTIMIZED FULL FACTORIAL SENSITIVITY ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        report.append("1. EXPERIMENTAL DESIGN")
        report.append("-" * 80)
        report.append(f"   Total scenarios evaluated: {len(df)}")
        report.append(f"   Optimal solutions found: {df['objective_value'].notna().sum()}")
        report.append(f"   Infeasible solutions: {(df['status'] == GRB.INFEASIBLE).sum()}")
        report.append(f"   Parameters analyzed: {len(self.param_names)}")
        report.append(f"   Levels per parameter: {len(self.levels)}")
        report.append(f"   Levels used: {self.levels}")
        report.append(f"   Design type: Full factorial (5^6)")
        report.append(f"   Parallel processes: {self.n_processes}")
        report.append("")
        
        report.append("2. BASE CASE")
        report.append("-" * 80)
        report.append(f"   Base objective value (Z0): ${Z0:,.2f}")
        for param, value in self.base_params.items():
            report.append(f"   {param}: {value}")
        report.append("")
        
        report.append("3. MAIN EFFECTS (ANOVA)")
        report.append("-" * 80)
        anova_sorted = anova_df.sort_values('eta_squared', ascending=False)
        for param, row in anova_sorted.iterrows():
            classification = "HIGH" if row['eta_squared'] > 30 else "MEDIUM" if row['eta_squared'] >= 10 else "LOW"
            report.append(f"   {param:20s}: η² = {row['eta_squared']:6.2f}% [{classification}]")
        report.append("")
        
        report.append("4. AVERAGE ELASTICITIES")
        report.append("-" * 80)
        avg_elasticities = elasticities_df.groupby('parameter')['elasticity'].apply(lambda x: np.abs(x).mean())
        avg_elasticities_sorted = avg_elasticities.sort_values(ascending=False)
        for param, elast in avg_elasticities_sorted.items():
            classification = "HIGH" if elast > 1 else "MEDIUM" if elast >= 0.5 else "LOW"
            report.append(f"   {param:20s}: |ε| = {elast:6.3f} [{classification}]")
        report.append("")
        
        report.append("5. METAMODEL QUALITY")
        report.append("-" * 80)
        report.append(f"   R²: {metamodel_metrics['R2']:.4f}")
        report.append(f"   R² adjusted: {metamodel_metrics['R2_adjusted']:.4f}")
        report.append(f"   RMSE: ${metamodel_metrics['RMSE']:,.2f}")
        report.append(f"   Samples used: {metamodel_metrics['n_samples']}")
        report.append("")
        
        report.append("6. PARAMETER CLASSIFICATION")
        report.append("-" * 80)
        report.append("   Based on total variance contribution (η²):")
        report.append("")
        report.append("   CRITICAL (η² > 30%):")
        for param, row in anova_sorted.iterrows():
            if row['eta_squared'] > 30:
                report.append(f"      - {param}: Requires precise estimation and monitoring")
        report.append("")
        report.append("   IMPORTANT (10% ≤ η² ≤ 30%):")
        for param, row in anova_sorted.iterrows():
            if 10 <= row['eta_squared'] <= 30:
                report.append(f"      - {param}: Requires careful estimation")
        report.append("")
        report.append("   ROBUST (η² < 10%):")
        for param, row in anova_sorted.iterrows():
            if row['eta_squared'] < 10:
                report.append(f"      - {param}: Approximate estimate is sufficient")
        report.append("")
        
        report.append("7. KEY INSIGHTS")
        report.append("-" * 80)
        most_important = anova_sorted.index[0]
        report.append(f"   - Most influential parameter: {most_important} (η² = {anova_sorted.iloc[0]['eta_squared']:.2f}%)")
        
        least_important = anova_sorted.index[-1]
        report.append(f"   - Least influential parameter: {least_important} (η² = {anova_sorted.iloc[-1]['eta_squared']:.2f}%)")
        
        total_explained = anova_sorted['eta_squared'].sum()
        report.append(f"   - Total variance explained by main effects: {total_explained:.2f}%")
        
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        with open(self.results_dir / 'sensitivity_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        return report_text


def main():
    print("="*80)
    print("OPTIMIZED FULL FACTORIAL SENSITIVITY ANALYSIS")
    print("Food Donation Optimization Model")
    print("="*80)
    print()
    print("OPTIMIZATIONS:")
    print("  1. Reduced levels: 7 -> 5 (from 117,649 to 15,625 scenarios)")
    print("  2. Parallel processing with multiprocessing")
    print("="*80)
    print()
    
    json_file = 'instance_distance_exp.json'
    
    # Initialize analyzer with parallel processing
    # You can specify n_processes manually, or let it use CPU count - 1
    analyzer = SensitivityAnalyzer(json_file, n_processes=12)
    
    print("Step 1: Generating design matrix...")
    design_matrix = analyzer.generate_design_matrix()
    print(f"✓ Design matrix created: {len(design_matrix)} scenarios\n")
    
    print("Step 2: Running full factorial experiment in parallel...")
    print(f"This will evaluate {len(design_matrix)} scenarios using {analyzer.n_processes} parallel processes.")
    
    # For very large experiments, you can still use batches if needed
    # But with 15,625 scenarios and parallel processing, you can likely run all at once
    batch_size = len(design_matrix)  # Process all scenarios at once
    n_batches = int(np.ceil(len(design_matrix) / batch_size))
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(design_matrix))
        
        print(f"\nBatch {batch_idx + 1}/{n_batches}: Scenarios {start_idx} to {end_idx}")
        
        results = analyzer.run_full_factorial_parallel(
            design_matrix,
            start_idx=start_idx,
            end_idx=end_idx,
            time_limit=60
        )
    
    print("\n✓ All batches completed\n")
    
    print("Step 3: Loading and consolidating results...")
    df = analyzer.load_all_results()
    print(f"✓ Loaded {len(df)} results\n")
    
    print("Step 4: Calculating main effects...")
    main_effects_df, elasticities_df, Z0 = analyzer.calculate_main_effects(df)
    print("✓ Main effects calculated\n")
    
    print("Step 5: Calculating two-way interactions...")
    interactions_df = analyzer.calculate_two_way_interactions(df)
    print("✓ Interactions calculated\n")
    
    print("Step 6: Performing ANOVA...")
    anova_df = analyzer.perform_anova(df)
    print("✓ ANOVA completed\n")
    
    print("Step 7: Fitting metamodel...")
    model, poly, metamodel_metrics = analyzer.fit_metamodel(df)
    print(f"✓ Metamodel fitted (R² = {metamodel_metrics['R2']:.4f})\n")
    
    print("Step 8: Generating visualizations...")
    analyzer.plot_main_effects(main_effects_df)
    analyzer.plot_elasticities(elasticities_df)
    analyzer.plot_pareto_chart(anova_df)
    
    for param_i, param_j in [('beta', 'pi'), ('c_trans', 'alpha'), 
                             ('bank_capacity', 'transport_capacity')]:
        analyzer.plot_interaction_heatmap(interactions_df, param_i, param_j)
    
    print("✓ Visualizations saved\n")
    
    print("Step 9: Generating final report...")
    analyzer.generate_report(df, main_effects_df, elasticities_df, anova_df, Z0, metamodel_metrics)
    print("✓ Report generated\n")
    
    print("="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved in: {analyzer.results_dir}")
    print("\nGenerated files:")
    print("  - design_matrix.csv")
    print("  - all_results.csv")
    print("  - main_effects.csv")
    print("  - elasticities.csv")
    print("  - two_way_interactions.csv")
    print("  - anova_results.csv")
    print("  - metamodel_coefficients.csv")
    print("  - metamodel_metrics.json")
    print("  - sensitivity_analysis_report.txt")
    print("  - main_effects_plot.png")
    print("  - elasticities_plot.png")
    print("  - pareto_chart.png")
    print("  - interaction_*.png")


if __name__ == "__main__":
    # Required for Windows compatibility with multiprocessing
    mp.freeze_support()
    main()



