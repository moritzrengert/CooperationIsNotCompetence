"""
New Metric System for LLM Public-Good Experiments

This module implements a comprehensive metric system that avoids arbitrary weights
and hand-tuned penalties, following the design principles:
1. No arbitrary weights
2. No hand-tuned penalties  
3. Prefer raw or rule-derived quantities
4. Keep composite interpretation separate (Pareto ranking)

Metrics implemented:
- Payoff Efficiency (E_s): normalized distance between actual and feasible payoffs
- Trajectory Gain (G_s): normalized improvement from first to last period
- Mechanism Use (M_s): fraction of cooperation from conditional mechanism use
- Conditional Cooperation Slope (beta_s): responsiveness to others' contributions
- Robustness Retention (R^E, R^C, R^M): worst-case relative performance under perturbations
- Pareto Overview: dominance share and layer assignment
- Bottleneck Score (B_s): minimum of normalized dimensions
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from src.games.ro_public_good import RO_ENDOWMENT, RO_MPCR, RO_PERIODS, RO_PLAYERS

if TYPE_CHECKING:
    from pandas import DataFrame, Series


def _group_period_cols(df: 'DataFrame') -> List[str]:
    """Columns that identify one independent group-period observation."""
    if df.empty or 'generation' not in df.columns:
        return []
    cols: List[str] = []
    for col in ['run_label', 'seed', 'repetition']:
        if col in df.columns:
            cols.append(col)
    if 'group_id' in df.columns:
        cols.append('group_id')
    elif 'chain_id' in df.columns:
        cols.append('chain_id')
    cols.append('generation')
    return list(dict.fromkeys(cols))


# Fischer CPR parameters
FISCHER_MAX_EXTRACTION = 8.0
FISCHER_EFFICIENT_EXTRACTION = 3.0


def _get_ro_payoff_params(n_players: int = RO_PLAYERS) -> Tuple[float, float]:
    """
    Compute theoretical min and max feasible payoffs for RO public-good game.
    
    For VCM/BCCM/CCM with n players, endowment e, MPCR m:
    - Worst case (full free-riding): each keeps e, total payoff = n * e
    - Best case (full contribution): each contributes e, payoff per player = m * e * n
      
    Returns: (min_total_payoff, max_total_payoff)
    """
    min_payoff_per_player = RO_ENDOWMENT  # defect: keep endowment
    max_payoff_per_player = RO_MPCR * RO_ENDOWMENT * n_players  # cooperate: full contribution
    
    return min_payoff_per_player, max_payoff_per_player


def _get_group_total_payoff(df: 'DataFrame') -> 'Series':
    """
    Compute total group payoff for each group-period.
    Pi_{g,t} = sum_i pi_{i,g,t}
    """
    if df.empty or 'payoff' not in df.columns:
        return pd.Series(dtype=float)
    
    group_cols = _group_period_cols(df)
    if not group_cols:
        return pd.Series(dtype=float)
    return df.groupby(group_cols, dropna=False)['payoff'].sum()


def compute_payoff_efficiency(
    df: 'DataFrame',
    n_players: int = RO_PLAYERS,
    endowment: float = RO_ENDOWMENT,
    mpcr: float = RO_MPCR
) -> Tuple['DataFrame', Dict[str, float]]:
    """
    Compute Payoff Efficiency (E_s) as defined in metric 1.
    
    For each group-period (g,t):
        E_{g,t} = (Pi_{g,t} - Pi^min_{g,t}) / (Pi^max_{g,t} - Pi^min_{g,t})
        
    Where:
    - Pi_{g,t} is actual total group payoff
    - Pi^min_{g,t} is worst feasible total group payoff (full free-riding)
    - Pi^max_{g,t} is best feasible total group payoff (full cooperation)
    
    Setup-level payoff efficiency:
        E_s = average across all observed group-periods
    
    Args:
        df: DataFrame with game results, must contain 'payoff' column
        n_players: number of players in each group
        endowment: per-player endowment
        mpcr: marginal per-capita return
        
    Returns:
        Tuple of (group-period efficiency DataFrame, setup-level efficiency dict)
    """
    if df.empty or 'payoff' not in df.columns:
        return pd.DataFrame(), {}
    
    # Compute feasible endpoints
    pi_min_per_player = endowment  # defect: keep endowment
    pi_max_per_player = mpcr * endowment * n_players  # full cooperation payoff per player
    pi_min_total = n_players * pi_min_per_player  # total when all defect
    pi_max_total = n_players * pi_max_per_player  # total when all cooperate
    
    group_cols = _group_period_cols(df)
    if not group_cols:
        return pd.DataFrame(), {}

    # Compute actual total payoffs per group-period.  Repetition/run_label are
    # included when present because chain ids are reused across multi-seed runs.
    group_total_payoff = df.groupby(group_cols, dropna=False)['payoff'].sum()

    # Compute efficiency for each group-period
    efficiency_gp = (group_total_payoff - pi_min_total) / (pi_max_total - pi_min_total)
    efficiency_gp = efficiency_gp.clip(lower=0, upper=1).rename('payoff_efficiency')

    # Compute setup-level efficiency (mean across group-periods)
    setup_cols = [col for col in ['model', 'mechanism', 'prompt_modifier', 'game'] if col in df.columns]

    gp_df = group_total_payoff.reset_index(name='total_payoff')
    group_index = gp_df.set_index(group_cols).index
    for col in setup_cols:
        mapping = df.groupby(group_cols, dropna=False)[col].first().to_dict()
        gp_df[col] = group_index.map(mapping)

    gp_df['payoff_efficiency'] = group_index.map(efficiency_gp.to_dict())

    setup_efficiency = {}
    if setup_cols:
        for keys, sub in gp_df.groupby(setup_cols, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            key_str = '_'.join(str(k) for k in keys)
            setup_efficiency[key_str] = sub['payoff_efficiency'].mean()

    return gp_df[group_cols + setup_cols + ['payoff_efficiency']], setup_efficiency


def compute_trajectory_gain(
    efficiency_by_period: Dict[int, float],
    use_last5: bool = False
) -> float:
    """
    Compute Trajectory Gain (G_s) as defined in metric 2.
    
    G_s = (E_{s,T} - E_{s,1}) / (1 - E_{s,1})
    
    where E_{s,1} is payoff efficiency in first period and E_{s,T} in final period.
    
    If final-period noise is high, use last-window version:
        G^{last5}_s = (E_{s,last5} - E_{s,first5}) / (1 - E_{s,first5})
    
    Args:
        efficiency_by_period: dict mapping period -> average efficiency
        use_last5: if True, compare first 5 periods to last 5 periods
        
    Returns:
        Trajectory gain (float)
    """
    if not efficiency_by_period:
        return np.nan
    
    sorted_periods = sorted(efficiency_by_period.keys())
    if len(sorted_periods) < 2:
        return np.nan
    
    if use_last5 and len(sorted_periods) >= 10:
        first5_eff = np.mean([efficiency_by_period[p] for p in sorted_periods[:5]])
        last5_eff = np.mean([efficiency_by_period[p] for p in sorted_periods[-5:]])
        numerator = last5_eff - first5_eff
        denominator = 1.0 - first5_eff
    else:
        e_1 = efficiency_by_period[sorted_periods[0]]
        e_T = efficiency_by_period[sorted_periods[-1]]
        numerator = e_T - e_1
        denominator = 1.0 - e_1
    
    if denominator == 0:
        # If E_{s,1} = 1, improvement is impossible (already at max)
        return 0.0 if numerator >= 0 else np.nan
    
    return numerator / denominator


def compute_mechanism_use(
    df: 'DataFrame',
    mechanism: str
) -> float:
    """
    Compute Mechanism Use (M_s) as defined in metric 3.
    
    For BCCM/SCCM/CCM/CCF-style mechanisms:
        M_s = sum_{i,g,t} q_{i,g,t} * 1[conditional mechanism choice] / sum_{i,g,t} q_{i,g,t}
        
    where q_{i,g,t} is the implemented contribution or contribution share.
    
    For BCCM:
        1[conditional mechanism choice] = 1 iff k in {1,...,n-1} (stated min contributors > 0 and < n)
        
    For SCCM/CCM:
        1[conditional mechanism choice] = 1 iff offered contribution > 0 AND threshold > 0
        
    For CCF:
        1[conditional mechanism choice] = 1 iff the submitted function varies with others_total

    For VCM: N/A (no conditional mechanism)
    
    Args:
        df: DataFrame with game results
        mechanism: 'bccm', 'sccm', 'ccm', or 'ccf'
        
    Returns:
        Mechanism use ratio (float) or NaN if not applicable
    """
    if df.empty:
        return np.nan
    
    mechanism = mechanism.lower()
    
    # Get contribution weights
    if 'implemented_contribution' in df.columns:
        q = pd.to_numeric(df['implemented_contribution'], errors='coerce').fillna(0)
    elif 'contribution_share' in df.columns:
        q = pd.to_numeric(df['contribution_share'], errors='coerce').fillna(0) * RO_ENDOWMENT
    else:
        # Use binary contribution as weight (10 for contribute, 0 for not)
        if 'contributes' in df.columns:
            q = df['contributes'].astype(bool).astype(float) * RO_ENDOWMENT
        else:
            return np.nan
    
    total_q = q.sum()
    if total_q == 0:
        return 0.0
    
    # Identify conditional choices based on mechanism
    if mechanism == 'bccm':
        # BCCM: condition k is in {1,...,n-1} (not 0 or n)
        if 'choice' in df.columns:
            conditional = pd.to_numeric(df['choice'], errors='coerce').between(1, RO_PLAYERS - 1)
        elif 'condition' in df.columns:
            conditional = pd.to_numeric(df['condition'], errors='coerce').between(1, RO_PLAYERS - 1)
        else:
            return np.nan
            
    elif mechanism in {'ccm', 'sccm'}:
        # CCM/SCCM: offered contribution > 0 AND threshold > 0.
        cont1 = pd.to_numeric(df.get('contribution_1', 0), errors='coerce').fillna(0)
        thresh1 = pd.to_numeric(df.get('threshold_1', 0), errors='coerce').fillna(0)
        cont2 = pd.to_numeric(df.get('contribution_2', 0), errors='coerce').fillna(0)
        thresh2 = pd.to_numeric(df.get('threshold_2', 0), errors='coerce').fillna(0)

        conditional = (cont1 > 0) & (thresh1 > 0)
        if mechanism == 'ccm':
            conditional = conditional | ((cont2 > 0) & (thresh2 > 0))

    elif mechanism == 'ccf':
        conditional = df.get('ccf_is_conditional', pd.Series(False, index=df.index)).fillna(False).astype(bool)

    else:
        # VCM or unknown
        return np.nan
    
    conditional_q = (q * conditional).sum()
    return conditional_q / total_q if total_q > 0 else 0.0


def compute_conditional_cooperation_slope(
    df: 'DataFrame'
) -> float:
    """
    Compute Conditional Cooperation Slope (beta_s) as defined in metric 4.
    
        beta_s = Cov(c_{i,t}, c_{-i,t-1}) / Var(c_{-i,t-1})
    
    where:
    - c_{i,t} is player i's contribution share in period t
    - c_{-i,t-1} is the average contribution share of others in period t-1
    
    Args:
        df: DataFrame with game results, must contain 'contribution_share' and 'generation'
        
    Returns:
        Conditional cooperation slope (float) or NaN if not computable
    """
    if df.empty:
        return np.nan
    
    required_cols = ['contribution_share', 'generation']
    if not all(col in df.columns for col in required_cols):
        # Try to use alternative columns
        alt_cols = []
        if 'implemented_contribution' in df.columns:
            c = pd.to_numeric(df['implemented_contribution'], errors='coerce') / RO_ENDOWMENT
        elif 'contributes' in df.columns:
            c = df['contributes'].astype(bool).astype(float)
        else:
            return np.nan
    else:
        c = pd.to_numeric(df['contribution_share'], errors='coarse').fillna(0)
    
    gen = pd.to_numeric(df['generation'], errors='coerce')
    player = df.get('player', range(len(df)))
    chain = df.get('chain_id', df.get('group_id', pd.Series([0] * len(df), index=df.index)))
    
    # For each player-period, compute average of others in previous period
    others_prev = []
    for idx, row in df.iterrows():
        current_chain = chain.iloc[idx]
        current_gen = gen.iloc[idx]
        current_player = player.iloc[idx] if hasattr(player, 'iloc') else 0
        
        # Get all decisions from same chain in previous period
        prev_mask = (chain == current_chain) & (gen == current_gen - 1)
        prev_data = df[prev_mask]
        
        if len(prev_data) < 2:
            others_prev.append(0.0)
            continue
        
        # Exclude current player's previous contribution
        if hasattr(player, 'iloc'):
            prev_others = [c.iloc[i] for i in prev_data.index if chain.iloc[i] == current_chain and gen.iloc[i] == current_gen - 1]
        else:
            prev_others = []
        
        if len(prev_others) < 2:
            others_prev.append(0.0)
        else:
            others_prev.append(np.mean(prev_others))
    
    c_vector = c.to_numpy()
    c_prev_mean = np.array(others_prev)
    
    # Remove NaN/invalid entries
    valid = ~np.isnan(c_vector) & ~np.isnan(c_prev_mean) & (c_prev_mean > 0)
    if np.sum(valid) < 2:
        return np.nan
    
    c_valid = c_vector[valid]
    c_prev_valid = c_prev_mean[valid]
    
    covariance = np.cov(c_valid, c_prev_valid)[0, 1]
    variance = np.var(c_prev_valid)
    
    if variance == 0:
        return np.nan
    
    return covariance / variance


def compute_robustness_retention(
    baseline_metrics: Dict[str, float],
    perturbation_metrics: Dict[str, Dict[str, float]]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute Robustness Retention as defined in metric 5.
    
        R^E_s = min_{p in P_s} E_{s,p} / E_{s,0}
        R^C_s = min_{p in P_s} C_{s,p} / C_{s,0}
        R^M_s = min_{p in P_s} M_{s,p} / M_{s,0}
    
    where:
    - E_{s,0} is payoff efficiency under baseline (no perturbation)
    - E_{s,p} is payoff efficiency under perturbation p
    - Similarly for contribution retention and mechanism use retention
    
    Args:
        baseline_metrics: dict with keys 'E', 'C', 'M' for baseline
        perturbation_metrics: dict mapping perturbation name to {'E', 'C', 'M'} values
        
    Returns:
        Tuple of (R^E, R^C, R^M) - worst-case relative retention ratios
    """
    if not baseline_metrics or not perturbation_metrics:
        return None, None, None
    
    # Extract metrics from baseline
    E_0 = baseline_metrics.get('E')
    C_0 = baseline_metrics.get('C')
    M_0 = baseline_metrics.get('M')
    
    if E_0 is None or E_0 == 0:
        R_E = None
    else:
        # Find minimum E_p / E_0 across perturbations
        min_ratio = float('inf')
        for pert_name, metrics in perturbation_metrics.items():
            E_p = metrics.get('E')
            if E_p is not None and E_0 != 0:
                ratio = E_p / E_0
                if ratio < min_ratio:
                    min_ratio = ratio
        R_E = min_ratio if min_ratio != float('inf') else 1.0
    
    if C_0 is None or C_0 == 0:
        R_C = None
    else:
        min_ratio = float('inf')
        for pert_name, metrics in perturbation_metrics.items():
            C_p = metrics.get('C')
            if C_p is not None and C_0 != 0:
                ratio = C_p / C_0
                if ratio < min_ratio:
                    min_ratio = ratio
        R_C = min_ratio if min_ratio != float('inf') else 1.0
    
    if M_0 is None or M_0 == 0:
        R_M = None
    else:
        min_ratio = float('inf')
        for pert_name, metrics in perturbation_metrics.items():
            M_p = metrics.get('M')
            if M_p is not None and M_0 != 0:
                ratio = M_p / M_0
                if ratio < min_ratio:
                    min_ratio = ratio
        R_M = min_ratio if min_ratio != float('inf') else 1.0
    
    return R_E, R_C, R_M


def compute_pareto_dominance(
    setups: List[Dict[str, float]],
    applicable_metrics: Optional[List[str]] = None
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Compute Pareto dominance rankings as defined in metric 6.
    
    For each setup s with metric vector x_s = (E_s, G_s, M_s, R^E_s), 
    excluding non-applicable dimensions (e.g., M_s for VCM):
    
    Setup a dominates b if:
        - x_{a,j} >= x_{b,j} for all applicable j
        - x_{a,j} > x_{b,j} for at least one applicable j
    
    Dominance share:
        D_s = (1 / (N-1)) * sum_{r != s} 1[s dominates r]
    
    Pareto layers:
        - Layer 1: setups not dominated by any other setup
        - Layer 2: setups not dominated after removing Layer 1
        - etc.
    
    Args:
        setups: list of dicts, each mapping metric names to values for one setup
        applicable_metrics: list of metric names that should be considered (default: all)
        
    Returns:
        Tuple of (layer_assignment: {setup_name -> layer}, dominance_share: {setup_name -> D_s})
    """
    if not setups or len(setups) < 2:
        return {}, {}
    
    # Assign names to anonymous setups
    for i, setup in enumerate(setups):
        if 'name' not in setup and 'setup' not in setup:
            setup['_internal_name'] = f'setup_{i}'
    
    # Determine applicable metrics if not specified
    if applicable_metrics is None:
        # Use all keys that start with uppercase letters or common metric prefixes
        all_keys = set()
        for setup in setups:
            all_keys.update(setup.keys())
        applicable_metrics = [k for k in sorted(all_keys) 
                           if not k.startswith('_') and 
                           not any(excl in k.lower() for excl in ['name', 'internal'])]
    
    # Filter setups to only include applicable metrics
    filtered_setups = []
    for setup in setups:
        filtered = {k: v for k, v in setup.items() if k in applicable_metrics}
        filtered['_name'] = setup.get('name', setup.get('setup', setup.get('_internal_name')))
        filtered_setups.append(filtered)
    
    # Get unique metric names
    metrics = sorted(set().union(*[list(s.keys()) for s in filtered_setups if '_Name' not in s]))
    
    # Build dominance matrix (N x N)
    n = len(filtered_setups)
    dominates_matrix = np.zeros((n, n), dtype=bool)
    
    for i, a in enumerate(filtered_setups):
        for j, b in enumerate(filtered_setups):
            if i == j:
                continue
            # Check if a dominates b: a metrics >= b metrics and > on at least one
            all_ge = True
            any_gt = False
            
            for metric in metrics:
                val_a = a.get(metric, 0)
                val_b = b.get(metric, 0)
                
                # Handle NaN values (treat as non-comparable)
                if pd.isna(val_a) or pd.isna(val_b):
                    continue
                    
                if val_a < val_b:
                    all_ge = False
                    break
                elif val_a > val_b:
                    any_gt = True
            
            # Also check for additional metrics that might only be in one setup
            for metric in set(a.keys()) - set(metrics) - {'_Name'}:
                if pd.isna(a[metric]):
                    continue
                if metric not in b or pd.isna(b.get(metric, 0)):
                    # If a has a metric that b doesn't, a dominates on this dimension by definition
                    any_gt = True
                else:
                    val_b_metric = b[metric]
                    if a[metric] < val_b_metric:
                        all_ge = False
                        break
            
            for metric in set(b.keys()) - set(metrics) - {'_Name'}:
                if pd.isna(b[metric]):
                    continue
                if metric not in a or pd.isna(a.get(metric, 0)):
                    all_ge = False
                    break
            
            dominates_matrix[i, j] = all_ge and any_gt
    
    # Compute dominance share
    dominance_share = {}
    for i in range(n):
        n_dominates = np.sum(dominates_matrix[i])  # Number of setups dominated by this one
        total_other = n - 1
        if total_other > 0:
            dominance_share[filtered_setups[i]['_name']] = n_dominates / total_other
        else:
            dominance_share[filtered_setups[i]['_name']] = 0.0
    
    # Assign Pareto layers using iterative removal
    remaining_indices = set(range(n))
    layer_assignment = {}
    current_layer = 1
    
    while remaining_indices:
        # Find which setups are not dominated by any other remaining setup
        non_dominated = []
        for i in remaining_indices:
            is_dominated = False
            for j in remaining_indices:
                if i != j and dominates_matrix[j, i]:
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(i)
        
        # If none found but there are still indices, assign all to current layer
        if not non_dominated:
            for i in remaining_indices:
                layer_assignment[filtered_setups[i]['_name']] = current_layer
            break
        
        # Assign current layer to non-dominated setups
        for i in non_dominated:
            layer_assignment[filtered_setups[i]['_name']] = current_layer
        
        # Remove assigned setups from remaining
        remaining_indices -= set(non_dominated)
        current_layer += 1
    
    return layer_assignment, dominance_share


def compute_bottleneck_score(
    E_s: float,
    G_s: float,
    M_s: Optional[float],
    R_E_s: float,
    mechanism_has_conditional: bool = True
) -> float:
    """
    Compute Bottleneck Score (B_s) as defined in metric 8.
    
        B_s = min(E_s, G^+_s, M_s, R^E_s)
    
    where G^+_s = max(0, G_s)
    
    For VCM setups without conditional mechanism, M_s is excluded.
    
    Args:
        E_s: payoff efficiency
        G_s: trajectory gain  
        M_s: mechanism use (optional for non-conditional mechanisms)
        R_E_s: robustness retention (payoff efficiency)
        mechanism_has_conditional: whether the mechanism supports conditional cooperation
        
    Returns:
        Bottleneck score (float)
    """
    G_plus = max(0, G_s)
    
    components = [E_s, G_plus, R_E_s]
    if mechanism_has_conditional and M_s is not None:
        components.append(M_s)
    
    # Handle NaN values
    valid_components = [c for c in components if not pd.isna(c)]
    
    if not valid_components:
        return np.nan
    
    return min(valid_components)


@dataclass
class SetupMetrics:
    """
    Container for all metrics for a single setup.
    """
    name: str
    # Core metrics
    payoff_efficiency: float  # E_s
    trajectory_gain: float    # G_s (full period)
    trajectory_gain_last5: Optional[float] = None  # G^{last5}_s
    mechanism_use: Optional[float] = None  # M_s
    conditional_cooperation_slope: Optional[float] = None  # beta_s
    robustness_retention_efficiency: Optional[float] = None  # R^E_s
    robustness_retention_contribution: Optional[float] = None  # R^C_s 
    robustness_retention_mechanism: Optional[float] = None    # R^M_s
    # Composite metrics
    pareto_layer: Optional[int] = None
    dominance_share: Optional[float] = None
    bottleneck_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, object]:
        """Convert all metrics to a dictionary."""
        result = {
            'name': self.name,
            'payoff_efficiency': self.payoff_efficiency,
            'trajectory_gain': self.trajectory_gain,
            'mechanism_use': self.mechanism_use,
            'conditional_cooperation_slope': self.conditional_cooperation_slope,
            'robustness_retention_efficiency': self.robustness_retention_efficiency,
            'robustness_retention_contribution': self.robustness_retention_contribution,
            'robustness_retention_mechanism': self.robustness_retention_mechanism,
            'pareto_layer': self.pareto_layer,
            'dominance_share': self.dominance_share,
            'bottleneck_score': self.bottleneck_score,
        }
        if self.trajectory_gain_last5 is not None:
            result['trajectory_gain_last5'] = self.trajectory_gain_last5
        return result
    
    def to_series(self) -> pd.Series:
        """Convert all metrics to a pandas Series."""
        return pd.Series(self.to_dict())


def compute_all_metrics(
    df: 'DataFrame',
    setup_name: str,
    mechanism: str,
    perturbation_dfs: Optional[Dict[str, 'DataFrame']] = None
) -> SetupMetrics:
    """
    Compute all metrics for a single setup.
    
    This is the main convenience function that computes all 8 metric types
    and returns them in a structured format.
    
    Args:
        df: DataFrame with game results for this setup
        setup_name: human-readable name for this setup
        mechanism: 'vcm', 'bccm', 'sccm', 'ccm', or 'ccf'
        perturbation_dfs: optional dict mapping perturbation names to DataFrames
            with results under those perturbations (for robustness metrics)
        
    Returns:
        SetupMetrics object containing all computed metrics
    """
    mechanism = mechanism.lower()
    has_conditional = mechanism in ['bccm', 'sccm', 'ccm', 'ccf']
    
    # 1. Payoff Efficiency
    gp_efficiency_df, setup_efficiency_dict = compute_payoff_efficiency(df)
    if setup_efficiency_dict and any(setup_name.lower() in key.lower() or any(name_part.lower() in key.lower() for name_part in setup_name.split()) for key in setup_efficiency_dict):
        E_s = [val for key, val in setup_efficiency_dict.items() if set(setup_name.split()).intersection(set(key.split('_')))]
        E_s = E_s[0] if E_s else np.nan
    elif not gp_efficiency_df.empty:
        E_s = gp_efficiency_df['payoff_efficiency'].mean() if 'payoff_efficiency' in gp_efficiency_df.columns else np.nan
    else:
        # Compute manually from group period data
        if 'normalized_welfare' in df.columns:
            E_s = df['normalized_welfare'].mean()
        elif 'payoff' in df.columns and 'chain_id' in df.columns and 'generation' in df.columns:
            # Manual computation using known formula
            n_players = RO_PLAYERS
            # Simplified: use normalized_welfare if available or compute from payoffs
            pi_min_per_group = RO_ENDOWMENT * n_players
            
            group_payoffs = df.groupby(['chain_id', 'generation'])['payoff'].sum()
            E_s = ((group_payoffs - pi_min_per_group) / (RO_MPCR * RO_ENDOWMENT * n_players * n_players - pi_min_per_group)).clip(0, 1).mean()
        else:
            E_s = np.nan
    
    if pd.isna(E_s):
        # Try alternate computation using normalized_welfare as proxy
        if 'normalized_welfare' in df.columns:
            E_s = df['normalized_welfare'].mean()
    
    # 2. Trajectory Gain
    eff_by_period = {}
    for gen in range(1, RO_PERIODS + 1):
        gen_data = df[df.get('generation', pd.Series([0] * len(df))) == gen]
        if not gen_data.empty and 'payoff' in gen_data.columns:
            # Compute period-level efficiency
            if len(gen_data) >= RO_PLAYERS:  # full group
                n_groups = len(gen_data.groupby('chain_id')) if 'chain_id' in gen_data else 1
            else:
                n_groups = 1
        else:
            continue
            
        period_eff = gen_data.get('normalized_welfare', gen_data['payoff'] if False else pd.Series([pd.NA] * len(gen_data))).mean()
        eff_by_period[gen] = float(period_eff) if not pd.isna(period_eff) and period_eff <= 1 and period_eff >= 0 else E_s
    
    if eff_by_period:
        G_s_base = compute_trajectory_gain(eff_by_period, use_last5=False)
        G_s_last5 = compute_trajectory_gain(eff_by_period, use_last5=True)
    else:
        # No period data or couldn't compute - use setup average as estimate for all periods
        G_s_base = 0.0
        G_s_last5 = None
    
    # Try alternative trajectory using contribution rates by period
    if pd.isna(G_s_base):
        CR_by_period = {}
        for gen in sorted(df['generation'].unique()) if 'generation' in df.columns else []:
            gen_df = df[df.get('generation') == gen]
            cr = (gen_df['contribution_share'].mean() * 100) if 'contribution_share' in gen_df.columns else np.nan
            CR_by_period[int(gen)] = float(cr) if not pd.isna(cr) else E_s * 100
        G_s_base = compute_trajectory_gain(CR_by_period) / 100
    
    # 3. Mechanism Use (only for conditional mechanisms)
    M_s = None
    if has_conditional:
        M_s = compute_mechanism_use(df, mechanism)
    
    # 4. Conditional Cooperation Slope
    beta_s = None
    try:
        beta_s = compute_conditional_cooperation_slope(df)
    except Exception as e:
        pass
    if pd.isna(beta_s):
        beta_s = None
    
    # 5. Robustness Retention (only with perturbation data)
    R_E, R_C, R_M = None, None, None
    if perturbation_dfs:
        baseline_metrics = {'E': E_s}
        try:
            if 'contribution_share' in df.columns:
                baseline_metrics['C'] = df['contribution_share'].mean()
            elif 'contributes' in df.columns:
                baseline_metrics['C'] = df['contributes'].astype(bool).astype(float).mean()
        except Exception as e:
            pass
        if has_conditional:
            baseline_metrics['M'] = M_s
        else:
            baseline_metrics['M'] = None
        
        perturbation_metrics = {}
        for pert_name, pert_df in perturbation_dfs.items():
            try:
                pert_eff = pert_df.get('normalized_welfare', pert_df.get('payoff', pd.Series([pd.NA] * len(pert_df)))).mean() if not pert_df.empty else np.nan
                pert_metrics = {'E': float(pert_eff) if not pd.isna(pert_eff) and 0 <= float(pert_eff) <= 1 else E_s}
                try:
                    pert_c = pert_df['contribution_share'].mean() if 'contribution_share' in pert_df.columns else None
                    if pert_c is not None and 'C' not in pert_metrics:
                        pert_metrics['C'] = float(pert_c)
                    elif pert_c is None and not pert_df.empty:
                        pert_metrics['C'] = baseline_metrics.get('C')
                except Exception as e:
                    pass
                try:
                    if has_conditional:
                        pert_m = compute_mechanism_use(pert_df, mechanism) or M_s or 0.0
                        pert_metrics['M'] = float(pert_m)
                except Exception as e:
                    pert_metrics.get('M', None)
                perturbation_metrics[pert_name] = pert_metrics
            except Exception as error_message:
                continue
        
        R_E, R_C, R_M = compute_robustness_retention(baseline_metrics, perturbation_metrics) or (None, None, None)
    
    # 6. Pareto and Bottleneck will be computed across all setups in a separate pass
    pareto_layer = None
    dominance_share = None
    bottleneck_score = None
    
    if not pd.isna(E_s):
        # Compute bottleneck score using available metrics
        bottleneck_score = compute_bottleneck_score(
            E_s=E_s,
            G_s=G_s_base,
            M_s=M_s,
            R_E_s=R_E if R_E is not None else E_s,  # Use efficiency as proxy for robustness if missing
            mechanism_has_conditional=has_conditional
        )
    
    return SetupMetrics(
        name=setup_name,
        payoff_efficiency=float(E_s) if not pd.isna(E_s) else np.nan,
        trajectory_gain=float(G_s_base) if not pd.isna(G_s_base) else 0.0,
        trajectory_gain_last5=float(G_s_last5) if G_s_last5 is not None and not pd.isna(G_s_last5) else None,
        mechanism_use=float(M_s) if M_s is not None and not pd.isna(M_s) else (None if has_conditional else np.nan),
        conditional_cooperation_slope=float(beta_s) if beta_s is not None and not pd.isna(beta_s) else None,
        robustness_retention_efficiency=float(R_E) if R_E is not None and not pd.isna(R_E) else None,
        robustness_retention_contribution=float(R_C) if R_C is not None and not pd.isna(R_C) else None,
        robustness_retention_mechanism=float(R_M) if R_M is not None and not pd.isna(R_M) else None,
        pareto_layer=pareto_layer,
        dominance_share=dominance_share,
        bottleneck_score=bottleneck_score
    )


def compute_metrics_for_all_setups(
    data_dict: Dict[str, Tuple['DataFrame', str]],
    mechanism_info: Optional[Dict[str, str]] = None,
    perturbation_data: Optional[Dict[str, Dict[str, 'DataFrame']]] = None
) -> List[SetupMetrics]:
    """
    Compute all metrics for multiple setups.
    
    Args:
        data_dict: dict mapping setup names to (DataFrame, mechanism_type)
        mechanism_info: optional dict providing mechanism type per setup if not in data_dict
        perturbation_data: nested dict mapping setup_name -> {perturbation_name -> DataFrame}
        
    Returns:
        List of SetupMetrics objects for all setups
    """
    results = []
    errors = 0
    
    mechanism_info = mechanism_info or {}
    perturbation_data = perturbation_data or {}
    
    for setup_name, (df, *rest_mechanism) in data_dict.items():
        mechanism = rest_mechanism[0] if rest_mechanism else mechanism_info.get(setup_name)
        if mechanism is None:
            # Infer from DataFrame
            mechanism_df = df.get('mechanism', ['unknown'])[0]
            mechanism = 'vcm' if 'vcm' in str(mechanism_df).lower() else ('ccm' if 'ccm' in mechanism_info_str(mechanism_df) else ('bccm' if any(x in ['mechanism', 'bccm'] for x in [str(df.get('mechanism')).lower()]) else ''))
            if isinstance(mechanism, pd.Series):
                mechanism = mechanism.iloc[0]
        
        perturbations = perturbation_data.get(setup_name, None)
        
        try:
            metrics = compute_all_metrics(df, setup_name, mechanism or 'vcm', perturbations)
            results.append(metrics)
        except Exception as e:
            errors += 1
            continue
    
    return results


def generate_overview_table(
    all_metrics: List[SetupMetrics],
    sort_by: Optional[str] = None,
) -> 'DataFrame':
    """
    Generate the recommended overview table.
    
    Sorted by:
    1. pareto_layer (ascending)
    2. dominance_share (descending)
    3. payoff_efficiency (descending)
    4. robustness_retention_efficiency (descending)
    
    Args:
        all_metrics: list of SetupMetrics objects
        sort_by: optional custom sorting (not implemented yet, uses default)
        
    Returns:
        DataFrame with the overview table
    """
    if not all_metrics:
        return pd.DataFrame()
    
    # First compute Pareto dominance across all setups
    setup_dicts = [{'name': sm.name, **sm.to_dict()} for sm in all_metrics]
    layer_assignment, dominance_share = compute_pareto_dominance(setup_dicts)
    
    # Update metrics with Pareto information
    updated_metrics = []
    for i, sm in enumerate(all_metrics):
        new_sm = SetupMetrics(**sm.to_dict())
        new_sm.pareto_layer = layer_assignment.get(sm.name)
        new_sm.dominance_share = dominance_share.get(sm.name)
        # Recompute bottleneck with correct values
        if sm.bottleneck_score is None and not pd.isna(new_sm.payoff_efficiency):
            has_conditional = (new_sm.mechanism_use is not None and not pd.isna(new_sm.mechanism_use))
            new_sm.bottleneck_score = compute_bottleneck_score(
                E_s=new_sm.payoff_efficiency,
                G_s=sm.trajectory_gain if not pd.isna(sm.trajectory_gain) else 0.0,
                M_s=new_sm.mechanism_use if has_conditional else None,
                R_E_s=new_sm.robustness_retention_efficiency if new_sm.robustness_retention_efficiency is not None and not pd.isna(new_sm.robustness_retention_efficiency) else sm.payoff_efficiency,
                mechanism_has_conditional=has_conditional
            )
        updated_metrics.append(new_sm)
    
    all_metrics = updated_metrics
    
    rows = []
    for sm in all_metrics:
        row = {
            'setup': sm.name,
            'pareto_layer': sm.pareto_layer if sm.pareto_layer is not None else 999,
            'dominance_share': round(sm.dominance_share, 3) if sm.dominance_share is not None and not pd.isna(sm.dominance_share) else np.nan,
            'payoff_efficiency': round(sm.payoff_efficiency, 3) if not pd.isna(sm.payoff_efficiency) else np.nan,
            'trajectory_gain': round(sm.trajectory_gain, 3) if not pd.isna(sm.trajectory_gain) else np.nan,
            'mechanism_use': round(sm.mechanism_use, 3) if sm.mechanism_use is not None and not pd.isna(sm.mechanism_use) else np.nan,
            'robustness_retention': round(sm.robustness_retention_efficiency, 3) if (sm.robustness_retention_efficiency is not None and not pd.isna(sm.robustness_retention_efficiency)) else np.nan,
            'bottleneck_score': round(sm.bottleneck_score, 3) if sm.bottleneck_score is not None and not pd.isna(sm.bottleneck_score) else np.nan,
        }
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    
    # Apply sorting
    result_df = result_df.sort_values(
        by=['pareto_layer', 'dominance_share', 'payoff_efficiency', 'robustness_retention'],
        ascending=[True, False, False, False],
        na_position='last'
    ).reset_index(drop=True)
    
    return result_df
