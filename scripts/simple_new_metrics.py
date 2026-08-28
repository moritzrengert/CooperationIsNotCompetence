#!/usr/bin/env python3
"""
Enhanced New Metric System - Complete Implementation
Computes all 8 metrics + Pareto ranking and bottleneck score.
"""

import json, os, re, sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.games.ro_public_good import RO_ENDOWMENT, RO_MPCR, RO_PLAYERS

# Constants
PI_MIN_TOTAL = RO_PLAYERS * RO_ENDOWMENT
PI_MAX_TOTAL = RO_PLAYERS * (RO_MPCR * RO_ENDOWMENT * RO_PLAYERS)


def group_period_cols(df):
    """Columns that identify one independent RO group-period."""
    if df.empty or 'generation' not in df.columns:
        return []
    cols = []
    for col in ['run_label', 'seed', 'repetition']:
        if col in df.columns:
            cols.append(col)
    if 'group_id' in df.columns:
        cols.append('group_id')
    elif 'chain_id' in df.columns:
        cols.append('chain_id')
    cols.append('generation')
    return list(dict.fromkeys(cols))


def group_identity_cols(df):
    return [col for col in group_period_cols(df) if col != 'generation']


TEXT_COLUMNS = {
    "raw_response",
    "reason",
    "thinking",
    "messages",
    "prompt",
    "reasoning_text",
    "reasoning_text_source",
}


def load_jsonl(path):
    """Load JSONL file to DataFrame."""
    rows = []
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows.append({key: value for key, value in row.items() if key not in TEXT_COLUMNS})
    except: pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def calc_payoff_efficiency(df):
    """Metric 1: E_s - average group-period contribution-rate efficiency."""
    if df.empty:
        return np.nan
    
    av = group_period_cols(df)
    if not av or 'generation' not in av:
        return np.nan

    if 'contribution_share' in df.columns:
        eff = pd.to_numeric(df['contribution_share'], errors='coerce').groupby([df[c] for c in av], dropna=False).mean()
    elif 'implemented_contribution' in df.columns:
        contribution = pd.to_numeric(df['implemented_contribution'], errors='coerce') / RO_ENDOWMENT
        eff = contribution.groupby([df[c] for c in av], dropna=False).mean()
    elif 'payoff' in df.columns:
        gp = df.groupby(av, dropna=False)['payoff'].sum()
        denominator = PI_MAX_TOTAL - PI_MIN_TOTAL
        if abs(denominator) < 1e-9:
            eff = pd.Series([0.5] * len(gp))
        else:
            eff = ((gp - PI_MIN_TOTAL) / denominator).clip(0, 1)
    else:
        return np.nan
    eff = pd.to_numeric(eff, errors='coerce').clip(0, 1)
    return float(eff.mean()) if not eff.empty else np.nan


def group_period_efficiency_by_generation(df):
    """Return generation -> mean group contribution-rate efficiency."""
    if df.empty or 'generation' not in df.columns:
        return {}
    group_cols = group_period_cols(df)
    if not group_cols:
        return {}
    work = df.copy()
    if 'contribution_share' in work.columns:
        work['_eff_source'] = pd.to_numeric(work['contribution_share'], errors='coerce').clip(0, 1)
    elif 'implemented_contribution' in work.columns:
        work['_eff_source'] = (pd.to_numeric(work['implemented_contribution'], errors='coerce') / RO_ENDOWMENT).clip(0, 1)
    else:
        gp = work.groupby(group_cols, dropna=False)['payoff'].sum().reset_index(name='total_payoff')
        denominator = PI_MAX_TOTAL - PI_MIN_TOTAL
        gp['_eff'] = ((gp['total_payoff'] - PI_MIN_TOTAL) / denominator).clip(0, 1)
        return gp.groupby('generation', dropna=False)['_eff'].mean().to_dict()
    gp = work.groupby(group_cols, dropna=False)['_eff_source'].mean().reset_index(name='_eff')
    return gp.groupby('generation', dropna=False)['_eff'].mean().to_dict()


def calc_trajectory_gain(df):
    """Metric 2: G_s - normalized improvement from first to last period."""
    if df.empty or 'generation' not in df.columns:
        return np.nan
    
    gen_eff = group_period_efficiency_by_generation(df)
    
    periods = sorted(gen_eff.keys())
    if len(periods) < 2: return np.nan
    
    e1, eT = gen_eff[periods[0]], gen_eff[periods[-1]]
    # Handle edge case where e1 is NaN or 1 (denominator zero)
    if pd.isna(e1) or pd.isna(eT): return np.nan
    denominator = 1.0 - e1
    if abs(denominator) < 1e-9: return 0.0  # Already at max, no room to improve
    return (eT - e1) / denominator

def calc_last5_trajectory_gain(df):
    """Metric 2b: G^{last5}_s - last-5 vs first-5 window comparison."""
    if df.empty or 'generation' not in df.columns:
        return np.nan
    
    gen_eff = group_period_efficiency_by_generation(df)
    
    periods = sorted(gen_eff.keys())
    if len(periods) < 10: return np.nan  # Need at least 10 periods for 5-period windows
    
    first5 = [gen_eff[p] for p in periods[:5]]
    last5 = [gen_eff[p] for p in periods[-5:]]
    if any(pd.isna(x) for x in first5 + last5): return np.nan
    
    e_first5 = np.mean(first5)
    e_last5 = np.mean(last5)
    denominator = 1.0 - e_first5
    if abs(denominator) < 1e-9: return 0.0
    return (e_last5 - e_first5) / denominator

def calc_mechanism_use(df, mech):
    """Metric 3: M_s - fraction of implemented cooperation from conditional mechanism use."""
    if df.empty or mech not in ['bccm', 'ccm', 'sccm', 'ccf']:
        return np.nan
    
    # Get contribution amounts (q)
    q = None
    q_temp = None
    for col in ['implemented_contribution', 'contribution_share', 'contributes']:
        if col in df.columns:
            if col == 'implemented_contribution':
                q_series_pd = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
                q_temp = q_series_pd
            elif col == 'contribution_share':
                q_temp = pd.to_numeric(df[col], errors='coerce').fillna(0)
            elif col == 'contributes':
                q_bool = df[col].astype(bool).astype(float) * RO_ENDOWMENT
                q_temp = q_bool
            break
    
    if q is None and q_temp is not None:
        q = q_temp
    else:
        q = q_temp or None
     
    if isinstance(q, type(None)):
        return np.nan
    
    total_q = float(q.sum())
    if abs(total_q) < 1e-9: return np.nan
    
    # Identify conditional choices based on mechanism  
    if mech == 'bccm':
        for col_candidate in ['choice', 'condition']:
            if col_candidate in df.columns:
                cond_series = pd.to_numeric(df[col_candidate], errors='coerce')
                break
        else: return np.nan
        
        # Use the condition column to create mask  
        conditional_mask = False
        for col_candidate in ['condition', 'choice']:
            if f'{col_candidate}' in df.columns:
                conditions = pd.to_numeric(df[col_candidate], errors='coerce')
                conditional_mask = conditions.between(1, RO_PLAYERS-1)
                break
    
    elif mech in {'ccm', 'sccm'}:
        # For CCM/SCCM: positive contribution AND positive threshold
        has_conditional = False
        slots = [1] if mech == 'sccm' else [1, 2]
        for slot in slots:
            cc_col = f'contribution_{slot}'
            ct_col = f'threshold_{slot}'
            if cc_col in df.columns and ct_col in df.columns:
                c_pos = pd.to_numeric(df[cc_col], errors='coerce') > 0
                t_pos = pd.to_numeric(df[ct_col], errors='coerce') > 0
                slot_conditional = c_pos & t_pos
                if not has_conditional or (slot == 1):
                    conditional_mask = slot_conditional
                else:
                    conditional_mask |= slot_conditional
                has_conditional = True
        

        if not has_conditional: return np.nan
    
    elif mech == 'ccf':
        conditional_mask = df.get('ccf_is_conditional', pd.Series(False, index=df.index)).fillna(False).astype(bool)
    
    # Apply mask to get weighted contribution from conditional use
    cond_q_sum = float((q * conditional_mask).sum())
    return cond_q_sum / total_q

def calc_cc_slope(df):
    if df.empty or 'generation' not in df.columns:
        return np.nan
    
    # Get contribution column
    for col in ['contribution_share', 'implemented_contribution', 'contributes']:
        if col not in df.columns: continue
        c = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if col == 'implemented_contribution': c /= RO_ENDOWMENT
        elif col == 'contributes': c = c.astype(float)
        break
    else: return np.nan
    
    work = df.copy()
    work['_c'] = pd.to_numeric(c, errors='coerce').fillna(0).clip(0, 1)
    work['_generation'] = pd.to_numeric(work['generation'], errors='coerce')
    group_cols = group_identity_cols(df)
    if not group_cols:
        return np.nan
    if 'player' not in work.columns:
        work['_player'] = np.arange(len(work))
        player_col = '_player'
    else:
        player_col = 'player'
    work[player_col] = pd.to_numeric(work[player_col], errors='coerce')

    prev_group = (
        work.groupby(group_cols + ['_generation'], dropna=False)['_c']
        .agg(prev_total='sum', prev_count='count')
        .reset_index()
    )
    prev_group['_generation'] = prev_group['_generation'] + 1
    prev_own = work[group_cols + ['_generation', player_col, '_c']].rename(columns={'_c': 'prev_own'})
    prev_own['_generation'] = prev_own['_generation'] + 1
    merged = work.merge(prev_group, on=group_cols + ['_generation'], how='left')
    merged = merged.merge(prev_own, on=group_cols + ['_generation', player_col], how='left')
    prev_means = (merged['prev_total'] - merged['prev_own']) / (merged['prev_count'] - 1)
    
    # Filter valid pairs
    valid = merged['_c'].notna() & pd.to_numeric(prev_means, errors='coerce').notna()
    cv, pv = merged.loc[valid, '_c'].values, pd.to_numeric(prev_means[valid], errors='coerce').values
    if len(cv) < 2: return np.nan
    
    cov_mx = np.cov(pv, cv)
    var = cov_mx[0,0]; cov = cov_mx[0,1]
    return float(cov/var) if var != 0 else 0.0

def selected_jsonl_files(out_dir):
    manifest_path = os.environ.get("LLM_PGG_SELECTED_MANIFEST")
    if not manifest_path:
        default_manifest = out_dir / "analysis" / "result_file_manifest.csv"
        if default_manifest.is_file():
            manifest_path = str(default_manifest)
    if not manifest_path:
        return sorted(list(out_dir.rglob("results.jsonl")))
    manifest = pd.read_csv(manifest_path)
    if "selected" in manifest.columns:
        manifest = manifest[manifest["selected"].astype(str).str.lower().isin(["true", "1", "yes"])]
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"].astype(str).eq("complete")]
    path_col = "results_path" if "results_path" in manifest.columns else "path"
    paths = [Path(p) for p in manifest[path_col].dropna().astype(str)]
    return paths


def load_and_analyze_setups(max_files=None):
    """Load all setups and compute metrics."""
    out_dir = Path(os.environ.get("LLM_PGG_OUTPUTS", ROOT / "outputs"))
    
    # First, load fixed-player robustness data to match baselines
    rob_df = pd.DataFrame()
    baselines = pd.DataFrame()
    rob_file = out_dir / "fixed_players" / "fixed_player_robustness_summary.csv"
    if rob_file.exists():
        try:
            rob_df = pd.read_csv(rob_file)
            # Extract baseline (fixed_count=0) data
            baselines = rob_df[rob_df['fixed_count'] == 0][['model', 'mechanism', 'prompt_modifier', 
                                                             'baseline_group_normalized_welfare']]
            baselines.columns = ['model', 'mechanism', 'prompt_modifier', 'baseline_efficiency']
        except:
            pass
    
    jsonl_files = selected_jsonl_files(out_dir)
    
    print(f"Found {len(jsonl_files)} selected JSONL files, processing up to {max_files or 'all'}")
    if max_files: jsonl_files = jsonl_files[:max_files]
    
    results = []
    for jf in jsonl_files:
        df = load_jsonl(jf)
        if len(df) < 20: continue
        
        # Extract metadata from columns
        mech = str(df.get('mechanism', ['unknown']).iloc[0]).lower()
        model = str(df.get('model', ['unknown']).iloc[0])
        prompt = str(df.get('prompt_modifier', ['none']).iloc[0])
        
        if mech not in ['bccm','ccm','sccm','ccf','vcm'] and 'bccm' not in mech and 'ccm' not in mech and 'sccm' not in mech and 'ccf' not in mech: continue
        
        # Normalize mechanism name  
        if 'bccm' in mech and 'ro_bccm' not in mech: mech = 'bccm'
        elif 'sccm' in mech or mech == 'ro_sccm': mech = 'sccm'
        elif 'ccm' in mech or mech == 'ro_ccm': mech = 'ccm'
        elif 'ccf' in mech or mech == 'ro_ccf': mech = 'ccf'
        elif 'ro_vcm' in mech or mech == 'vcm': mech = 'vcm'
        
        # Setup name components
        model_short = model.split(':')[0] if ':' in model else model
        parts = [p for p in [model_short, mech, prompt] if p not in ['unknown','none']]
        ts = jf.parent.name.split('_')[-1].replace('-','').replace(':','')[:6]
        setup_name = f"{'_'.join(parts)}_{ts}" if parts else f"setup_{ts}"
        
        # Compute metrics following the spec exactly
        eff = calc_payoff_efficiency(df)  # E_s
        
        # Trajectory gain: G_s = (E_T - E_1)/(1 - E_1) where E is payoff efficiency  
        traj = calc_trajectory_gain(df)
        traj_last5 = calc_last5_trajectory_gain(df)
        
        # Mechanism use for BCCM/CCM only
        meas = calc_mechanism_use(df, mech) if mech in ['bccm','ccm','sccm','ccf'] else np.nan
        
        cc_slope = calc_cc_slope(df)
        
        # Robustness retention: look up from fixed player data
        robust_ret = None
        if not baselines.empty and len(baselines) > 0:
            baseline_row = baselines[
                (baselines['model'] == model_short) & 
                (baselines['mechanism'] == mech) &
                (baselines['prompt_modifier'].astype(str).str.lower() == prompt if prompt != 'none' else baselines['prompt_modifier'].isna())
            ]
            if len(baseline_row) > 0 and 'robustness_retention' in baseline_row.columns:
                robust_ret = float(baseline_row['robustness_retention'].iloc[0])
        
        results.append({
            'setup': setup_name,
            'model': model, 'mechanism': mech, 'prompt_modifier': prompt,
            'payoff_efficiency': eff, 
            'trajectory_gain': traj if not pd.isna(traj) else 0.0,
            'trajectory_last5': traj_last5,
            'mechanism_use': meas, 
            'cc_slope': cc_slope,
            'robustness_retention': robust_ret if robust_ret is not None and not np.isnan(robust_ret) else None,
        })
    
    return results

def compute_pareto_rankings(df):
    """Compute Pareto layers and dominance share across setups."""
    if df.empty or len(df) < 2:
        return df.copy()
    
    # Metrics to consider for Pareto ranking (higher is better)
    metrics = ['payoff_efficiency', 'trajectory_gain', 'mechanism_use', 'robustness_retention']
    available_metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
    
    if len(available_metrics) < 2:
        # Just add placeholder columns
        df['pareto_layer'] = 1
        df['dominance_share'] = 0.0
        return df
    
    n = len(df)
    dominates = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            # Check if setup i dominates setup j
            all_ge = True
            any_gt = False
            
            for m in available_metrics:
                val_i = df.iloc[i][m]
                val_j = df.iloc[j][m]
                
                if pd.isna(val_i) or pd.isna(val_j):
                    continue
                
                if val_i < val_j - 1e-9:  # i is worse on this metric
                    all_ge = False
                    break
                elif val_i > val_j + 1e-9:  # i is better on this metric
                    any_gt = True
            
            dominates[i, j] = all_ge and any_gt
    
    # Compute dominance share for each setup
    dominance_share = []
    for i in range(n):
        n_dominated = dominates[i].sum()
        if n > 1:
            dominance_share.append(float(n_dominated / (n - 1)))
        else:
            dominance_share.append(0.0)
    
    # Assign Pareto layers using iterative removal
    remaining = set(range(n))
    layers = [None] * n
    current_layer = 1
    
    while remaining:
        # Find undominated setups in the remaining set
        non_dominated = []
        for i in remaining:
            is_dominated = False
            for j in remaining:
                if i != j and dominates[j, i]:
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(i)
        
        # Assign current layer to undominated setups
        for i in non_dominated:
            layers[i] = current_layer
        
        remaining -= set(non_dominated)
        current_layer += 1
    
    df['pareto_layer'] = layers
    df['dominance_share'] = dominance_share
    return df

def compute_bottleneck_score(row):
    """Compute bottleneck score: B_s = min(E_s, G^+_s, M_s, R^E_s)
    
    where G^+_s = max(0, G_s). Excludes non-applicable dimensions.
    """
    # Get E_s (required - can't compute without it)
    e_val = row.get('payoff_efficiency')
    if pd.isna(e_val):
        return np.nan
    
    # G^+ = max(0, trajectory_gain)  
    g_val = row.get('trajectory_gain', 0.0) or 0.0
    g_plus = max(0.0, float(g_val))
    
    components = [float(e_val), g_plus]
    
    # Add M_s only for conditional mechanisms (BCCM/CCM)  
    mech = row.get('mechanism', '')
    m_val = row.get('mechanism_use')
    if mech in ['bccm', 'ccm', 'sccm', 'ccf'] and not pd.isna(m_val):
        components.append(float(m_val))
    
    # Add R^E_s (robustness_retention) if available
    r_val = row.get('robustness_retention')
    if r_val is not None and not np.isnan(r_val):
        components.append(float(r_val))
    
    return min(components) if components else np.nan

def main():
    print("Loading data...")
    max_files_env = os.environ.get("LLM_PGG_MAX_METRIC_FILES")
    max_files = int(max_files_env) if max_files_env else None
    metrics_list = load_and_analyze_setups(max_files=max_files)
    
    if not metrics_list:
        print("No valid setups found!")
        return 1
    
    df = pd.DataFrame(metrics_list)
    n_valid = (~df['payoff_efficiency'].isna()).sum()
    print(f"Loaded {len(df)} setups, {n_valid} with valid efficiency scores")
    
    # Add Pareto rankings and dominance share
    print("\nComputing Pareto rankings...")
    df_with_pareto = compute_pareto_rankings(df)
    
    # Compute bottleneck score for each setup
    if 'payoff_efficiency' in df.columns:
        df_with_pareto['bottleneck_score'] = df_with_pareto.apply(compute_bottleneck_score, axis=1)
    
    # Sort by: pareto_layer ASC, dominance_share DESC, payoff_efficiency DESC, robustness_retention DESC (per spec)
    sort_order = ['pareto_layer', 'dominance_share', 'payoff_efficiency']
    if 'robustness_retention' in df_with_pareto.columns:
        sort_order.append('robustness_retention')
    ascending_order = [True, False, False] + ([False] if 'robustness_retention' in df_with_pareto.columns else [])
    
    df_sorted = df_with_pareto.sort_values(by=sort_order, ascending=ascending_order)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("NEW METRIC SYSTEM RESULTS")
    print("="*60)
    print(f"\nTotal setups: {len(df_sorted)}")
    
    # Pareto layer distribution
    if 'pareto_layer' in df_sorted.columns:
        layer_counts = df_sorted['pareto_layer'].value_counts().sort_index()
        print("\nPareto Layer Distribution:")
        for layer, count in layer_counts.items():
            print(f"  Layer {layer}: {count} setups")
    
    # Top setups by bottleneck score
    if 'bottleneck_score' in df_sorted.columns and df_sorted['bottleneck_score'].notna().any():
        top_bottleneck = df_sorted[df_sorted['bottleneck_score'].notna()].sort_values('bottleneck_score', ascending=False)['setup'].head(5).tolist()
        print(f"\nTop 5 by Bottleneck Score:")
        for i, setup_name in enumerate(top_bottleneck, 1):
            row = df_sorted[df_sorted['setup'] == setup_name].iloc[0]
            bn = f"{row['bottleneck_score']:.3f}" if not pd.isna(row.get('bottleneck_score')) else 'N/A'
            pe = f"{row['payoff_efficiency']:.3f}" if not pd.isna(row.get('payoff_efficiency')) else 'N/A'
            print(f"  {i}. {setup_name[:40]}: B={bn}, E={pe}")
    
    # Print top setups table
    print("\nTop 15 Setups (sorted by Pareto layer, then dominance share):")
    for _, row in df_sorted.head(15).iterrows():
        pe = f"{row['payoff_efficiency']:.3f}" if not pd.isna(row.get('payoff_efficiency')) else '---'
        tg = f"{row['trajectory_gain']:.3f}" if not pd.isna(row.get('trajectory_gain')) else '---'
        mu = f"{row['mechanism_use']:.3f}" if not pd.isna(row.get('mechanism_use')) else '---'
        pl = str(int(row['pareto_layer'])) if not pd.isna(row.get('pareto_layer')) else '--'
        ds = f"{row['dominance_share']:.3f}" if not pd.isna(row.get('dominance_share')) else '--'
        bn = f"{row['bottleneck_score']:.3f}" if 'bottleneck_score' in row and not pd.isna(row.get('bottleneck_score')) else '--'
        
        setup_short = row['setup'][:40] + ('...' if len(row['setup']) > 40 else '')
        print(f"  L{pl} | DS:{ds:5s} | E:{pe:>6} | G:{tg:>6} | M:{mu:>6} | BN:{bn:>5} : {setup_short}")
    
    # Save results with all metrics per spec
    out_path = Path(os.environ.get("LLM_PGG_OUTPUTS", ROOT / "outputs")) / "metrics_final"
    out_path.mkdir(exist_ok=True)
    
    # Overview table with columns in spec order:
    # setup, pareto_layer, dominance_share, payoff_efficiency, trajectory_gain, 
    #   mechanism_use, robustness_retention, bottleneck_score  
    output_cols = ['setup', 'pareto_layer', 'dominance_share', 'payoff_efficiency', 
                   'trajectory_gain', 'mechanism_use']
    if 'robustness_retention' in df_sorted.columns:
        output_cols.append('robustness_retention')
    output_cols.extend(['bottleneck_score', 'cc_slope'])
    
    # Filter to available columns
    output_df = df_sorted[[c for c in output_cols if c in df_sorted.columns]]
    output_df.to_csv(out_path / "overview_table.csv", index=False)
    
    # Also save detailed CSV
    df_to_save = df_with_pareto.dropna(how='all')  # Remove completely empty rows
    df_to_save.to_csv(out_path / "detailed_metrics.csv", index=False)
    
    print(f"\nSaved overview table to {out_path}/overview_table.csv")
    print(f"Saved detailed metrics to {out_path}/detailed_metrics.csv")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
