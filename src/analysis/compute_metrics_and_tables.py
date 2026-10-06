#!/usr/bin/env python
# coding: utf-8

# # Final manuscript figures and tables
# 
# This notebook produces the following copy-ready items from the frozen held-out test predictions:
# 
# 1. Two separate paper-style comparison PDFs and one vertically stacked PDF, with no embedded captions.
# 2. A revised Table II containing micro/macro-AUC, micro/macro-F1, pooled/macro-ECE, MAE, RMSE, patient-bootstrap 95\% confidence intervals, and the original Hyperband score.
# 3. Separate GRU-DSIMB and XGBoost-DSI tables with per-scan-type and stage-trajectory subgroup results.
# 
# Bootstrap calculations are cached incrementally in the aggregate output directory. Matching cached rows are reused automatically; changing the prediction source, bootstrap count, seed, or cache version triggers only the affected recalculation.

# In[ ]:


import os
from pathlib import Path
import ast
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
try:
    from IPython.display import display, Markdown
except ImportError:
    def display(obj):
        print(obj)

    def Markdown(value):
        return value
from sklearn.metrics import (
    f1_score, mean_absolute_error, mean_squared_error, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 180)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
restricted_root = Path(
    os.environ.get('RESTRICTED_ARTIFACT_ROOT', PROJECT_ROOT / 'restricted_artifacts')
).expanduser().resolve()
default_results = restricted_root / 'results'
default_prepared = PROJECT_ROOT / 'data' / 'prepared'
FROZEN_RESULTS = Path(
    os.environ.get('MODEL_RESULTS_DIR', default_results)
).expanduser().resolve()
FROZEN_MULTITASK = Path(
    os.environ.get('MULTITASK_DATA_DIR', default_prepared / 'multitask')
).expanduser().resolve()
FROZEN_TABULAR = Path(
    os.environ.get('MULTITASK_TABULAR_DATA_DIR', default_prepared / 'multitask_tabular')
).expanduser().resolve()
TARGET_PATH = FROZEN_TABULAR / 'target.csv'
GUIDELINE_DIR = FROZEN_RESULTS / 'guideline_matched_comparison'
AVERAGE_DIR = FROZEN_RESULTS / 'average_matched_comparison'
OUTPUT_DIR = Path(
    os.environ.get('AGGREGATE_OUTPUT_DIR', PROJECT_ROOT / 'aggregate_results')
).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BOOTSTRAP_CACHE_PATH = OUTPUT_DIR / 'bootstrap_confidence_interval_cache.csv'
TABLE_II_CACHE_PATH = OUTPUT_DIR / 'table_ii_metrics.csv'
SUBGROUP_CACHE_PATH = OUTPUT_DIR / 'table_iii_subgroups.csv'

SCAN_TYPES = ['CT', 'MRI', 'PET', 'ULTRASOUND', 'XRAY', 'MAMMO_TOMO', 'NUCLEAR_MED']
SCAN_LABELS = {
    'CT': 'CT', 'MRI': 'MRI', 'PET': 'PET', 'ULTRASOUND': 'Ultrasound',
    'XRAY': 'X-ray', 'MAMMO_TOMO': 'Mammo', 'NUCLEAR_MED': 'Nuclear medicine',
}
FEATURE_SETS = ['DS', 'DSI', 'DSM', 'DSB', 'DSIM', 'DSIB', 'DSMB', 'DSIMB']
N_ECE_BINS = 10
N_BOOTSTRAP = int(os.environ.get('N_BOOTSTRAP', '1000'))
BOOTSTRAP_SEED = 42
CI_PERCENTILES = (2.5, 97.5)
CACHE_VERSION = 'patient-bootstrap-v2-micro-macro-f1-ece'

print('Frozen results:', FROZEN_RESULTS)
print('Figures and reusable calculation files will be saved in:', OUTPUT_DIR)


# ## 1. Integrated guideline comparison
# 
# `Nearest guideline` is the original oracle-selected ultrasound baseline that selects the closest of 6, 12, and 24 months using the observed target interval. It is displayed for transparency but is not a leakage-free comparator.

# In[22]:


DETAIL_PATH = GUIDELINE_DIR / 'matched_guideline_detail_GRU-DSIMB.csv'
SUMMARY_PATH = GUIDELINE_DIR / 'matched_guideline_summary_wide_GRU-DSIMB.csv'
for path in [DETAIL_PATH, SUMMARY_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

guideline_detail = pd.read_csv(DETAIL_PATH, dtype={'patient_id': 'string'})
saved_guideline_summary = pd.read_csv(SUMMARY_PATH)
for col in ['true_interval_months', 'guideline_pred_interval_months', 'model_pred_interval_months']:
    guideline_detail[col] = pd.to_numeric(guideline_detail[col], errors='raise')

def summarize_interval_errors(part, baseline_prediction=None):
    true_value = part['true_interval_months'].to_numpy(float)
    model_value = part['model_pred_interval_months'].to_numpy(float)
    if baseline_prediction is None:
        baseline_value = part['guideline_pred_interval_months'].to_numpy(float)
    else:
        baseline_value = np.full(len(part), float(baseline_prediction))
    return {
        'n': int(len(part)),
        'baseline_mae': float(np.mean(np.abs(baseline_value - true_value))),
        'model_mae': float(np.mean(np.abs(model_value - true_value))),
        'baseline_rmse': float(np.sqrt(np.mean(np.square(baseline_value - true_value)))),
        'model_rmse': float(np.sqrt(np.mean(np.square(model_value - true_value)))),
    }

old_rows = []
for transition, part in guideline_detail.groupby('transition', sort=False):
    old_rows.append({'transition': transition, **summarize_interval_errors(part)})
old_rows.append({'transition': 'Overall', **summarize_interval_errors(guideline_detail)})
original_summary = pd.DataFrame(old_rows)

check = original_summary.rename(columns={
    'baseline_mae': 'guideline_mae', 'baseline_rmse': 'guideline_rmse'
}).merge(saved_guideline_summary, on=['transition', 'n'], suffixes=('_new', '_saved'))
for metric in ['guideline_mae', 'model_mae', 'guideline_rmse', 'model_rmse']:
    if not np.allclose(check[f'{metric}_new'], check[f'{metric}_saved'], atol=1e-12, rtol=0):
        raise AssertionError(f'Original Figure 3 reconstruction failed for {metric}.')

US_RULE = 'ULTRASOUND_to_ULTRASOUND'
us_rows = guideline_detail.loc[guideline_detail['guideline_rule'].eq(US_RULE)].copy()
if len(us_rows) != 330:
    raise AssertionError(f'Expected 330 ultrasound-to-ultrasound events, found {len(us_rows):,}.')
fixed_us = {months: summarize_interval_errors(us_rows, months) for months in [6, 12, 24]}

transition_order = [
    'MRI -> MRI', 'MRI -> MAMMO_TOMO', 'MAMMO_TOMO -> MRI',
    'MAMMO_TOMO -> MAMMO_TOMO', 'ULTRASOUND -> ULTRASOUND', 'Overall',
]
transition_labels = {
    'MRI -> MRI': 'MRI\n→ MRI',
    'MRI -> MAMMO_TOMO': 'MRI\n→ Mammo',
    'MAMMO_TOMO -> MRI': 'Mammo\n→ MRI',
    'MAMMO_TOMO -> MAMMO_TOMO': 'Mammo\n→ Mammo',
    'ULTRASOUND -> ULTRASOUND': 'Ultrasound\n→ Ultrasound',
    'Overall': 'Overall',
}
fixed_policy = {
    'MRI -> MRI': '12-month guideline',
    'MRI -> MAMMO_TOMO': '6-month guideline',
    'MAMMO_TOMO -> MRI': '6-month guideline',
    'MAMMO_TOMO -> MAMMO_TOMO': '12-month guideline',
}

plot_rows = []
for transition in transition_order:
    row = original_summary.loc[original_summary['transition'].eq(transition)].iloc[0]
    points = {
        'GRU-DSIMB': {'mae': float(row['model_mae']), 'rmse': float(row['model_rmse'])}
    }
    if transition in fixed_policy:
        points[fixed_policy[transition]] = {
            'mae': float(row['baseline_mae']), 'rmse': float(row['baseline_rmse'])
        }
    elif transition == 'ULTRASOUND -> ULTRASOUND':
        points['Nearest guideline'] = {
            'mae': float(row['baseline_mae']), 'rmse': float(row['baseline_rmse'])
        }
        for months in [6, 12, 24]:
            points[f'{months}-month guideline'] = {
                'mae': fixed_us[months]['baseline_mae'],
                'rmse': fixed_us[months]['baseline_rmse'],
            }
    else:
        points['Nearest guideline'] = {
            'mae': float(row['baseline_mae']), 'rmse': float(row['baseline_rmse'])
        }
    plot_rows.append({'transition': transition, 'n': int(row['n']), 'points': points})

guideline_export_rows = []
for _, row in original_summary.iterrows():
    guideline_export_rows.extend([
        {
            'transition': row['transition'], 'model': 'Guideline', 'n': int(row['n']),
            'MAE': row['baseline_mae'], 'RMSE': row['baseline_rmse'],
        },
        {
            'transition': row['transition'], 'model': 'GRU-DSIMB (ours)', 'n': int(row['n']),
            'MAE': row['model_mae'], 'RMSE': row['model_rmse'],
        },
    ])
for months in [6, 12, 24]:
    guideline_export_rows.append({
        'transition': 'ULTRASOUND -> ULTRASOUND',
        'model': f'{months}-month guideline',
        'n': len(us_rows),
        'MAE': fixed_us[months]['baseline_mae'],
        'RMSE': fixed_us[months]['baseline_rmse'],
    })
pd.DataFrame(guideline_export_rows).to_csv(
    OUTPUT_DIR / 'guideline_comparison.csv', index=False
)

display(pd.DataFrame([
    {
        'Ultrasound comparator': name,
        'MAE': values['mae'], 'RMSE': values['rmse'],
    }
    for name, values in plot_rows[4]['points'].items()
]).style.format({'MAE': '{:.4f}', 'RMSE': '{:.4f}'}))


# In[45]:


SERIES_STYLE = {
    'GRU-DSIMB': {'color': '#1b9e77', 'marker': 'o'},
    'Nearest guideline': {'color': '#7570b3', 'marker': 'D'},
    'Overall guideline': {'color': '#4d4d4d', 'marker': 'X'},
    '6-month guideline': {'color': '#d95f02', 'marker': 's'},
    '12-month guideline': {'color': '#1f78b4', 'marker': '^'},
    '24-month guideline': {'color': '#e31a1c', 'marker': 'P'},
    'Average': {'color': '#e7298a', 'marker': 'v'},
}

LABEL_OFFSET = {
    'GRU-DSIMB': -0.20,
    'Nearest guideline': 0.21,
    'Overall guideline': 0.21,
    '6-month guideline': -0.20,
    '12-month guideline': 0.21,
    '24-month guideline': -0.20,
}


# Give the overall baseline its own name, color, and marker.
guideline_plot_rows = []
for row in plot_rows:
    points = {
        name: values.copy()
        for name, values in row['points'].items()
    }

    if row['transition'] == 'Overall' and 'Nearest guideline' in points:
        points['Overall guideline'] = points.pop('Nearest guideline')

    guideline_plot_rows.append({
        'transition': row['transition'],
        'n': row['n'],
        'points': points,
    })


# Prepare the matched-average rows.
AVERAGE_SUMMARY_PATH = (
    AVERAGE_DIR / 'matched_average_summary_wide_GRU-DSIMB.csv'
)
if not AVERAGE_SUMMARY_PATH.exists():
    raise FileNotFoundError(AVERAGE_SUMMARY_PATH)

average_summary = pd.read_csv(AVERAGE_SUMMARY_PATH)

average_rows = []
for _, row in average_summary.iterrows():
    average_rows.append({
        'transition': str(row['scan_type']),
        'n': int(row['n']),
        'points': {
            'GRU-DSIMB': {
                'mae': float(row['model_mae']),
                'rmse': float(row['model_rmse']),
            },
            'Average': {
                'mae': float(row['average_mae']),
                'rmse': float(row['average_rmse']),
            },
        },
    })

average_export_rows = []
for row in average_rows:
    average_export_rows.extend([
        {
            'scan_type': row['transition'], 'model': 'Average baseline', 'n': row['n'],
            'MAE': row['points']['Average']['mae'], 'RMSE': row['points']['Average']['rmse'],
        },
        {
            'scan_type': row['transition'], 'model': 'GRU-DSIMB (ours)', 'n': row['n'],
            'MAE': row['points']['GRU-DSIMB']['mae'], 'RMSE': row['points']['GRU-DSIMB']['rmse'],
        },
    ])
pd.DataFrame(average_export_rows).to_csv(
    OUTPUT_DIR / 'average_comparison.csv', index=False
)


def row_axis_label(row, is_guideline):
    if row['transition'] == 'Overall':
        return f"Overall\n(n={row['n']:,})"

    if is_guideline:
        previous_scan, next_scan = row['transition'].split(' -> ')
        base = (
            f"{SCAN_LABELS.get(previous_scan, previous_scan)}\n"
            f"→ {SCAN_LABELS.get(next_scan, next_scan)}"
        )
    else:
        base = SCAN_LABELS.get(
            row['transition'], row['transition']
        )

    return f"{base}\n(n={row['n']:,})"


def draw_comparison_panel(
    ax,
    rows,
    metric,
    xlabel,
    is_guideline,
    show_ylabels=True,
):
    values = [
        point[metric]
        for row in rows
        for point in row['points'].values()
    ]

    value_range = max(values) - min(values)
    right_margin = max(0.15 * value_range, 0.65)
    left_margin = max(0.18 * value_range, 0.8)

    ax.set_xlim(
        max(0, min(values) - left_margin),
        max(values) + right_margin,
    )

    label_dx = max(
        0.018 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
        0.05,
    )

    for y, row in enumerate(rows):
        row_values = [
            point[metric]
            for point in row['points'].values()
        ]

        ax.plot(
            [min(row_values), max(row_values)],
            [y, y],
            color='0.38',
            linewidth=2.2,
            zorder=1,
        )

        ordered_points = list(row['points'].items())

        for point_index, (name, point) in enumerate(ordered_points):
            style = SERIES_STYLE[name]
            value = point[metric]

            ax.scatter(
                value,
                y,
                s=42,
                marker=style['marker'],
                facecolor=style['color'],
                edgecolor='black',
                linewidth=0.7,
                zorder=3,
            )

            x_min, x_max = ax.get_xlim()
            x_range = x_max - x_min

            # Labels within this distance from the y-axis are placed on the right.
            left_clearance = 0.14 * x_range

            near_left_axis = value < x_min + left_clearance
            near_right_axis = value > x_min + 0.88 * x_range

            place_left = (
                (name == 'GRU-DSIMB' and not near_left_axis)
                or near_right_axis
            )

            offset_y = (
                -0.24 if point_index % 2 == 0 else 0.25
            )

            # Separate labels in the MRI-to-Mammo row.
            if row['transition'] in {
                'ULTRASOUND -> ULTRASOUND',
                'MRI -> MRI',
                'Overall',
            }:
                offset_y = LABEL_OFFSET.get(name, offset_y)

            # Apply this afterward so it cannot be overwritten.
            if row['transition'] == 'MRI -> MAMMO_TOMO':
                if name == 'GRU-DSIMB':
                    offset_y = -0.32
                elif name == '6-month guideline':
                    offset_y = 0.32

            text_x = (
                value - label_dx
                if place_left
                else value + label_dx
            )
            text_ha = 'right' if place_left else 'left'

            if is_guideline and name == 'GRU-DSIMB':

                if row['transition'] == 'MRI -> MRI':
                    text_x = value + 1.2
                    text_ha = 'left'

                elif row['transition'] == 'MRI -> MAMMO_TOMO':
                    text_x = value - 0.9
                    text_ha = 'left'

                elif row['transition'] == 'ULTRASOUND -> ULTRASOUND':
                    # Move the green label right enough to clear the y-axis.
                    text_x = value + 0.7
                    text_ha = 'right'

            ax.text(
                text_x,
                y + offset_y,
                f'{value:.4f}',
                color=style['color'],
                fontsize=10,
                ha=text_ha,
                va='center',
            )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_yticks(np.arange(len(rows)))

    if show_ylabels:
        ax.set_yticklabels(
            [
                row_axis_label(row, is_guideline)
                for row in rows
            ],
            fontsize=9.1,
        )
        ax.get_yticklabels()[-1].set_fontweight('bold')
    else:
        ax.tick_params(axis='y', labelleft=False)

    ax.tick_params(axis='x', labelsize=8.6)
    ax.spines[['top', 'right']].set_visible(False)


def legend_handles(names):
    return [
        Line2D(
            [0],
            [0],
            linestyle='none',
            marker=SERIES_STYLE[name]['marker'],
            markersize=6,
            markerfacecolor=SERIES_STYLE[name]['color'],
            markeredgecolor='black',
            label=name,
        )
        for name in names
    ]


GUIDELINE_LEGEND_NAMES = [
    'GRU-DSIMB',
    'Nearest guideline',
    'Overall guideline',
    '6-month guideline',
    '12-month guideline',
    '24-month guideline',
]

AVERAGE_LEGEND_NAMES = [
    'GRU-DSIMB',
    'Average',
]

COMBINED_LEGEND_NAMES = (
    GUIDELINE_LEGEND_NAMES + ['Average']
)


def make_separate_figure(rows, is_guideline):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.8),
        sharey=True,
    )

    draw_comparison_panel(
        axes[0],
        rows,
        'mae',
        'MAE (months)',
        is_guideline,
        True,
    )
    draw_comparison_panel(
        axes[1],
        rows,
        'rmse',
        'RMSE (months)',
        is_guideline,
        False,
    )

    names = (
        GUIDELINE_LEGEND_NAMES
        if is_guideline
        else AVERAGE_LEGEND_NAMES
    )

    fig.legend(
        handles=legend_handles(names),
        loc='upper center',
        ncol=3 if is_guideline else 2,
        frameon=False,
        fontsize=7.7,
        columnspacing=1.0,
        handletextpad=0.3,
    )

    fig.tight_layout(pad=0.25, w_pad=0.8)
    fig.subplots_adjust(
        left=0.16,
        right=0.995,
        bottom=0.11,
        top=0.82 if is_guideline else 0.84,
    )

    return fig


def make_vertical_figure(guideline_rows, mean_rows):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 9.0),
    )

    draw_comparison_panel(
        axes[0, 0],
        guideline_rows,
        'mae',
        'MAE (months)',
        True,
        True,
    )
    draw_comparison_panel(
        axes[0, 1],
        guideline_rows,
        'rmse',
        'RMSE (months)',
        True,
        False,
    )
    draw_comparison_panel(
        axes[1, 0],
        mean_rows,
        'mae',
        'MAE (months)',
        False,
        True,
    )
    draw_comparison_panel(
        axes[1, 1],
        mean_rows,
        'rmse',
        'RMSE (months)',
        False,
        False,
    )

    fig.legend(
        handles=legend_handles(COMBINED_LEGEND_NAMES),
        loc='upper center',
        ncol=4,
        frameon=False,
        fontsize=8.0,
        columnspacing=1.1,
        handletextpad=0.3,
    )

    fig.tight_layout(
        pad=0.3,
        h_pad=1.0,
        w_pad=0.8,
    )
    fig.subplots_adjust(
        left=0.16,
        right=0.995,
        bottom=0.06,
        top=0.89,
        hspace=0.28,
    )

    return fig


guideline_separate_figure = make_separate_figure(
    guideline_plot_rows,
    True,
)
average_separate_figure = make_separate_figure(
    average_rows,
    False,
)
guideline_average_vertical_figure = make_vertical_figure(
    guideline_plot_rows,
    average_rows,
)


FIGURE_OUTPUTS = {
    OUTPUT_DIR / 'guideline_comparison_separate.pdf':
        guideline_separate_figure,
    OUTPUT_DIR / 'average_comparison_separate.pdf':
        average_separate_figure,
    OUTPUT_DIR / 'guideline_average_vertical.pdf':
        guideline_average_vertical_figure,
}

for output_path, figure in FIGURE_OUTPUTS.items():
    figure.savefig(
        output_path,
        format='pdf',
        bbox_inches='tight',
    )
    print(f'Saved: {output_path}')

plt.show()


# ## 2. Revised Table II
# 
# Micro-F1 pools the seven binary scan decisions, while macro-F1 averages the seven scan-specific F1 values. Pooled ECE flattens all labels and probabilities; macro-ECE averages seven scan-specific 10-bin ECE values. Saved validation thresholds are used for F1. All test metrics except the original Hyperband score include patient-level bootstrap 95\% confidence intervals.

# In[21]:


if not TARGET_PATH.exists():
    raise FileNotFoundError(TARGET_PATH)
target = pd.read_csv(TARGET_PATH, dtype={'patient_id': 'string'})
target['patient_id'] = target['patient_id'].astype(str)
all_target_patients = sorted(target['patient_id'].dropna().unique())
_, temporary_patients = train_test_split(
    all_target_patients, test_size=0.30, random_state=BOOTSTRAP_SEED
)
_, test_patients = train_test_split(
    temporary_patients, test_size=0.50, random_state=BOOTSTRAP_SEED
)
test_patient_set = set(test_patients)
test_map = target[target['patient_id'].isin(test_patient_set)].reset_index(drop=True).copy()
TEST_PATIENT_IDS = test_map['patient_id'].astype(str).to_numpy()
TEST_TRUE_SCAN = np.vstack(
    test_map['target_scan_vector'].apply(lambda value: np.asarray(ast.literal_eval(str(value)), dtype=int))
)
if len(test_map) != 28_830 or pd.Series(TEST_PATIENT_IDS).nunique() != 4_713:
    raise AssertionError('Frozen held-out test mapping could not be reconstructed.')

MODEL_SPECS = {
    'Linear': {
        'glob': 'lr_tabular_multitask_compare_next_scan_data_hyperband_normalized_score_*_{setting}_pid*',
        'detail': 'prediction_detail_lr_{setting}.csv', 'schema': 'tabular',
    },
    'XGBoost': {
        'glob': 'xgboost_tabular_multitask_compare_next_scan_data_hyperband_normalized_score_*_{setting}_pid*',
        'detail': 'prediction_detail_xgboost_{setting}.csv', 'schema': 'tabular',
    },
    'MLP': {
        'glob': 'mlp_tabular_multitask_compare_next_scan_data_hyperband_normalized_score_*_{setting}_pid*',
        'detail': 'prediction_detail_mlp_{setting}.csv', 'schema': 'tabular',
    },
    'GRU': {
        'glob': 'gru_tree_multitask_compare_next_scan_data_hyperband_normalized_score_*_{setting}_pid*',
        'detail': 'prediction_detail_{setting}.csv', 'schema': 'sequence',
    },
    'LSTM': {
        'glob': 'lstm_tree_multitask_compare_next_scan_data_hyperband_normalized_score_*_{setting}_pid*',
        'detail': 'prediction_detail_{setting}.csv', 'schema': 'sequence',
    },
    'Decoder-only Transformer': {
        'glob': 'decoder_only_transformer_multitask_compare_next_scan_data_hyperband_normalized_score_*_{setting}_pid*',
        'detail': 'prediction_detail_{setting}.csv', 'schema': 'sequence',
    },
}

def locate_run(model, setting):
    spec = MODEL_SPECS[model]
    matches = sorted(
        set(FROZEN_RESULTS.glob(spec['glob'].format(setting=setting)))
        | set(FROZEN_RESULTS.glob(spec['glob'].format(setting='ALL'))),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in matches:
        detail_path = run_dir / spec['detail'].format(setting=setting)
        summary_path = run_dir / 'experiment_metrics_summary.csv'
        if detail_path.is_file() and summary_path.is_file():
            return run_dir, detail_path, summary_path
    raise FileNotFoundError(f'No completed run found for {model}/{setting}.')

def expected_calibration_error(y_true, y_prob, n_bins=N_ECE_BINS, sample_weight=None):
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if sample_weight is None:
        sample_weight = np.ones(len(y_true), dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for index in range(n_bins):
        mask = bin_id == index
        if mask.any():
            bin_weight = sample_weight[mask].sum()
            if bin_weight > 0:
                observed = np.average(y_true[mask], weights=sample_weight[mask])
                predicted = np.average(y_prob[mask], weights=sample_weight[mask])
                ece += bin_weight / sample_weight.sum() * abs(observed - predicted)
    return float(ece)

def safe_auc(y_true, y_prob, sample_weight=None):
    y_true = np.asarray(y_true)
    if sample_weight is None:
        active = np.ones(len(y_true), dtype=bool)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        active = sample_weight > 0
    return (
        float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight))
        if np.unique(y_true[active]).size == 2 else np.nan
    )

def calculate_multitask_metrics(
    y_scan, p_scan, pred_scan, true_interval, pred_interval, sample_weight=None,
):
    if sample_weight is None:
        row_weight = np.ones(len(y_scan), dtype=float)
        metric_weight = None
    else:
        row_weight = np.asarray(sample_weight, dtype=float)
        metric_weight = row_weight
    pooled_weight = np.repeat(row_weight, y_scan.shape[1])
    per_type_auc = [
        safe_auc(y_scan[:, i], p_scan[:, i], metric_weight) for i in range(y_scan.shape[1])
    ]
    valid_auc = [value for value in per_type_auc if np.isfinite(value)]
    per_type_f1 = [
        float(f1_score(
            y_scan[:, i], pred_scan[:, i], zero_division=0, sample_weight=metric_weight
        ))
        for i in range(y_scan.shape[1])
    ]
    per_type_ece = [
        expected_calibration_error(
            y_scan[:, i], p_scan[:, i], sample_weight=metric_weight
        )
        for i in range(y_scan.shape[1])
    ]
    return {
        'scan_micro_auc': safe_auc(y_scan.ravel(), p_scan.ravel(), pooled_weight),
        'scan_macro_auc': float(np.mean(valid_auc)) if valid_auc else np.nan,
        'scan_micro_f1': float(f1_score(
            y_scan.ravel(), pred_scan.ravel(), zero_division=0, sample_weight=pooled_weight
        )),
        'scan_macro_f1': float(np.mean(per_type_f1)),
        'scan_pooled_ece': expected_calibration_error(
            y_scan, p_scan, sample_weight=pooled_weight
        ),
        'scan_macro_ece': float(np.mean(per_type_ece)),
        'interval_mae': float(mean_absolute_error(
            true_interval, pred_interval, sample_weight=metric_weight
        )),
        'interval_rmse': float(np.sqrt(mean_squared_error(
            true_interval, pred_interval, sample_weight=metric_weight
        ))),
        'valid_scan_types_for_macro_auc': len(valid_auc),
    }

CI_METRICS = [
    'scan_micro_auc', 'scan_macro_auc', 'scan_micro_f1', 'scan_macro_f1',
    'scan_pooled_ece', 'scan_macro_ece', 'interval_mae', 'interval_rmse',
]

def patient_bootstrap_confidence_intervals(
    patient_id, y_scan, p_scan, pred_scan, true_interval, pred_interval,
    n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
):
    patient_id = np.asarray(patient_id).astype(str)
    unique_patients, patient_code = np.unique(patient_id, return_inverse=True)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in CI_METRICS}
    for _ in range(n_bootstrap):
        sampled_codes = rng.integers(0, len(unique_patients), size=len(unique_patients))
        patient_weight = np.bincount(sampled_codes, minlength=len(unique_patients))
        row_weight = patient_weight[patient_code]
        metrics = calculate_multitask_metrics(
            y_scan, p_scan, pred_scan, true_interval, pred_interval, sample_weight=row_weight,
        )
        for metric in CI_METRICS:
            if np.isfinite(metrics[metric]):
                samples[metric].append(metrics[metric])
    intervals = {}
    for metric, values in samples.items():
        if not values:
            intervals[f'{metric}_ci_low'] = np.nan
            intervals[f'{metric}_ci_high'] = np.nan
        else:
            low, high = np.percentile(values, CI_PERCENTILES)
            intervals[f'{metric}_ci_low'] = float(low)
            intervals[f'{metric}_ci_high'] = float(high)
    return intervals

CACHE_VALUE_COLUMNS = CI_METRICS + [
    f'{metric}_ci_{bound}' for metric in CI_METRICS for bound in ['low', 'high']
] + ['valid_scan_types_for_macro_auc']
CACHE_KEY_COLUMNS = [
    'cache_version', 'cache_key', 'source_signature', 'n_bootstrap', 'bootstrap_seed'
]

def load_bootstrap_cache():
    if not BOOTSTRAP_CACHE_PATH.exists():
        return pd.DataFrame(columns=CACHE_KEY_COLUMNS + CACHE_VALUE_COLUMNS)
    cache = pd.read_csv(BOOTSTRAP_CACHE_PATH)
    required = set(CACHE_KEY_COLUMNS + CACHE_VALUE_COLUMNS)
    if not required.issubset(cache.columns):
        print('Ignoring an incompatible bootstrap cache; a new cache will be created.')
        return pd.DataFrame(columns=CACHE_KEY_COLUMNS + CACHE_VALUE_COLUMNS)
    return cache

BOOTSTRAP_CACHE = load_bootstrap_cache()
CACHE_HITS = 0
CACHE_MISSES = 0

def prediction_source_signature(detail_path):
    detail_path = Path(detail_path)
    stat = detail_path.stat()
    return f'{detail_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}'

def cached_metrics_with_ci(cache_key, source_signature, seed, patient_id, y_scan, p_scan,
                           pred_scan, true_interval, pred_interval):
    global BOOTSTRAP_CACHE, CACHE_HITS, CACHE_MISSES
    match = BOOTSTRAP_CACHE[
        BOOTSTRAP_CACHE['cache_version'].astype(str).eq(CACHE_VERSION)
        & BOOTSTRAP_CACHE['cache_key'].astype(str).eq(str(cache_key))
        & BOOTSTRAP_CACHE['source_signature'].astype(str).eq(str(source_signature))
        & pd.to_numeric(BOOTSTRAP_CACHE['n_bootstrap'], errors='coerce').eq(N_BOOTSTRAP)
        & pd.to_numeric(BOOTSTRAP_CACHE['bootstrap_seed'], errors='coerce').eq(seed)
    ]
    if not match.empty:
        CACHE_HITS += 1
        saved = match.iloc[-1]
        return {column: float(saved[column]) for column in CACHE_VALUE_COLUMNS}

    CACHE_MISSES += 1
    metrics = calculate_multitask_metrics(
        y_scan, p_scan, pred_scan, true_interval, pred_interval
    )
    intervals = patient_bootstrap_confidence_intervals(
        patient_id, y_scan, p_scan, pred_scan, true_interval, pred_interval,
        n_bootstrap=N_BOOTSTRAP, seed=seed,
    )
    result = {**metrics, **intervals}
    cache_row = {
        'cache_version': CACHE_VERSION, 'cache_key': cache_key,
        'source_signature': source_signature, 'n_bootstrap': N_BOOTSTRAP,
        'bootstrap_seed': seed,
        **{column: result[column] for column in CACHE_VALUE_COLUMNS},
    }
    BOOTSTRAP_CACHE = pd.concat([BOOTSTRAP_CACHE, pd.DataFrame([cache_row])], ignore_index=True)
    BOOTSTRAP_CACHE = BOOTSTRAP_CACHE.drop_duplicates(
        subset=CACHE_KEY_COLUMNS, keep='last'
    )
    # Save after every completed configuration so an interrupted run can resume.
    BOOTSTRAP_CACHE.to_csv(BOOTSTRAP_CACHE_PATH, index=False)
    return result

def load_prediction_arrays(model, setting):
    run_dir, detail_path, summary_path = locate_run(model, setting)
    detail = pd.read_csv(detail_path, dtype={'patient_id': 'string'})
    schema = MODEL_SPECS[model]['schema']
    if schema == 'tabular':
        y_scan = detail[[f'true_{scan}' for scan in SCAN_TYPES]].to_numpy(dtype=int)
        p_scan = detail[[f'prob_{scan}' for scan in SCAN_TYPES]].to_numpy(dtype=float)
        pred_scan = detail[[f'pred_{scan}' for scan in SCAN_TYPES]].to_numpy(dtype=int)
        true_interval = detail['interval_true_months'].to_numpy(dtype=float)
        pred_interval = detail['interval_pred_months'].to_numpy(dtype=float)
        if 'row_idx' in detail and not np.array_equal(
            detail['row_idx'].to_numpy(dtype=int), np.arange(len(detail), dtype=int)
        ):
            raise AssertionError(f'{model}/{setting}: unexpected tabular test-row order.')
        patient_id = TEST_PATIENT_IDS.copy()
    else:
        y_scan = detail[[f'y_scan_{scan}' for scan in SCAN_TYPES]].to_numpy(dtype=int)
        p_scan = detail[[f'p_scan_{scan}' for scan in SCAN_TYPES]].to_numpy(dtype=float)
        pred_scan = detail[[f'pred_scan_{scan}' for scan in SCAN_TYPES]].to_numpy(dtype=int)
        true_interval = detail['true_interval_months'].to_numpy(dtype=float)
        pred_interval = detail['pred_interval_months'].to_numpy(dtype=float)
        patient_id = detail['patient_id'].astype(str).to_numpy()
        if not np.array_equal(patient_id, TEST_PATIENT_IDS):
            raise AssertionError(f'{model}/{setting}: sequence test-row order mismatch.')
    if len(detail) != len(TEST_PATIENT_IDS):
        raise AssertionError(f'{model}/{setting}: unexpected number of test rows.')
    if not np.array_equal(y_scan, TEST_TRUE_SCAN):
        raise AssertionError(f'{model}/{setting}: true scan labels do not match frozen test targets.')
    return {
        'run_dir': run_dir, 'summary_path': summary_path, 'detail_path': detail_path,
        'source_signature': prediction_source_signature(detail_path),
        'patient_id': patient_id, 'y_scan': y_scan, 'p_scan': p_scan,
        'pred_scan': pred_scan, 'true_interval': true_interval, 'pred_interval': pred_interval,
    }


# In[5]:


table_ii_rows = []
prediction_cache = {}
for model in MODEL_SPECS:
    for setting in FEATURE_SETS:
        pred = load_prediction_arrays(model, setting)
        result = cached_metrics_with_ci(
            cache_key=f'overall|{model}|{setting}',
            source_signature=pred['source_signature'],
            seed=BOOTSTRAP_SEED + len(table_ii_rows),
            patient_id=pred['patient_id'], y_scan=pred['y_scan'], p_scan=pred['p_scan'],
            pred_scan=pred['pred_scan'], true_interval=pred['true_interval'],
            pred_interval=pred['pred_interval'],
        )
        metrics = {key: result[key] for key in CI_METRICS + ['valid_scan_types_for_macro_auc']}
        confidence_intervals = {
            key: result[key] for key in result if key.endswith('_ci_low') or key.endswith('_ci_high')
        }
        saved_rows = pd.read_csv(pred['summary_path'])
        saved_rows = saved_rows.loc[saved_rows['experiment'].eq(setting)]
        if len(saved_rows) != 1:
            raise ValueError(f'{model}/{setting}: expected one matching summary row.')
        saved = saved_rows.iloc[0]
        for metric in ['scan_micro_auc', 'scan_macro_auc', 'interval_mae', 'interval_rmse']:
            if not np.isclose(round(metrics[metric], 4), round(float(saved[metric]), 4)):
                raise AssertionError(f'{model}/{setting}: {metric} does not reproduce the saved result.')
        interval_mape = float(saved['interval_mape'])
        mape_scaled = interval_mape / 100.0
        hyperband_score = metrics['scan_micro_auc'] - mape_scaled / (1.0 + mape_scaled)
        table_ii_rows.append({
            'model': model, 'feature_set': setting, **metrics, **confidence_intervals,
            'hyperband_score': hyperband_score,
        })
        if (model, setting) in {('GRU', 'DSIMB'), ('XGBoost', 'DSI')}:
            prediction_cache[f'{model}-{setting}'] = pred

table_ii = pd.DataFrame(table_ii_rows)
table_ii['model'] = pd.Categorical(table_ii['model'], categories=list(MODEL_SPECS), ordered=True)
table_ii['feature_set'] = pd.Categorical(table_ii['feature_set'], categories=FEATURE_SETS, ordered=True)
table_ii = table_ii.sort_values(['model', 'feature_set']).reset_index(drop=True)
if len(table_ii) != 48:
    raise AssertionError(f'Expected 48 Table II rows, found {len(table_ii)}.')
table_ii.to_csv(TABLE_II_CACHE_PATH, index=False)
print(f'Bootstrap cache: {CACHE_HITS} reused, {CACHE_MISSES} calculated.')
print(f'Saved: {TABLE_II_CACHE_PATH}')

PAIR_DEFINITIONS = {
    'Imaging': [('DS', 'DSI'), ('DSM', 'DSIM'), ('DSB', 'DSIB'), ('DSMB', 'DSIMB')],
    'Mastectomy': [('DS', 'DSM'), ('DSI', 'DSIM'), ('DSB', 'DSMB'), ('DSIB', 'DSIMB')],
    'Biomarker': [('DS', 'DSB'), ('DSI', 'DSIB'), ('DSM', 'DSMB'), ('DSIM', 'DSIMB')],
}
MODEL_KEYS = {
    'Linear': 'LR', 'XGBoost': 'XGBOOST', 'MLP': 'MLP', 'GRU': 'GRU',
    'LSTM': 'LSTM', 'Decoder-only Transformer': 'DECODER_ONLY_TRANSFORMER',
}
contribution_rows = []
for model in MODEL_SPECS:
    model_df = table_ii.loc[table_ii['model'].astype(str).eq(model)].set_index('feature_set')
    for feature_group, pairs in PAIR_DEFINITIONS.items():
        micro_auc_deltas, macro_auc_deltas, mae_reductions, rmse_reductions = [], [], [], []
        for without_setting, with_setting in pairs:
            micro_auc_deltas.append(
                float(model_df.loc[with_setting, 'scan_micro_auc'] - model_df.loc[without_setting, 'scan_micro_auc'])
            )
            macro_auc_deltas.append(
                float(model_df.loc[with_setting, 'scan_macro_auc'] - model_df.loc[without_setting, 'scan_macro_auc'])
            )
            mae_reductions.append(
                float(model_df.loc[without_setting, 'interval_mae'] - model_df.loc[with_setting, 'interval_mae'])
            )
            rmse_reductions.append(
                float(model_df.loc[without_setting, 'interval_rmse'] - model_df.loc[with_setting, 'interval_rmse'])
            )
        contribution_rows.append({
            'model': MODEL_KEYS[model], 'model_display': model,
            'feature_group': feature_group, 'n_pairs': len(pairs),
            'micro_auc_contribution_mean': np.mean(micro_auc_deltas),
            'macro_auc_contribution_mean': np.mean(macro_auc_deltas),
            'mae_reduction_mean': np.mean(mae_reductions),
            'rmse_reduction_mean': np.mean(rmse_reductions),
        })
pd.DataFrame(contribution_rows).to_csv(
    OUTPUT_DIR / 'feature_group_contributions.csv', index=False
)

display(table_ii[[
    'model', 'feature_set', 'scan_micro_auc', 'scan_macro_auc',
    'scan_micro_f1', 'scan_macro_f1', 'scan_pooled_ece', 'scan_macro_ece',
    'interval_mae', 'interval_rmse', 'hyperband_score',
]].style.format({metric: '{:.4f}' for metric in [
    'scan_micro_auc', 'scan_macro_auc', 'scan_micro_f1', 'scan_macro_f1',
    'scan_pooled_ece', 'scan_macro_ece',
    'interval_mae', 'interval_rmse', 'hyperband_score',
]}))


# In[6]:


TABLE_METRICS = [
    'scan_micro_auc', 'scan_macro_auc', 'scan_micro_f1', 'scan_macro_f1',
    'scan_pooled_ece', 'scan_macro_ece',
    'interval_mae', 'interval_rmse', 'hyperband_score',
]
HIGHER_IS_BETTER = {
    'scan_micro_auc', 'scan_macro_auc', 'scan_micro_f1', 'scan_macro_f1', 'hyperband_score'
}
GLOBAL_MARK_METRICS = set(TABLE_METRICS) - {'hyperband_score'}

def best_worst(values, metric):
    values = pd.to_numeric(values, errors='coerce').dropna()
    if values.empty:
        return None, None
    if metric in HIGHER_IS_BETTER:
        return float(values.max()), float(values.min())
    return float(values.min()), float(values.max())

def metric_cell(row, metric, block_best, block_worst, global_best, global_worst):
    value = float(row[metric])
    if not np.isfinite(value):
        return '--'
    text = f'{float(value):.4f}'
    if metric != 'hyperband_score':
        low = float(row[f'{metric}_ci_low'])
        high = float(row[f'{metric}_ci_high'])
        text += f' [{low:.4f}, {high:.4f}]'
    if metric in GLOBAL_MARK_METRICS and np.isclose(value, global_best):
        text = rf'\boxedcell{{{text}}}'
    elif metric in GLOBAL_MARK_METRICS and np.isclose(value, global_worst):
        text = rf'\underline{{{text}}}'
    if np.isclose(value, block_best):
        text += r'$^{*}$'
    elif np.isclose(value, block_worst):
        text += r'$^{\triangledown}$'
    return text

def bold_cell(text):
    return rf'\textbf{{{text}}}'

def build_revised_table_ii(source):
    global_extrema = {metric: best_worst(source[metric], metric) for metric in TABLE_METRICS}
    global_hyperband_best = global_extrema['hyperband_score'][0]
    rows = []
    model_order = list(MODEL_SPECS)
    for model_index, model in enumerate(model_order):
        block = source[source['model'].astype(str).eq(model)].copy()
        block_extrema = {metric: best_worst(block[metric], metric) for metric in TABLE_METRICS}
        for row_index, (_, row) in enumerate(block.iterrows()):
            model_cell = rf'\multirow{{8}}{{2cm}}{{\centering {model}}}' if row_index == 0 else ''
            cells = [model_cell, str(row['feature_set'])]
            for metric in TABLE_METRICS:
                cells.append(metric_cell(
                    row, metric, *block_extrema[metric], *global_extrema[metric]
                ))
            if np.isclose(float(row['hyperband_score']), global_hyperband_best):
                cells = [cells[0]] + [bold_cell(cell) for cell in cells[1:]]
            rows.append(' & '.join(cells) + r' \\')
        if model_index != len(model_order) - 1:
            rows.append(r'\midrule')
    body = '\n'.join(rows)
    return rf'''\begin{{table*}}[htbp]
\caption{{Model performance across feature sets with patient-bootstrap 95\% confidence intervals. $*/\triangledown$: best/worst feature set within model; box/underline: global best/worst scan and interval values. Bold row: best Hyperband score.}}
\label{{tab:revised_model_results}}
\begin{{center}}
\scriptsize
\setlength{{\tabcolsep}}{{2.0pt}}
% \renewcommand{{\arraystretch}}{{1.15}}
\providecommand{{\boxedcell}}[1]{{{{\setlength{{\fboxsep}}{{1pt}}\smash{{\fbox{{#1}}}}\vphantom{{#1}}}}}}
\vspace{{-1em}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}ll*{{9}}{{c}}@{{}}}}
\toprule
\multirow{{2.5}}{{2cm}}{{\centering \textbf{{Model}}}} &
\multirow{{2.5}}{{1.7cm}}{{\centering \textbf{{Feature Set}}}} &
\multicolumn{{6}}{{c}}{{\textbf{{Scan Type Task}}}} &
\multicolumn{{2}}{{c}}{{\textbf{{Interval Task}}}} &
\multirow{{2.5}}{{1.72cm}}{{\centering \shortstack{{$\uparrow$ \textbf{{Hyperband}}\\\textbf{{Score}}}}}} \\
\cmidrule(lr){{3-8}} \cmidrule(lr){{9-10}}
 & & $\uparrow$ micro-AUC & $\uparrow$ macro-AUC & $\uparrow$ micro-F1 & $\uparrow$ macro-F1 &
 $\downarrow$ pooled ECE & $\downarrow$ macro-ECE & $\downarrow$ MAE & $\downarrow$ RMSE & \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{center}}
\end{{table*}}'''

revised_table_ii_latex = build_revised_table_ii(table_ii)
print(revised_table_ii_latex)
print(r'% Required packages: \usepackage{booktabs,multirow,array,graphicx,amssymb}')


# ## 3. Compact GRU-DSIMB and XGBoost-DSI subgroup tables
# 
# For per-scan-type rows, `n` is the number of held-out target events positive for that scan type and the percentage uses all 28,830 held-out prediction events as the denominator. Because a same-day target can contain more than one scan type, these rows can overlap. The grouping scan label is constant within its row, so macro-AUC averages the other evaluable scan labels.
# 
# For stage rows, `n` is the number of unique held-out patients and the percentage uses all 4,713 held-out patients. Stage groups are mutually exclusive.

# In[7]:


COHORT_PATH = FROZEN_MULTITASK / 'demographic.csv'
STAGING_PATH = FROZEN_MULTITASK / 'staging_timeseries.csv'
STAGE_SOURCE_SIGNATURE = '|'.join(
    f'{path.resolve()}|{path.stat().st_size}|{path.stat().st_mtime_ns}'
    for path in [COHORT_PATH, STAGING_PATH]
)
cohort = pd.read_csv(COHORT_PATH, usecols=['patient_id'], dtype={'patient_id': 'string'}).drop_duplicates()
staging = pd.read_csv(STAGING_PATH, dtype={'patient_id': 'string'})
staging['stage_date'] = pd.to_datetime(staging['stage_date'], errors='coerce')
staging = staging[staging['stage_date'].notna()].copy()
staging['is_stage_iv'] = staging['stage_group_name'].astype(str).str.upper().eq('IV')
daily_stage = (
    staging.groupby(['patient_id', 'stage_date'], as_index=False)['is_stage_iv'].max()
    .sort_values(['patient_id', 'stage_date'], kind='mergesort')
)
trajectory = daily_stage.groupby('patient_id', sort=False).agg(
    first_is_stage_iv=('is_stage_iv', 'first'), ever_stage_iv=('is_stage_iv', 'max'),
).reset_index()
trajectory['stage_group'] = np.select(
    [~trajectory['ever_stage_iv'], trajectory['first_is_stage_iv']],
    ['Stage 0-III only', 'Stage IV throughout'],
    default='Stage 0-III to IV',
)
patient_stage = cohort.merge(trajectory, on='patient_id', how='left')
patient_stage['stage_group'] = patient_stage['stage_group'].fillna('Unknown stage')
stage_map = dict(zip(patient_stage['patient_id'].astype(str), patient_stage['stage_group']))
STAGE_ORDER = ['Stage 0-III only', 'Stage 0-III to IV', 'Stage IV throughout', 'Unknown stage']

def selected_model_subgroup_table(model_key, display_name, seed_offset):
    pred = prediction_cache[model_key]
    n_test_events = len(pred['patient_id'])
    n_test_patients = pd.Series(pred['patient_id']).nunique()
    if (n_test_events, n_test_patients) != (28_830, 4_713):
        raise AssertionError(f'Unexpected {display_name} test size: {(n_test_events, n_test_patients)}')
    rows = []
    for index, scan_type in enumerate(SCAN_TYPES):
        mask = pred['y_scan'][:, index].astype(bool)
        result = cached_metrics_with_ci(
            cache_key=f'subgroup|{model_key}|scan|{scan_type}',
            source_signature=pred['source_signature'], seed=BOOTSTRAP_SEED + seed_offset + index,
            patient_id=pred['patient_id'][mask], y_scan=pred['y_scan'][mask],
            p_scan=pred['p_scan'][mask], pred_scan=pred['pred_scan'][mask],
            true_interval=pred['true_interval'][mask], pred_interval=pred['pred_interval'][mask],
        )
        n = int(mask.sum())
        rows.append({
            'model': display_name, 'section': 'Per-scan-type results',
            'subgroup': SCAN_LABELS[scan_type], 'n': n, 'percent': 100 * n / n_test_events,
            **result,
        })
    prediction_stage = np.array(
        [stage_map.get(str(patient), 'Unknown stage') for patient in pred['patient_id']], dtype=object
    )
    for stage_index, stage_group in enumerate(STAGE_ORDER):
        mask = prediction_stage == stage_group
        result = cached_metrics_with_ci(
            cache_key=f'subgroup|{model_key}|stage|{stage_group}',
            source_signature=f"{pred['source_signature']}|{STAGE_SOURCE_SIGNATURE}",
            seed=BOOTSTRAP_SEED + seed_offset + 100 + stage_index,
            patient_id=pred['patient_id'][mask], y_scan=pred['y_scan'][mask],
            p_scan=pred['p_scan'][mask], pred_scan=pred['pred_scan'][mask],
            true_interval=pred['true_interval'][mask], pred_interval=pred['pred_interval'][mask],
        )
        n = pd.Series(pred['patient_id'][mask]).nunique()
        rows.append({
            'model': display_name, 'section': 'Per-stage-cohort results',
            'subgroup': stage_group, 'n': int(n), 'percent': 100 * n / n_test_patients,
            **result,
        })
    return pd.DataFrame(rows)

gru_subgroup_table = selected_model_subgroup_table('GRU-DSIMB', 'GRU-DSIMB', 1000)
xgboost_subgroup_table = selected_model_subgroup_table('XGBoost-DSI', 'XGBoost-DSI', 3000)
selected_model_subgroups = pd.concat(
    [gru_subgroup_table, xgboost_subgroup_table], ignore_index=True
)
selected_model_subgroups.to_csv(SUBGROUP_CACHE_PATH, index=False)
print(f'Bootstrap cache total: {CACHE_HITS} reused, {CACHE_MISSES} calculated.')
print(f'Saved: {SUBGROUP_CACHE_PATH}')

xgboost_detail = pd.read_csv(prediction_cache['XGBoost-DSI']['detail_path'])
per_scan_type_rows = []
for scan_type in SCAN_TYPES:
    y_true = xgboost_detail[f'true_{scan_type}'].to_numpy(dtype=int)
    y_pred = xgboost_detail[f'pred_{scan_type}'].to_numpy(dtype=int)
    per_scan_type_rows.append({
        'Scan type': SCAN_LABELS[scan_type],
        'Positive n': int(y_true.sum()),
        'Positive (%)': 100 * float(y_true.mean()),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Threshold': float(xgboost_detail[f'threshold_{scan_type}'].iloc[0]),
    })
pd.DataFrame(per_scan_type_rows).to_csv(
    OUTPUT_DIR / 'xgboost_dsi_per_scan_type_f1.csv', index=False
)
display(selected_model_subgroups[[
    'section', 'subgroup', 'n', 'percent', 'scan_micro_auc', 'scan_macro_auc',
    'scan_micro_f1', 'scan_macro_f1', 'scan_pooled_ece', 'scan_macro_ece',
    'interval_mae', 'interval_rmse',
]].style.format({
    'n': '{:,}', 'percent': '{:.1f}%',
    'scan_micro_auc': '{:.4f}', 'scan_macro_auc': '{:.4f}',
    'scan_micro_f1': '{:.4f}', 'scan_macro_f1': '{:.4f}',
    'scan_pooled_ece': '{:.4f}', 'scan_macro_ece': '{:.4f}',
    'interval_mae': '{:.4f}', 'interval_rmse': '{:.4f}',
}))


# In[8]:


def latex_escape(value):
    return str(value).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')

def subgroup_display_name(row):
    names = {
        'Stage 0-III only': r'Stage 0--III only',
        'Stage 0-III to IV': r'Stage 0--III $\rightarrow$ IV',
        'Stage IV throughout': r'Stage IV throughout',
        'Unknown stage': r'Unknown stage',
    }
    base = names.get(row['subgroup'], latex_escape(row['subgroup']))
    return rf'{base} (n={int(row["n"]):,}, {float(row["percent"]):.1f}\%)'

SUBGROUP_METRICS = [
    'scan_micro_auc', 'scan_macro_auc', 'scan_micro_f1', 'scan_macro_f1',
    'scan_pooled_ece', 'scan_macro_ece', 'interval_mae', 'interval_rmse',
]
SUBGROUP_HIGHER_IS_BETTER = {
    'scan_micro_auc', 'scan_macro_auc', 'scan_micro_f1', 'scan_macro_f1'
}

def subgroup_metric_cell(row, section_source, metric):
    value = float(row[metric])
    values = pd.to_numeric(section_source[metric], errors='coerce').dropna()
    low = float(row[f'{metric}_ci_low'])
    high = float(row[f'{metric}_ci_high'])
    text = f'{value:.4f} [{low:.4f}, {high:.4f}]'
    best = values.max() if metric in SUBGROUP_HIGHER_IS_BETTER else values.min()
    worst = values.min() if metric in SUBGROUP_HIGHER_IS_BETTER else values.max()
    if np.isclose(value, best):
        text += r'$^{*}$'
    elif np.isclose(value, worst):
        text += r'$^{\triangledown}$'
    return text

def build_subgroup_latex(source, model_name, table_label):
    rows = []
    sections = [
        ('Per-scan-type results', 'Scan-Type Subgroups'),
        ('Per-stage-cohort results', 'Stage-Trajectory Subgroups'),
    ]
    for section_index, (section_key, section_label) in enumerate(sections):
        if section_index:
            rows.append(r'\midrule')
        rows.append(rf'\multicolumn{{9}}{{c}}{{\textbf{{{section_label}}}}} \\')
        rows.append(r'\midrule')
        section_source = source[source['section'].eq(section_key)].copy()
        for _, row in section_source.iterrows():
            cells = [
                subgroup_display_name(row),
                *[subgroup_metric_cell(row, section_source, metric) for metric in SUBGROUP_METRICS],
            ]
            rows.append(' & '.join(cells) + r' \\')
    body = '\n'.join(rows)
    return rf'''\begin{{table*}}[htbp]
\caption{{{model_name} test performance by scan type and stage trajectory with patient-bootstrap 95\% confidence intervals.}}
\label{{{table_label}}}
\begin{{center}}
\scriptsize
\setlength{{\tabcolsep}}{{2.0pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}l*{{8}}{{c}}@{{}}}}
\toprule
\multirow{{2}}{{*}}{{\textbf{{Subgroup}}}} &
\multicolumn{{6}}{{c}}{{\textbf{{Scan Type Task}}}} &
\multicolumn{{2}}{{c}}{{\textbf{{Interval Task}}}} \\
\cmidrule(lr){{2-7}} \cmidrule(lr){{8-9}}
 & $\uparrow$ micro-AUC & $\uparrow$ macro-AUC & $\uparrow$ micro-F1 & $\uparrow$ macro-F1 &
 $\downarrow$ pooled ECE & $\downarrow$ macro-ECE & $\downarrow$ MAE & $\downarrow$ RMSE \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{center}}
\end{{table*}}'''

gru_subgroup_latex = build_subgroup_latex(
    gru_subgroup_table, 'GRU-DSIMB', 'tab:gru_dsimb_subgroup_results'
)
xgboost_subgroup_latex = build_subgroup_latex(
    xgboost_subgroup_table, 'XGBoost-DSI', 'tab:xgboost_dsi_subgroup_results'
)
print(gru_subgroup_latex)
print('\n% ----- Separate XGBoost-DSI table -----\n')
print(xgboost_subgroup_latex)
print(r'% Required packages: \usepackage{booktabs,multirow,graphicx}')


# ## Output files
# 
# The figure cell saves all three PDFs automatically. The metric cells save the incremental bootstrap cache and final CSV tables automatically.

# In[46]:


print('Figure PDFs:')
for path in FIGURE_OUTPUTS:
    print(' -', path)
print('Reusable bootstrap cache:', BOOTSTRAP_CACHE_PATH)
print('Table II CSV:', TABLE_II_CACHE_PATH)
print('Selected-model subgroup CSV:', SUBGROUP_CACHE_PATH)

