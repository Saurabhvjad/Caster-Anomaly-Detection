"""
Caster Anomaly Detection Dashboard
===================================
Streamlit app for real-time monitoring and anomaly detection demonstration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
import shap
from sklearn.preprocessing import StandardScaler
import time
import streamlit.components.v1 as components


# Hide Streamlit default top-right menu & footer
hide_default_format = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;} /* hides title space if any */
</style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)



# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Caster Anomaly Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS – compressed layout, sticky top, reduced padding
# ============================================================================
st.markdown("""
<style>
    /* ---- Compact header ---- */
    .main-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 0.50rem 0.5rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        border-radius: 8px;
        margin-bottom: 0.15rem;
    }

    /* ---- Global spacing reduction ---- */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0 !important;
    }
    h3, h4 { margin-top: 0.2rem !important; margin-bottom: 0.1rem !important; font-size: 1rem !important; }
    .stMarkdown { margin-bottom: 0 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
    div[data-testid="stMetricDelta"]  { font-size: 0.7rem !important; }
    .stDivider { margin: 0.15rem 0 !important; }

    /* ---- Sticky top section is applied via JS injection below ---- */

    /* ---- Cards ---- */
    .metric-card {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .alert-card {
        padding: 0.35rem 0.5rem;
        border-radius: 5px;
        margin: 0.15rem 0;
        font-size: 0.85rem;
    }
    .alert-normal  { background-color: #d4edda; border-left: 4px solid #28a745; }
    .alert-warning  { background-color: #fff3cd; border-left: 4px solid #ffc107; }
    .alert-critical { background-color: #f8d7da; border-left: 4px solid #dc3545; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }

    /* ---- Rule summary strip ---- */
    .rule-strip {
        display: flex;
        gap: 1rem;
        font-size: 0.78rem;
        background: #f0f2f6;
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        margin: 0.2rem 0;
        align-items: center;
    }
    .rule-strip b { color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

from pathlib import Path

# Directory containing this script (app.py)
BASE_DIR = Path(__file__).resolve().parent

SAMPLE_DATA_FILE = BASE_DIR / "Sample Data_Good&Bad.xlsx"
MODEL_FILE       = BASE_DIR / "caster_isolation_forest_model.pkl"
SCALER_FILE      = BASE_DIR / "caster_scaler.pkl"
FEATURES_FILE    = BASE_DIR / "feature_names.pkl"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(file_path):
    """Load and prepare sample data"""
    df = pd.read_excel(file_path, engine='openpyxl')
    if len(df) > 7:
        df = df.drop(df.index[1:7])
        df = df.reset_index(drop=True)
    if 'TIME' in df.columns:
        df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
    return df

@st.cache_resource
def load_model_artifacts():
    """Load trained model, scaler, and feature names"""
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_names = joblib.load(FEATURES_FILE)
    return model, scaler, feature_names

def identify_tc_columns(df):
    """Identify and group thermocouple columns"""
    tc_columns = [col for col in df.columns if 'MMS' in col and 'TC' in col]
    fix_side_tc = [col for col in tc_columns if 'FIX SIDE' in col]
    loose_side_tc = [col for col in tc_columns if 'LOOSE SIDE' in col]

    fix_row1 = sorted([col for col in fix_side_tc if 'ROW1' in col])
    fix_row2 = sorted([col for col in fix_side_tc if 'ROW2' in col])
    fix_row3 = sorted([col for col in fix_side_tc if 'ROW3' in col])
    loose_row1 = sorted([col for col in loose_side_tc if 'ROW1' in col])
    loose_row2 = sorted([col for col in loose_side_tc if 'ROW2' in col])
    loose_row3 = sorted([col for col in loose_side_tc if 'ROW3' in col])

    return {
        'all': tc_columns,
        'fix': fix_side_tc, 'loose': loose_side_tc,
        'fix_row1': fix_row1, 'fix_row2': fix_row2, 'fix_row3': fix_row3,
        'loose_row1': loose_row1, 'loose_row2': loose_row2, 'loose_row3': loose_row3
    }

def engineer_features(df, tc_groups):
    """Create engineered features for the dataframe"""
    df_features = df.copy()

    df_features['Fix_Row1_Avg'] = df_features[tc_groups['fix_row1']].median(axis=1)
    df_features['Fix_Row2_Avg'] = df_features[tc_groups['fix_row2']].median(axis=1)
    df_features['Fix_Row3_Avg'] = df_features[tc_groups['fix_row3']].median(axis=1)
    df_features['Loose_Row1_Avg'] = df_features[tc_groups['loose_row1']].median(axis=1)
    df_features['Loose_Row2_Avg'] = df_features[tc_groups['loose_row2']].median(axis=1)
    df_features['Loose_Row3_Avg'] = df_features[tc_groups['loose_row3']].median(axis=1)

    df_features['Fix_Side_Avg'] = df_features[tc_groups['fix']].median(axis=1)
    df_features['Loose_Side_Avg'] = df_features[tc_groups['loose']].median(axis=1)
    df_features['Overall_TC_Avg'] = df_features[tc_groups['all']].median(axis=1)

    df_features['Fix_Row1_Row2_Gradient'] = df_features['Fix_Row1_Avg'] - df_features['Fix_Row2_Avg']
    df_features['Fix_Row2_Row3_Gradient'] = df_features['Fix_Row2_Avg'] - df_features['Fix_Row3_Avg']
    df_features['Loose_Row1_Row2_Gradient'] = df_features['Loose_Row1_Avg'] - df_features['Loose_Row2_Avg']
    df_features['Loose_Row2_Row3_Gradient'] = df_features['Loose_Row2_Avg'] - df_features['Loose_Row3_Avg']

    df_features['Fix_Loose_Diff'] = df_features['Fix_Side_Avg'] - df_features['Loose_Side_Avg']
    df_features['Fix_Loose_Row1_Diff'] = df_features['Fix_Row1_Avg'] - df_features['Loose_Row1_Avg']
    df_features['Fix_Loose_Row2_Diff'] = df_features['Fix_Row2_Avg'] - df_features['Loose_Row2_Avg']
    df_features['Fix_Loose_Row3_Diff'] = df_features['Fix_Row3_Avg'] - df_features['Loose_Row3_Avg']

    for tc_num in range(1, 9):
        tc_cols = [col for col in tc_groups['all'] if f'TC{tc_num}' in col]
        if tc_cols:
            df_features[f'TC{tc_num}_Position_Avg'] = df_features[tc_cols].median(axis=1)

    df_features['Fix_Side_Std'] = df_features[tc_groups['fix']].std(axis=1)
    df_features['Loose_Side_Std'] = df_features[tc_groups['loose']].std(axis=1)
    df_features['Fix_Row1_Std'] = df_features[tc_groups['fix_row1']].std(axis=1)
    df_features['Fix_Row2_Std'] = df_features[tc_groups['fix_row2']].std(axis=1)
    df_features['Fix_Row3_Std'] = df_features[tc_groups['fix_row3']].std(axis=1)
    df_features['Loose_Row1_Std'] = df_features[tc_groups['loose_row1']].std(axis=1)
    df_features['Loose_Row2_Std'] = df_features[tc_groups['loose_row2']].std(axis=1)
    df_features['Loose_Row3_Std'] = df_features[tc_groups['loose_row3']].std(axis=1)

    df_features['Fix_Row1_Range'] = df_features[tc_groups['fix_row1']].max(axis=1) - df_features[tc_groups['fix_row1']].min(axis=1)
    df_features['Fix_Row2_Range'] = df_features[tc_groups['fix_row2']].max(axis=1) - df_features[tc_groups['fix_row2']].min(axis=1)
    df_features['Fix_Row3_Range'] = df_features[tc_groups['fix_row3']].max(axis=1) - df_features[tc_groups['fix_row3']].min(axis=1)
    df_features['Loose_Row1_Range'] = df_features[tc_groups['loose_row1']].max(axis=1) - df_features[tc_groups['loose_row1']].min(axis=1)
    df_features['Loose_Row2_Range'] = df_features[tc_groups['loose_row2']].max(axis=1) - df_features[tc_groups['loose_row2']].min(axis=1)
    df_features['Loose_Row3_Range'] = df_features[tc_groups['loose_row3']].max(axis=1) - df_features[tc_groups['loose_row3']].min(axis=1)

    if 'Mold Cool Water Fix Side Heat Flux' in df_features.columns and 'Mold Cool Water Fix Side Flow' in df_features.columns:
        df_features['Fix_Heat_Flux_Per_Flow'] = df_features['Mold Cool Water Fix Side Heat Flux'] / (df_features['Mold Cool Water Fix Side Flow'] + 1e-6)
    if 'Mold Cool Water Loose Side Heat Flux' in df_features.columns and 'Mold Cool Water Loose Side Flow' in df_features.columns:
        df_features['Loose_Heat_Flux_Per_Flow'] = df_features['Mold Cool Water Loose Side Heat Flux'] / (df_features['Mold Cool Water Loose Side Flow'] + 1e-6)

    for side, cols in [('Fix', tc_groups['fix']), ('Loose', tc_groups['loose'])]:
        edge_cols = [col for col in cols if 'TC1' in col or 'TC8' in col]
        center_cols = [col for col in cols if any(f'TC{i}' in col for i in [3, 4, 5, 6])]
        if edge_cols and center_cols:
            df_features[f'{side}_Edge_Avg'] = df_features[edge_cols].median(axis=1)
            df_features[f'{side}_Center_Avg'] = df_features[center_cols].median(axis=1)
            df_features[f'{side}_Edge_Center_Diff'] = df_features[f'{side}_Center_Avg'] - df_features[f'{side}_Edge_Avg']

    df_features['Fix_Row1_Max_Diff'] = df_features[tc_groups['fix_row1']].max(axis=1) - df_features[tc_groups['fix_row1']].min(axis=1)
    df_features['Fix_Row2_Max_Diff'] = df_features[tc_groups['fix_row2']].max(axis=1) - df_features[tc_groups['fix_row2']].min(axis=1)
    df_features['Fix_Row3_Max_Diff'] = df_features[tc_groups['fix_row3']].max(axis=1) - df_features[tc_groups['fix_row3']].min(axis=1)
    df_features['Loose_Row1_Max_Diff'] = df_features[tc_groups['loose_row1']].max(axis=1) - df_features[tc_groups['loose_row1']].min(axis=1)
    df_features['Loose_Row2_Max_Diff'] = df_features[tc_groups['loose_row2']].max(axis=1) - df_features[tc_groups['loose_row2']].min(axis=1)
    df_features['Loose_Row3_Max_Diff'] = df_features[tc_groups['loose_row3']].max(axis=1) - df_features[tc_groups['loose_row3']].min(axis=1)

    if 'Mold Cool Water Fix Side Heat Flux' in df_features.columns:
        df_features['Fix_TC_to_HeatFlux_Ratio'] = df_features['Fix_Side_Avg'] / (df_features['Mold Cool Water Fix Side Heat Flux'] + 1e-6)
    if 'Mold Cool Water Loose Side Heat Flux' in df_features.columns:
        df_features['Loose_TC_to_HeatFlux_Ratio'] = df_features['Loose_Side_Avg'] / (df_features['Mold Cool Water Loose Side Heat Flux'] + 1e-6)

    return df_features

def check_rules(df, current_idx, tc_groups):
    """Check the 3 priority rules. Returns enriched alert dicts."""
    alerts = []
    current_row = df.iloc[current_idx]
    if 'TIME' in df.columns and pd.notna(current_row.get('TIME')):
        _t = current_row['TIME']
        row_time = _t.strftime('%H:%M:%S') if hasattr(_t, 'strftime') else str(_t).split('.')[0]
    else:
        row_time = ''

    # Rule 1: Deviation > 15°C in last 2 minutes (120 seconds)
    window_start = max(0, current_idx - 120)
    window_data = df.iloc[window_start:current_idx+1]

    if len(window_data) > 1:
        for tc in tc_groups['all']:
            if tc in window_data.columns:
                tc_values = window_data[tc].dropna()
                if len(tc_values) > 1:
                    deviation = tc_values.iloc[-1] - tc_values.iloc[0]
                    if abs(deviation) >= 15:
                        _action = 'Reduce casting speed' if deviation > 0 else 'Check cooling water flow'
                        alerts.append({
                            'rule': 'Rule 1',
                            'severity': 'critical',
                            'message': f"{tc}: {deviation:+.1f}°C in 2 min",
                            'parameter': tc,
                            'value': deviation,
                            'index': current_idx,
                            'time': row_time,
                            'tc': tc,
                            'description': f"{tc} deviated {deviation:+.1f}°C over 2 min",
                            'actual_value': f"{tc_values.iloc[-1]:.1f}°C",
                            'compared_with': f"{tc_values.iloc[0]:.1f}°C (2 min ago)",
                            'action': _action
                        })

    # Rule 2: TC current value difference > 10°C from its own recent median
    for row_name, row_cols in [
        ('Fix ROW1', tc_groups['fix_row1']),
        ('Fix ROW2', tc_groups['fix_row2']),
        ('Fix ROW3', tc_groups['fix_row3']),
        ('Loose ROW1', tc_groups['loose_row1']),
        ('Loose ROW2', tc_groups['loose_row2']),
        ('Loose ROW3', tc_groups['loose_row3'])
    ]:
        for tc in row_cols:
            if tc in current_row and pd.notna(current_row[tc]):
                recent_start = max(0, current_idx - 60)
                tc_recent = df.iloc[recent_start:current_idx][tc].dropna()
                if len(tc_recent) > 0:
                    tc_median = tc_recent.median()
                    diff_from_tc_median = abs(current_row[tc] - tc_median)
                    if diff_from_tc_median >= 15:
                        _action = 'Inspect mold condition; verify TC sensor' if diff_from_tc_median >= 25 else 'Monitor closely; adjust secondary cooling'
                        alerts.append({
                            'rule': 'Rule 2',
                            'severity': 'critical',
                            'message': f"{tc}: {diff_from_tc_median:.1f}°C off median",
                            'parameter': tc,
                            'value': diff_from_tc_median,
                            'index': current_idx,
                            'time': row_time,
                            'tc': tc,
                            'description': f"{tc} deviates {diff_from_tc_median:.1f}°C from median",
                            'actual_value': f"{current_row[tc]:.1f}°C",
                            'compared_with': f"{tc_median:.1f}°C (median)",
                            'action': _action
                        })

    # Rule 3: >=6 std deviations from recent readings
    if current_idx > 0:
        prev_row = df.iloc[current_idx - 1]
        for tc in tc_groups['all']:
            if tc in current_row and tc in prev_row:
                if pd.notna(current_row[tc]) and pd.notna(prev_row[tc]):
                    recent_start = max(0, current_idx - 60)
                    recent_data = df.iloc[recent_start:current_idx][tc].dropna()
                    if len(recent_data) > 2:
                        mean_val = recent_data.median()
                        std_val = recent_data.std()
                        if std_val > 0:
                            z_score = abs((current_row[tc] - mean_val) / std_val)
                            if z_score >= 6:
                                _action = 'Reduce casting speed; check breakout risk' if z_score >= 10 else 'Decrease casting speed gradually'
                                alerts.append({
                                    'rule': 'Rule 3',
                                    'severity': 'critical',
                                    'message': f"{tc}: {z_score:.1f}σ spike",
                                    'parameter': tc,
                                    'value': z_score,
                                    'index': current_idx,
                                    'time': row_time,
                                    'tc': tc,
                                    'description': f"{tc} at {z_score:.1f}σ from recent mean",
                                    'actual_value': f"{current_row[tc]:.1f}°C",
                                    'compared_with': f"{mean_val:.1f}°C ± {std_val:.1f} (mean±std)",
                                    'action': _action
                                })

    return alerts

def calculate_anomaly_score(df_row, model, scaler, feature_names):
    """Calculate anomaly score for a single row"""
    try:
        features = []
        for feat in feature_names:
            if feat in df_row.index:
                val = df_row[feat]
                try:
                    features.append(float(val))
                except (ValueError, TypeError):
                    features.append(np.nan)
            else:
                features.append(np.nan)

        features = np.array(features, dtype=float).reshape(1, -1)
        if np.isnan(features).any():
            return 0.0, None

        features_scaled = scaler.transform(features)
        score = model.decision_function(features_scaled)[0]
        prediction = model.predict(features_scaled)[0]
        return score, prediction
    except Exception as e:
        st.warning(f"Error calculating anomaly score: {e}")
        return 0.0, None

def get_shap_explanation(df_row, model, scaler, feature_names):
    """Get SHAP explanation for current data point"""
    try:
        features = []
        for feat in feature_names:
            if feat in df_row.index:
                val = df_row[feat]
                try:
                    features.append(float(val))
                except (ValueError, TypeError):
                    features.append(np.nan)
            else:
                features.append(np.nan)

        features = np.array(features, dtype=float).reshape(1, -1)
        if np.isnan(features).any():
            return None

        features_scaled = scaler.transform(features)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_scaled)

        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': features[0],
            'SHAP': shap_values[0]
        }).sort_values('SHAP', key=abs, ascending=False).head(10)

        return shap_df
    except Exception as e:
        st.warning(f"Error calculating SHAP values: {e}")
        return None

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load data and model
    try:
        df = load_data(SAMPLE_DATA_FILE)
        model, scaler, feature_names = load_model_artifacts()
        tc_groups = identify_tc_columns(df)
        df = engineer_features(df, tc_groups)
    except Exception as e:
        st.error(f"Error loading data or model: {e}")
        return

    total_rows = len(df)

    # ---- Session state ----
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = total_rows // 2
    if 'time_window' not in st.session_state:
        st.session_state.time_window = 600
    if 'alert_log' not in st.session_state:
        st.session_state.alert_log = []
    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if 'speed_multiplier' not in st.session_state:
        st.session_state.speed_multiplier = 10
    if 'play_total_alerts' not in st.session_state:
        st.session_state.play_total_alerts = 0
    if 'play_r1_alerts' not in st.session_state:
        st.session_state.play_r1_alerts = 0
    if 'play_r2_alerts' not in st.session_state:
        st.session_state.play_r2_alerts = 0
    if 'play_r3_alerts' not in st.session_state:
        st.session_state.play_r3_alerts = 0

    # ====================================================================
    # STICKY TOP SECTION  (Header + Controls + Timeline)
    # ====================================================================
    with st.container():
        # Invisible anchor div for sticky CSS selector
        st.markdown('<div class="sticky-anchor"></div>', unsafe_allow_html=True)

        # ---- Compact header ----
        st.markdown('<div class="main-header">🔥 Caster Anomaly Detection</div>', unsafe_allow_html=True)

        # ---- Control strip: Rule summary + Time window ----
        ctrl_left, ctrl_right = st.columns([4, 1])
        with ctrl_left:
            st.markdown(
                '<div class="rule-strip">'
                '<b style="color:#d62728">R1</b>: ≥15°C deviation in 2 min &nbsp;│&nbsp; '
                '<b style="color:#ff7f0e">R2</b>: ≥15°C from own median &nbsp;│&nbsp; '
                '<b style="color:#9467bd">R3</b>: ≥6σ from recent readings'
                '</div>',
                unsafe_allow_html=True
            )
        with ctrl_right:
            st.session_state.time_window = st.selectbox(
                "Window",
                options=[600, 1200, 3600],
                format_func=lambda x: f"{x//60} min",
                index=0,
                label_visibility="collapsed"
            )

        # ---- Timeline Navigation with Play button + Speed selector ----
        slider_col, play_col, speed_col = st.columns([10, 1, 1])

        # Build slider label that includes time
        time_label_str = ""
        if 'TIME' in df.columns and pd.notna(df.iloc[st.session_state.current_idx].get('TIME')):
            t = df.iloc[st.session_state.current_idx]['TIME']
            time_label_str = f" | {t.strftime('%H:%M:%S')}" if hasattr(t, 'strftime') else f" | {t}"

        with slider_col:
            current_idx = st.slider(
                "📊 Timeline",
                min_value=0,
                max_value=total_rows - 1,
                value=st.session_state.current_idx,
                format=f"Row %d / {total_rows - 1}{time_label_str}",
                label_visibility="collapsed"
            )

        with play_col:
            play_label = "⏸" if st.session_state.playing else "▶"
            if st.button(play_label, use_container_width=True, key="play_btn"):
                st.session_state.playing = not st.session_state.playing
                # Reset cumulative alert counter when starting a new play session
                if st.session_state.playing:
                    st.session_state.play_total_alerts = 0
                    st.session_state.play_r1_alerts = 0
                    st.session_state.play_r2_alerts = 0
                    st.session_state.play_r3_alerts = 0
                st.rerun()

        with speed_col:
            st.session_state.speed_multiplier = st.selectbox(
                "Speed",
                options=[1, 2, 5, 10, 20],
                index=[1, 2, 5, 10, 20].index(st.session_state.speed_multiplier),
                format_func=lambda x: f"{x}×",
                label_visibility="collapsed"
            )

        # Update index from slider
        if current_idx != st.session_state.current_idx:
            st.session_state.current_idx = current_idx
            new_alerts = check_rules(df, current_idx, tc_groups)
            if new_alerts:
                st.session_state.alert_log.extend(new_alerts)
            st.rerun()

        # ---- Current data (computed inside sticky so metrics can use it) ----
        current_idx = st.session_state.current_idx
        current_row = df.iloc[current_idx]

        # ----------------------------------------------------------------
        # KEY METRICS  (inside sticky container)
        # ----------------------------------------------------------------
        score, prediction = calculate_anomaly_score(current_row, model, scaler, feature_names)
        current_alerts = check_rules(df, current_idx, tc_groups)
        critical_alerts = [a for a in current_alerts if a['severity'] == 'critical']

        # Accumulate alerts during play (total + per-rule) and log them
        if st.session_state.playing and critical_alerts:
            st.session_state.play_total_alerts += len(critical_alerts)
            for _a in critical_alerts:
                if _a['rule'] == 'Rule 1':
                    st.session_state.play_r1_alerts += 1
                elif _a['rule'] == 'Rule 2':
                    st.session_state.play_r2_alerts += 1
                elif _a['rule'] == 'Rule 3':
                    st.session_state.play_r3_alerts += 1
            # Log enriched alerts into session alert_log
            st.session_state.alert_log.extend(critical_alerts)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            if score > 0:
                _sc_color, _sc_icon, _sc_label = '#28a745', '🟢', 'Normal'
            elif score > -0.15:
                _sc_color, _sc_icon, _sc_label = '#ffc107', '🟡', 'Borderline'
            else:
                _sc_color, _sc_icon, _sc_label = '#dc3545', '🔴', 'ANOMALY'
            st.markdown(
                f'<div style="background:#f0f2f6;border-radius:8px;padding:0.35rem 0.5rem;border-left:4px solid {_sc_color};">'
                f'<div style="font-size:0.72rem;color:#888;">{_sc_icon} Anomaly Score</div>'
                f'<div style="font-size:1.15rem;font-weight:bold;color:{_sc_color};">{score:.4f}</div>'
                f'<div style="font-size:0.65rem;color:#777;margin-top:1px;">'
                f'<span style="color:#dc3545;">-0.3</span>'
                f' ◄ <b style="color:{_sc_color}">{_sc_label}</b> ► '
                f'<span style="color:#28a745;">+0.3</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with m2:
            st.metric("Overall Temp Median", f"{current_row['Overall_TC_Avg']:.1f}°C")
        with m3:
            st.metric("Fix-Loose Diff", f"{current_row['Fix_Loose_Diff']:.1f}°C")
        with m4:
            st.metric("Active Alerts", len(critical_alerts), delta_color="inverse")
        with m5:
            # Clickable session alerts — scrolls to alert-history-log anchor
            st.markdown(
                f'<a href="#alert-history-log" style="text-decoration:none;color:inherit;">'
                f'<div style="background:#f0f2f6;border-radius:8px;padding:0.4rem 0.5rem;'
                f'border-left:4px solid #1f77b4;cursor:pointer;transition:background 0.2s;"'
                f' onmouseover="this.style.background=&#39;#e0e4ea&#39;" onmouseout="this.style.background=&#39;#f0f2f6&#39;">'
                f'<div style="font-size:0.72rem;color:#888;">▶ Session Alerts</div>'
                f'<div style="font-size:1.1rem;font-weight:bold;">{st.session_state.play_total_alerts}</div>'
                f'<div style="font-size:0.7rem;margin-top:2px;color:#555;">'
                f'R1: <b>{st.session_state.play_r1_alerts}</b> &nbsp;│&nbsp; '
                f'R2: <b>{st.session_state.play_r2_alerts}</b> &nbsp;│&nbsp; '
                f'R3: <b>{st.session_state.play_r3_alerts}</b></div>'
                f'</div></a>',
                unsafe_allow_html=True
            )

    
    # ---- JS injection to make sticky container work reliably ----
    components.html("""
    <script>
        const doc = window.parent.document;
        const anchor = doc.querySelector('.sticky-anchor');
        if (anchor) {
            let el = anchor;
            while (el && el.parentElement) {
                el = el.parentElement;
                const testid = el.getAttribute && el.getAttribute('data-testid');
                if (testid === 'stVerticalBlockBorderWrapper') {
                    el.style.position = 'sticky';
                    el.style.top = '0';
                    el.style.zIndex = '999';
                    el.style.background = 'white';
                    el.style.borderBottom = '2px solid #e0e0e0';
                    el.style.boxShadow = '0 2px 6px rgba(0,0,0,0.08)';
                    el.style.paddingBottom = '0.25rem';
                    break;
                }
            }
        }
    </script>
    """, height=0)

    # ---- Derived data for charts below ----
    current_time = str(df.iloc[current_idx]['TIME']) if 'TIME' in df.columns and pd.notna(df.iloc[current_idx]['TIME']) else current_idx
    half_window = st.session_state.time_window // 2
    window_start = max(0, current_idx - half_window)
    window_end = min(total_rows, current_idx + half_window)
    window_df = df.iloc[window_start:window_end].copy()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    _chart_config = {'displayModeBar': not st.session_state.playing}

    # ====================================================================
    # 2. ACTIVE ALERTS  (fixed 3×2 grid, critical only)
    # ====================================================================
    st.markdown("### 🚨 Active Alerts")
    # Build a fixed 3×2 grid (always 6 cells)
    display_alerts = critical_alerts[:6]  # cap at 6
    grid_html = (
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);'
        'grid-template-rows:repeat(2,auto);gap:4px;height:90px;overflow:hidden;">'
    )
    _rule_colors = {'Rule 1': '#d62728', 'Rule 2': '#ff7f0e', 'Rule 3': '#9467bd'}
    for i in range(6):
        if i < len(display_alerts):
            a = display_alerts[i]
            _rc = _rule_colors.get(a['rule'], '#dc3545')
            _act = a.get('action', '')
            grid_html += (
                f'<div style="background:#fff;border-left:4px solid {_rc};border-radius:5px;'
                f'margin:0;font-size:0.73rem;padding:0.2rem 0.4rem;">'
                f'<b style="color:{_rc}">{a["rule"]}</b> {a["tc"].split("TC")[-1] if "TC" in a.get("tc","") else ""}'
                f'<br/><span style="color:#333;">⚡ {_act}</span></div>'
            )
        else:
            grid_html += (
                '<div style="background:#f0f0f0;border-radius:5px;display:flex;'
                'align-items:center;justify-content:center;font-size:0.7rem;color:#aaa;">—</div>'
            )
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

    # ====================================================================
    # 3. KEY PROCESS PARAMETERS  (moved above TC trends)
    # ====================================================================
    st.markdown("### 📉 Key Process Parameters")
    kp1, kp2, kp3 = st.columns(3)

    with kp1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=window_df['TIME'], y=window_df['Overall_TC_Avg'], name='Overall Avg', line=dict(color='blue')))
        fig.add_vline(x=current_time, line_dash="dash", line_color="red")
        fig.update_layout(title="Overall TC Average", height=200, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Time")
        st.plotly_chart(fig, use_container_width=True, config=_chart_config)

    with kp2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=window_df['TIME'], y=window_df['Fix_Loose_Diff'], name='Fix-Loose Diff', line=dict(color='orange')))
        fig.add_vline(x=current_time, line_dash="dash", line_color="red")
        fig.update_layout(title="Fix-Loose Temp Diff", height=200, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Time")
        st.plotly_chart(fig, use_container_width=True, config=_chart_config)

    with kp3:
        if 'CASTING SPEED' in window_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=window_df['TIME'], y=window_df['CASTING SPEED'], name='Speed', line=dict(color='green')))
            fig.add_vline(x=current_time, line_dash="dash", line_color="red")
            fig.update_layout(title="Casting Speed", height=200, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Time")
            st.plotly_chart(fig, use_container_width=True, config=_chart_config)

    # ====================================================================
    # 4. THERMOCOUPLE TRENDS BY ROW  (compressed)
    # ====================================================================
    st.markdown("### 🌡️ Thermocouple Trends by Row")

    def display_row_parameters(row_cols, side_name):
        """Compact Z-Score parameters for a row"""
        param_data = []
        for tc in row_cols:
            if tc in current_row and pd.notna(current_row[tc]):
                recent_start = max(0, current_idx - 60)
                recent_data = df.iloc[recent_start:current_idx][tc].dropna()
                if len(recent_data) > 2:
                    mean_val = recent_data.median()
                    std_val = recent_data.std()
                    if std_val > 0:
                        z_score = abs((current_row[tc] - mean_val) / std_val)
                        tc_num = tc.split('TC')[-1] if 'TC' in tc else tc[-3:]
                        param_data.append({
                            'TC': f"TC{tc_num}", 'Value': f"{current_row[tc]:.1f}°C",
                            'Z-Score': f"{z_score:.2f}", 'Color': colors[int(tc_num) - 1]
                        })
        if param_data:
            html = '<div style="border:1px solid #d3d3d3;background:#f9f9f9;padding:2px;border-radius:4px;display:grid;grid-template-columns:repeat(8,1fr);gap:1px;">'
            for d in param_data:
                html += f'<div style="text-align:center;font-size:9px;line-height:1.1;"><span style="color:{d["Color"]}"><b>{d["TC"]}</b><br>{d["Value"]}<br>Z:{d["Z-Score"]}</span></div>'
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

    # --- Build all 3 rows as side-by-side subplot charts (Fix | Loose) ---
    for row_num, (fix_key, loose_key, fix_label, loose_label) in enumerate([
        ('fix_row1', 'loose_row1', 'Fix Side – ROW 1', 'Loose Side – ROW 1'),
        ('fix_row2', 'loose_row2', 'Fix Side – ROW 2', 'Loose Side – ROW 2'),
        ('fix_row3', 'loose_row3', 'Fix Side – ROW 3', 'Loose Side – ROW 3'),
    ], start=1):
        fig = make_subplots(rows=1, cols=2, subplot_titles=[fix_label, loose_label], horizontal_spacing=0.06)

        # Fix side traces
        for i, tc in enumerate(tc_groups[fix_key]):
            fig.add_trace(
                go.Scatter(x=window_df['TIME'], y=window_df[tc],
                           name=f'F-TC{tc.split("TC")[-1]}',
                           line=dict(color=colors[i % len(colors)]),
                           legendgroup=f'fix_r{row_num}', showlegend=True),
                row=1, col=1
            )
        fig.add_vline(x=current_time, line_dash="dash", line_color="red", row=1, col=1)

        # Loose side traces
        for i, tc in enumerate(tc_groups[loose_key]):
            fig.add_trace(
                go.Scatter(x=window_df['TIME'], y=window_df[tc],
                           name=f'L-TC{tc.split("TC")[-1]}',
                           line=dict(color=colors[i % len(colors)]),
                           legendgroup=f'loose_r{row_num}', showlegend=True),
                row=1, col=2
            )
        fig.add_vline(x=current_time, line_dash="dash", line_color="red", row=1, col=2)

        fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="°C", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True, config=_chart_config)

        # Z-Score parameter strips below the subplot
        zc1, zc2 = st.columns(2)
        with zc1:
            display_row_parameters(tc_groups[fix_key], f"Fix R{row_num}")
        with zc2:
            display_row_parameters(tc_groups[loose_key], f"Loose R{row_num}")

    # ====================================================================
    # 5. RULE 2 DEVIATION TABLE
    # ====================================================================
    st.markdown("### 📊 Rule 2: TC Deviation from Own Median")

    rule2_data = []
    for row_name, row_cols in [
        ('Fix ROW1', tc_groups['fix_row1']),
        ('Fix ROW2', tc_groups['fix_row2']),
        ('Fix ROW3', tc_groups['fix_row3']),
        ('Loose ROW1', tc_groups['loose_row1']),
        ('Loose ROW2', tc_groups['loose_row2']),
        ('Loose ROW3', tc_groups['loose_row3'])
    ]:
        max_deviation = 0
        max_tc = ""
        max_tc_median = 0
        for tc in row_cols:
            if tc in current_row and pd.notna(current_row[tc]):
                recent_start = max(0, current_idx - 60)
                tc_recent = df.iloc[recent_start:current_idx][tc].dropna()
                if len(tc_recent) > 0:
                    tc_median = tc_recent.median()
                    dev = abs(current_row[tc] - tc_median)
                    if dev > max_deviation:
                        max_deviation = dev
                        max_tc = tc
                        max_tc_median = tc_median

        if max_tc:
            rule2_data.append({
                'Row': row_name,
                'TC': max_tc.split('TC')[-1] if 'TC' in max_tc else max_tc[-3:],
                'Current': f"{current_row[max_tc]:.1f}°C",
                'Median': f"{max_tc_median:.1f}°C",
                'Dev': f"{max_deviation:.1f}°C",
                'Alert': '🔴' if max_deviation > 10 else ('⚠️' if max_deviation > 8 else '✓')
            })

    st.dataframe(pd.DataFrame(rule2_data), use_container_width=True, hide_index=True)

    # ====================================================================
    # 6. ROW TEMPERATURE DETAILS
    # ====================================================================
    with st.expander("🔍 Row Temperature Details", expanded=False):
        st.markdown("<small><b>Row Temperature Details (TC Median Values)</b></small>", unsafe_allow_html=True)
        detail_cols = st.columns(4)
        col_idx = 0
        for row_name, row_cols, side in [
            ('Fix ROW1', tc_groups['fix_row1'], 'Fix'),
            ('Fix ROW2', tc_groups['fix_row2'], 'Fix'),
            ('Fix ROW3', tc_groups['fix_row3'], 'Fix'),
            ('Loose ROW1', tc_groups['loose_row1'], 'Loose'),
            ('Loose ROW2', tc_groups['loose_row2'], 'Loose'),
            ('Loose ROW3', tc_groups['loose_row3'], 'Loose'),
        ]:
            with detail_cols[col_idx % 4]:
                row_values = current_row[row_cols].dropna()
                if len(row_values) > 0:
                    st.markdown(f"<small><b>{row_name}</b><br>Median: {row_values.median():.1f}°C<br>Range: {row_values.min():.1f}-{row_values.max():.1f}°C</small>", unsafe_allow_html=True)
            col_idx += 1

    # ====================================================================
    # 7. SHAP EXPLANATION  (skipped during playback for smoother transitions)
    # ====================================================================
    if not st.session_state.playing:
        st.markdown("### 🔍 SHAP Explanation – Why Anomalous?")

        shap_df = get_shap_explanation(current_row, model, scaler, feature_names)

        if shap_df is not None:
            fig = go.Figure()
            colors_shap = ['red' if x > 0 else 'blue' for x in shap_df['SHAP']]
            fig.add_trace(go.Bar(
                y=shap_df['Feature'], x=shap_df['SHAP'], orientation='h',
                marker=dict(color=colors_shap),
                text=[f"{v:.1f}" for v in shap_df['Value']], textposition='outside'
            ))
            fig.update_layout(
                title="Top 10 Features (Red → anomaly)", xaxis_title="SHAP Value",
                yaxis_title="Feature", height=350, showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            top_feature = shap_df.iloc[0]
            if abs(top_feature['SHAP']) > 0.01:
                st.error(f"🎯 **Primary Contributor**: {top_feature['Feature']} = {top_feature['Value']:.1f} (SHAP: {top_feature['SHAP']:+.3f})")
        else:
            st.warning("SHAP explanation not available (missing data)")
    else:
        st.markdown("<small><i>SHAP explanation paused during playback</i></small>", unsafe_allow_html=True)

    # ====================================================================
    # 8. ALERT HISTORY LOG  (below everything)
    # ====================================================================
    # Anchor for clickable session alerts to scroll to
    st.markdown('<div id="alert-history-log"></div>', unsafe_allow_html=True)
    with st.expander("📋 Alert History Log", expanded=True if st.session_state.alert_log else False):
        if st.session_state.alert_log:
            log_df = pd.DataFrame(st.session_state.alert_log)

            # Rule breakup summary
            _r1c = len(log_df[log_df['rule'] == 'Rule 1']) if 'rule' in log_df.columns else 0
            _r2c = len(log_df[log_df['rule'] == 'Rule 2']) if 'rule' in log_df.columns else 0
            _r3c = len(log_df[log_df['rule'] == 'Rule 3']) if 'rule' in log_df.columns else 0
            st.markdown(
                f'<div style="display:flex;gap:1rem;margin-bottom:0.4rem;font-size:0.82rem;">'
                f'<span style="background:#fde8e8;padding:2px 10px;border-radius:4px;border-left:3px solid #d62728;">'
                f'<b style="color:#d62728">R1</b>: {_r1c}</span>'
                f'<span style="background:#fff4e5;padding:2px 10px;border-radius:4px;border-left:3px solid #ff7f0e;">'
                f'<b style="color:#ff7f0e">R2</b>: {_r2c}</span>'
                f'<span style="background:#f3eefa;padding:2px 10px;border-radius:4px;border-left:3px solid #9467bd;">'
                f'<b style="color:#9467bd">R3</b>: {_r3c}</span>'
                f'<span style="color:#888;">Total: <b>{len(log_df)}</b></span>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Color-coded HTML table
            _rule_bg = {'Rule 1': '#fde8e8', 'Rule 2': '#fff4e5', 'Rule 3': '#f3eefa'}
            _rule_fg = {'Rule 1': '#d62728', 'Rule 2': '#ff7f0e', 'Rule 3': '#9467bd'}
            display_cols = ['index', 'time', 'rule', 'tc', 'description', 'actual_value', 'compared_with', 'action']
            available_cols = [c for c in display_cols if c in log_df.columns]
            log_tail = log_df[available_cols].tail(50)

            tbl = ('<table style="width:100%;border-collapse:collapse;font-size:0.78rem;">'
                   '<thead><tr style="background:#f0f2f6;">')
            col_labels = {'index': '#', 'time': 'Time', 'rule': 'Rule', 'tc': 'TC',
                          'description': 'Description', 'actual_value': 'Actual',
                          'compared_with': 'Compared With', 'action': 'Action'}
            for c in available_cols:
                tbl += f'<th style="padding:4px 6px;text-align:left;border-bottom:2px solid #ddd;">{col_labels.get(c, c)}</th>'
            tbl += '</tr></thead><tbody>'
            for _, row in log_tail.iterrows():
                _bg = _rule_bg.get(row.get('rule', ''), '#fff')
                _fg = _rule_fg.get(row.get('rule', ''), '#333')
                tbl += f'<tr style="background:{_bg};border-bottom:1px solid #eee;">'
                for c in available_cols:
                    val = str(row.get(c, ''))
                    style = f'color:{_fg};font-weight:bold;' if c == 'rule' else ''
                    tbl += f'<td style="padding:3px 6px;{style}">{val}</td>'
                tbl += '</tr>'
            tbl += '</tbody></table>'
            st.markdown(tbl, unsafe_allow_html=True)
        else:
            st.info("No alerts logged yet. Press ▶ to start monitoring.")

    # ====================================================================
    # PLAY AUTO-ADVANCE  (runs AFTER full page renders)
    # ====================================================================
    if st.session_state.playing:
        step_size = st.session_state.speed_multiplier
        next_idx = st.session_state.current_idx + step_size
        if next_idx >= total_rows:
            st.session_state.playing = False
            st.session_state.current_idx = total_rows - 1
        else:
            st.session_state.current_idx = next_idx
        time.sleep(0.1)
        st.rerun()

if __name__ == "__main__":
    main()
