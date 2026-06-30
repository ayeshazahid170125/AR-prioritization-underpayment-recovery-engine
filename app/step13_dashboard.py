"""
STEP 13 - AR Recovery Dashboard (Project 2)
WellMind Data Solutions - AR Prioritization & Underpayment Recovery Engine

Run: streamlit run app/step13_dashboard.py
Requires (Claim Checker tab only): uvicorn app.step12_fastapi:app --reload --port 8001

Tabs:
  1. AR Priority Queue     -- filterable/sortable workqueue + live recovery projections
                              (reads report_outputs/top_underpayments.csv, the Step 11
                              top-500-by-recovery file -- NOT the full multi-million-row
                              queue, by design, per Step 11's own closing note).
  2. Underpayment Report   -- $ gap by HCPCS / state / provider type / payer type
                              (reads the four Step 11 summary CSVs).
  3. Claim Checker         -- single-claim recovery-priority scorer via the Step 12
                              FastAPI endpoint (this is the original Step 13 UI).
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL  = "http://localhost:8001"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT

# ── DATA SOURCES (Step 09 / Step 11 outputs) ────────────────
TOP_UNDERPAY_PATH    = BASE_DIR / "report_outputs" / "top_underpayments.csv"
SUMMARY_HCPCS_PATH   = BASE_DIR / "report_outputs" / "underpayment_summary_by_hcpcs.csv"
SUMMARY_STATE_PATH   = BASE_DIR / "report_outputs" / "underpayment_summary_by_state.csv"
SUMMARY_PROVIDER_PATH= BASE_DIR / "report_outputs" / "underpayment_summary_by_provider_type.csv"
SUMMARY_PAYER_PATH   = BASE_DIR / "report_outputs" / "underpayment_summary_by_payer_type.csv"
EXEC_SUMMARY_PATH    = BASE_DIR / "report_outputs" / "underpayment_report_summary.csv"

st.set_page_config(
    page_title="AR Recovery Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── THEME ──────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"

THEME_TOKENS = {
    "dark": {
        "bg-page":              "#0B0F19",
        "bg-navbar":            "rgba(6,10,19,0.85)",
        "bg-card":              "rgba(21,29,48,0.45)",
        "bg-form-panel":        "rgba(21,29,48,0.34)",
        "bg-input":             "#111827",
        "bg-input-hover":       "#162033",
        "bg-track":             "rgba(255,255,255,0.05)",
        "shadow-color":         "rgba(0,0,0,0.3)",
        "card-hover-shadow":    "rgba(59,130,246,0.15)",
        "border-color":         "rgba(255,255,255,0.06)",
        "border-color-soft":    "rgba(255,255,255,0.04)",
        "border-color-strong":  "rgba(255,255,255,0.08)",
        "input-border":         "#1E293B",
        "text-primary":         "#F8FAFC",
        "text-secondary":       "#94A3B8",
        "text-tertiary":        "#CBD5E1",
        "text-muted":           "#64748B",
        "text-faint":           "#475569",
        "input-text":           "#F1F5F9",
        "accent":               "#3B82F6",
        "accent-light":         "#60A5FA",
        "accent-lighter":       "#38BDF8",
        "success-text":         "#34D399",
        "warning-text":         "#FBBF24",
        "danger-text":          "#F87171",
        "nav-logo-bg-1":        "rgba(37,99,235,0.3)",
        "nav-logo-bg-2":        "rgba(13,148,136,0.3)",
        "nav-logo-border":      "rgba(59,130,246,0.35)",
        "nav-badge-bg":         "rgba(59,130,246,0.08)",
        "nav-badge-border":     "rgba(59,130,246,0.18)",
        "nav-badge-text":       "#60A5FA",
        "top-header-bg-1":      "rgba(37,99,235,0.12)",
        "top-header-bg-2":      "rgba(13,148,136,0.12)",
        "top-header-border":    "rgba(59,130,246,0.15)",
        "cause-bg-1":           "rgba(30,58,138,0.2)",
        "cause-bg-2":           "rgba(6,78,59,0.2)",
        "cause-border":         "rgba(59,130,246,0.2)",
        "cause-label":          "#60A5FA",
        "cause-conf":           "#93C5FD",
        "fix-label-text":       "#34D399",
        "fix-text-text":        "#A7F3D0",
        "pill-bg":              "rgba(255,255,255,0.04)",
        "pill-border":          "rgba(255,255,255,0.08)",
        "pill-text":            "#CBD5E1",
        "badge-high-bg":        "rgba(239,68,68,0.1)",
        "badge-high-text":      "#F87171",
        "badge-high-border":    "rgba(239,68,68,0.3)",
        "badge-med-bg":         "rgba(245,158,11,0.1)",
        "badge-med-text":       "#FBBF24",
        "badge-med-border":     "rgba(245,158,11,0.3)",
        "badge-low-bg":         "rgba(16,185,129,0.1)",
        "badge-low-text":       "#34D399",
        "badge-low-border":     "rgba(16,185,129,0.3)",
        "ready-icon-bg-1":      "rgba(37,99,235,0.15)",
        "ready-icon-bg-2":      "rgba(13,148,136,0.15)",
        "ready-icon-border":    "rgba(59,130,246,0.2)",
        "ready-subtitle-strong":"#94A3B8",
        "feature-bg":           "rgba(255,255,255,0.02)",
        "feature-border":       "rgba(255,255,255,0.04)",
        "fi-blue-bg":           "rgba(59,130,246,0.12)",
        "fi-blue-border":       "rgba(59,130,246,0.15)",
        "fi-blue-text":         "#38BDF8",
        "fi-amber-bg":          "rgba(251,191,36,0.10)",
        "fi-amber-border":      "rgba(251,191,36,0.15)",
        "fi-amber-text":        "#FBBF24",
        "fi-green-bg":          "rgba(52,211,153,0.10)",
        "fi-green-border":      "rgba(52,211,153,0.15)",
        "fi-green-text":        "#34D399",
        "fi-red-bg":            "rgba(248,113,113,0.10)",
        "fi-red-border":        "rgba(248,113,113,0.15)",
        "fi-red-text":          "#F87171",
        "toggle-bg":            "rgba(21,29,48,0.7)",
        "toggle-border":        "rgba(255,255,255,0.12)",
        "toggle-text":          "#F8FAFC",
    },
    "light": {
        "bg-page":              "#F4F6FB",
        "bg-navbar":            "rgba(255,255,255,0.88)",
        "bg-card":              "rgba(255,255,255,0.85)",
        "bg-form-panel":        "rgba(255,255,255,0.62)",
        "bg-input":             "#FFFFFF",
        "bg-input-hover":       "#F8FAFC",
        "bg-track":             "rgba(15,23,42,0.07)",
        "shadow-color":         "rgba(15,23,42,0.08)",
        "card-hover-shadow":    "rgba(37,99,235,0.12)",
        "border-color":         "rgba(15,23,42,0.08)",
        "border-color-soft":    "rgba(15,23,42,0.05)",
        "border-color-strong":  "rgba(15,23,42,0.12)",
        "input-border":         "#CBD5E1",
        "text-primary":         "#0F172A",
        "text-secondary":       "#475569",
        "text-tertiary":        "#334155",
        "text-muted":           "#64748B",
        "text-faint":           "#94A3B8",
        "input-text":           "#0F172A",
        "accent":               "#2563EB",
        "accent-light":         "#1D4ED8",
        "accent-lighter":       "#2563EB",
        "success-text":         "#047857",
        "warning-text":         "#92400E",
        "danger-text":          "#B91C1C",
        "nav-logo-bg-1":        "rgba(37,99,235,0.16)",
        "nav-logo-bg-2":        "rgba(13,148,136,0.16)",
        "nav-logo-border":      "rgba(37,99,235,0.3)",
        "nav-badge-bg":         "rgba(37,99,235,0.08)",
        "nav-badge-border":     "rgba(37,99,235,0.2)",
        "nav-badge-text":       "#1D4ED8",
        "top-header-bg-1":      "rgba(37,99,235,0.07)",
        "top-header-bg-2":      "rgba(13,148,136,0.07)",
        "top-header-border":    "rgba(37,99,235,0.16)",
        "cause-bg-1":           "rgba(37,99,235,0.06)",
        "cause-bg-2":           "rgba(13,148,136,0.06)",
        "cause-border":         "rgba(37,99,235,0.18)",
        "cause-label":          "#1D4ED8",
        "cause-conf":           "#2563EB",
        "fix-label-text":       "#047857",
        "fix-text-text":        "#065F46",
        "pill-bg":              "rgba(15,23,42,0.04)",
        "pill-border":          "rgba(15,23,42,0.09)",
        "pill-text":            "#334155",
        "badge-high-bg":        "rgba(239,68,68,0.1)",
        "badge-high-text":      "#B91C1C",
        "badge-high-border":    "rgba(239,68,68,0.32)",
        "badge-med-bg":         "rgba(217,119,6,0.12)",
        "badge-med-text":       "#92400E",
        "badge-med-border":     "rgba(217,119,6,0.32)",
        "badge-low-bg":         "rgba(5,150,105,0.1)",
        "badge-low-text":       "#047857",
        "badge-low-border":     "rgba(5,150,105,0.32)",
        "ready-icon-bg-1":      "rgba(37,99,235,0.1)",
        "ready-icon-bg-2":      "rgba(13,148,136,0.1)",
        "ready-icon-border":    "rgba(37,99,235,0.2)",
        "ready-subtitle-strong":"#334155",
        "feature-bg":           "rgba(15,23,42,0.02)",
        "feature-border":       "rgba(15,23,42,0.05)",
        "fi-blue-bg":           "rgba(37,99,235,0.1)",
        "fi-blue-border":       "rgba(37,99,235,0.18)",
        "fi-blue-text":         "#1D4ED8",
        "fi-amber-bg":          "rgba(217,119,6,0.1)",
        "fi-amber-border":      "rgba(217,119,6,0.18)",
        "fi-amber-text":        "#92400E",
        "fi-green-bg":          "rgba(5,150,105,0.1)",
        "fi-green-border":      "rgba(5,150,105,0.18)",
        "fi-green-text":        "#047857",
        "fi-red-bg":            "rgba(220,38,38,0.1)",
        "fi-red-border":        "rgba(220,38,38,0.18)",
        "fi-red-text":          "#B91C1C",
        "toggle-bg":            "rgba(255,255,255,0.9)",
        "toggle-border":        "rgba(15,23,42,0.12)",
        "toggle-text":          "#0F172A",
    },
}

theme = st.session_state.theme
tokens = THEME_TOKENS[theme]
_root_vars = "\n".join(f"    --{k}: {v};" for k, v in tokens.items())
st.markdown(f"<style>\n:root {{\n{_root_vars}\n}}\n</style>", unsafe_allow_html=True)

# ── MAIN STYLESHEET (single injection -- the original file injected this
# whole block twice; that duplicate has been removed here) ──────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) none;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
}
.st-emotion-cache-11xx4re {
    -webkit-box-align: center;
    align-items: center;
    background-color: rgb(255, 75, 75) !important;
    border-radius: 100%;
    border-style: none;
    display: flex;
    -webkit-box-pack: center;
    justify-content: center;
    height: 0.75rem;
    width: 0.75rem;
    box-shadow: none;
}

html, body {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}
::selection {
    background: rgba(37, 99, 235, 0.18);
    color: var(--text-primary);
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
a, button, select, input, textarea, .card, .stat-card, .pill {
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
# .st-c1 {
#     color: white;!important;
# }

/* Hide sidebar completely */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Main background */
.main { background: var(--bg-page) !important; }
[data-testid="stAppViewContainer"] { background: var(--bg-page) !important; }

/* Hide defaults */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { display: none !important; }

/* ── THEME TOGGLE BUTTON — circular icon FAB, premium glassy feel ── */
div[data-testid="stButton"] {
    position: fixed;
    top: 14px;
    right: 32px;
    z-index: 1001;
    width: auto !important;
}
div[data-testid="stButton"] > button {
    position: relative;
    background: linear-gradient(150deg, var(--toggle-bg), var(--bg-card)) !important;
    border: 1.5px solid var(--toggle-border) !important;
    color: var(--toggle-text) !important;
    border-radius: 50% !important;
    width: 46px !important;
    height: 46px !important;
    min-width: 46px !important;
    min-height: 46px !important;
    padding: 0 !important;
    line-height: 1 !important;
    font-size: 19px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow:
        0 4px 18px var(--shadow-color),
        0 0 0 1px var(--border-color-soft) inset !important;
    transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1),
                box-shadow 0.3s ease,
                border-color 0.3s ease !important;
}
div[data-testid="stButton"] > button p {
    font-size: 19px !important;
    line-height: 1 !important;
    margin: 0 !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    box-shadow:
        0 6px 22px var(--card-hover-shadow),
        0 0 16px var(--card-hover-shadow),
        0 0 0 1px var(--accent) inset !important;
    transform: translateY(-2px) rotate(18deg) scale(1.08) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) rotate(18deg) scale(0.92) !important;
}
div[data-testid="stButton"] > button:focus {
    outline: none !important;
    box-shadow:
        0 4px 18px var(--shadow-color),
        0 0 0 3px var(--card-hover-shadow) !important;
}

/* ── NAVBAR ── */
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: var(--bg-navbar);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-color);
    padding: 0 190px 0 36px;
    height: 62px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-logo {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, var(--nav-logo-bg-1), var(--nav-logo-bg-2));
    border: 1.5px solid var(--nav-logo-border);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; color: var(--accent-lighter);
}
.nav-tagline {
    font-size: 11px; font-weight: 600; color: var(--text-faint);
    text-transform: uppercase; letter-spacing: 0.08em;
}
.nav-right {
    display: flex;
    align-items: center;
    gap: 20px;
}
.nav-status {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 13px; font-weight: 500;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
}
.status-connected {
    background-color: #10B981;
    box-shadow: 0 0 8px #10B981;
    animation: pulse-c 2s infinite;
}
.status-disconnected {
    background-color: #EF4444;
    box-shadow: 0 0 8px #EF4444;
    animation: pulse-d 2s infinite;
}
@keyframes pulse-c {
    0%   { box-shadow: 0 0 0 0   rgba(16,185,129,0.7); }
    70%  { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0   rgba(16,185,129,0); }
}
@keyframes pulse-d {
    0%   { box-shadow: 0 0 0 0   rgba(239,68,68,0.7); }
    70%  { box-shadow: 0 0 0 6px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0   rgba(239,68,68,0); }
}
.nav-badge {
    background: var(--nav-badge-bg);
    border: 1px solid var(--nav-badge-border);
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
    color: var(--nav-badge-text);
    letter-spacing: 0.04em;
}

/* Top header banner */
.top-header {
    background: linear-gradient(135deg, var(--top-header-bg-1) 0%, var(--top-header-bg-2) 100%) !important;
    border: 1px solid var(--top-header-border) !important;
    padding: 32px 36px !important;
    border-radius: 16px !important;
    margin-bottom: 30px !important;
    box-shadow: 0 8px 32px 0 var(--shadow-color) !important;
    backdrop-filter: blur(8px);
}
.top-header h1 {
    color: var(--text-primary) !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.02em !important;
}
.top-header p {
    color: var(--text-secondary) !important;
    font-size: 14px !important;
    margin: 6px 0 0 0 !important;
}

/* Cards */
.card, .stat-card {
    background: var(--bg-card) !important;
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px 0 var(--shadow-color) !important;
    margin-bottom: 20px !important;
}
.card:hover, .stat-card:hover {
    border-color: rgba(59,130,246,0.35) !important;
    box-shadow: 0 8px 32px 0 var(--card-hover-shadow) !important;
    transform: translateY(-2px);
}
.card-title {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-bottom: 20px !important;
}

/* Risk badges */
.badge-HIGH {
    display: inline-block;
    background: var(--badge-high-bg) !important; color: var(--badge-high-text) !important;
    border: 1.5px solid var(--badge-high-border) !important;
    padding: 6px 22px !important; border-radius: 100px !important;
    font-weight: 700 !important; font-size: 14px !important;
    letter-spacing: 0.08em !important; box-shadow: 0 0 15px rgba(239,68,68,0.15) !important;
}
.badge-MEDIUM {
    display: inline-block;
    background: var(--badge-med-bg) !important; color: var(--badge-med-text) !important;
    border: 1.5px solid var(--badge-med-border) !important;
    padding: 6px 22px !important; border-radius: 100px !important;
    font-weight: 700 !important; font-size: 14px !important;
    letter-spacing: 0.08em !important; box-shadow: 0 0 15px rgba(245,158,11,0.15) !important;
}
.badge-LOW {
    display: inline-block;
    background: var(--badge-low-bg) !important; color: var(--badge-low-text) !important;
    border: 1.5px solid var(--badge-low-border) !important;
    padding: 6px 22px !important; border-radius: 100px !important;
    font-weight: 700 !important; font-size: 14px !important;
    letter-spacing: 0.08em !important; box-shadow: 0 0 15px rgba(16,185,129,0.15) !important;
}

/* Driver bars */
.driver-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.driver-name { font-family: 'Fira Code', monospace !important; font-size: 12px !important; color: var(--text-tertiary) !important; min-width: 190px !important; }
.driver-bar-wrap { flex: 1; height: 8px; background: var(--bg-track); border-radius: 6px; overflow: hidden; }
.driver-bar-high { height: 8px; background: linear-gradient(90deg, #EF4444, #F87171) !important; border-radius: 6px; }
.driver-bar-low  { height: 8px; background: linear-gradient(90deg, #10B981, #34D399) !important; border-radius: 6px; }
.driver-val { font-family: 'Fira Code', monospace !important; font-size: 12px !important; color: var(--text-secondary) !important; min-width: 60px !important; text-align: right !important; }

/* Cause card */
.cause-card {
    background: linear-gradient(135deg, var(--cause-bg-1) 0%, var(--cause-bg-2) 100%) !important;
    border: 1px solid var(--cause-border) !important;
    border-radius: 16px !important; padding: 24px !important;
    margin-bottom: 16px !important; box-shadow: 0 8px 32px 0 var(--shadow-color) !important;
}
.cause-label { font-size: 18px !important; font-weight: 700 !important; color: var(--cause-label) !important; margin-bottom: 4px !important; }
.cause-conf { font-size: 13px !important; color: var(--cause-conf) !important; font-weight: 500 !important; }
.fix-label { font-size: 11px !important; font-weight: 700 !important; color: var(--fix-label-text) !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
.fix-text { font-size: 14px !important; color: var(--fix-text-text) !important; margin-top: 6px !important; line-height: 1.6 !important; }
.pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
.pill {
    background: var(--pill-bg) !important; border: 1px solid var(--pill-border) !important;
    border-radius: 100px !important; padding: 6px 14px !important;
    font-size: 12px !important; color: var(--pill-text) !important;
}

/* Form section labels */
[data-testid="stForm"] {
    background: var(--bg-form-panel) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 18px !important;
    padding: 22px 22px 24px 22px !important;
    box-shadow: 0 10px 34px var(--shadow-color) !important;
}

.form-section {
    font-size: 12px !important; font-weight: 700 !important; color: var(--accent-light) !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    margin: 28px 0 12px 0 !important;
    border-bottom: 1px solid var(--top-header-border) !important; padding-bottom: 6px !important;
}

/* Submit button */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2563EB, #0D9488) !important;
    color: #FFFFFF !important; border: none !important;
    border-radius: 12px !important; padding: 16px 0 !important;
    font-size: 16px !important; font-weight: 600 !important; width: 100% !important;
    letter-spacing: 0.05em !important; box-shadow: 0 4px 20px rgba(37,99,235,0.25) !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #0F766E) !important;
    box-shadow: 0 6px 24px rgba(37,99,235,0.4) !important;
    transform: translateY(-2px) !important;
}

/* Inputs */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stMultiSelect"] [data-baseweb="select"] {
    background: var(--bg-input) !important;
    border: 1.5px solid var(--input-border) !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] div {
    background-color: transparent !important;
    color: var(--input-text) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] svg,
[data-testid="stMultiSelect"] [data-baseweb="select"] span,
[data-testid="stMultiSelect"] [data-baseweb="select"] svg {
    color: var(--input-text) !important;
    fill: var(--input-text) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:hover,
[data-testid="stMultiSelect"] [data-baseweb="select"]:hover {
    background: var(--bg-input-hover) !important;
    border-color: var(--border-color-strong) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: var(--bg-input) !important;
    color: var(--input-text) !important;
    border: 1.5px solid var(--input-border) !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04) !important;
}
[data-testid="stTextInput"] input:hover,
[data-testid="stTextArea"] textarea:hover {
    background-color: var(--bg-input-hover) !important;
    border-color: var(--border-color-strong) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
[data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    background-color: var(--bg-input) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color: var(--text-faint) !important;
    opacity: 0.75 !important;
}
label[data-testid="stWidgetLabel"] p { color: var(--text-secondary) !important; font-weight: 600 !important; font-size: 13px !important; }
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] div[role="radiogroup"] label,
[data-testid="stRadio"] div[role="radiogroup"] label p,
[data-testid="stRadio"] div[role="radiogroup"] label span {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}
[data-testid="stRadio"] [data-baseweb="radio"] {
    color: var(--text-primary) !important;
}
[data-testid="stRadio"] [role="radio"] {
    border-color: var(--input-border) !important;
    background-color: var(--bg-input) !important;
}
[data-testid="stRadio"] [role="radio"][aria-checked="true"] {
    border-color: var(--accent) !important;
    background-color: var(--accent) !important;
}
[data-testid="stCheckbox"] label {
    align-items: center !important;
}
[data-testid="stCheckbox"] label p { color: var(--text-tertiary) !important; font-weight: 500 !important; }
[data-testid="stCheckbox"] [data-testid="stCheckboxRoot"] > div {
    border-color: var(--input-border) !important;
    background: var(--bg-input) !important;
}
[data-testid="stCheckbox"] [data-testid="stCheckboxRoot"] > div[aria-checked="true"] {
    border-color: var(--accent) !important;
    background: var(--accent) !important;
}
[data-testid="stSlider"] [data-testid="stWidgetLabel"] { min-height: 40px !important; display: flex !important; align-items: flex-end !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child { background: var(--bg-track) !important; }

/* Dropdown / select menus (BaseWeb portals) */
[data-baseweb="popover"] div[role="listbox"],
[data-baseweb="menu"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color-strong) !important;
    box-shadow: 0 16px 36px var(--shadow-color) !important;
}
[data-baseweb="menu"] li,
ul[role="listbox"] li {
    color: var(--input-text) !important;
    background: var(--bg-input) !important;
}
[data-baseweb="menu"] li:hover,
ul[role="listbox"] li:hover {
    background: var(--bg-input-hover) !important;
}

/* Disclaimer */
.disclaimer-bar {
    background: var(--bg-track) !important; border: 1px solid var(--border-color) !important;
    border-radius: 10px !important; padding: 12px 18px !important;
    font-size: 11px !important; color: var(--text-muted) !important;
    margin-top: 16px !important; line-height: 1.5 !important;
}

/* Ready card */
.ready-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 36px 24px;
    box-shadow: 0 8px 32px 0 var(--shadow-color);
    margin-top: 24px;
    text-align: center;
}
.ready-icon {
    width: 52px; height: 52px; border-radius: 14px;
    background: linear-gradient(135deg, var(--ready-icon-bg-1), var(--ready-icon-bg-2));
    border: 1.5px solid var(--ready-icon-border);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 18px auto; font-size: 26px; color: var(--accent-lighter);
}
.ready-title { font-size: 19px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; letter-spacing: -0.02em; }
.ready-subtitle { font-size: 13px; color: var(--text-muted); line-height: 1.6; margin-bottom: 28px; }
.ready-subtitle strong { color: var(--ready-subtitle-strong); }
.feature-list { display: flex; flex-direction: column; gap: 10px; text-align: left; }
.feature-item {
    display: flex; align-items: center; gap: 14px; padding: 14px 16px;
    background: var(--feature-bg); border: 1px solid var(--feature-border); border-radius: 12px;
}
.feature-icon {
    width: 38px; height: 38px; border-radius: 10px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 20px;
}
.fi-blue  { background: var(--fi-blue-bg); border: 1px solid var(--fi-blue-border); color: var(--fi-blue-text); }
.fi-amber { background: var(--fi-amber-bg); border: 1px solid var(--fi-amber-border); color: var(--fi-amber-text); }
.fi-green { background: var(--fi-green-bg); border: 1px solid var(--fi-green-border); color: var(--fi-green-text); }
.fi-red   { background: var(--fi-red-bg); border: 1px solid var(--fi-red-border); color: var(--fi-red-text); }
.feature-title { font-size: 14px; font-weight: 600; color: var(--text-primary); }
.feature-desc  { font-size: 12px; color: var(--text-muted); margin-top: 3px; }

/* ── NEW: tabs, KPI stat grid, filter bar ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border-color); }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--text-secondary) !important;
    font-weight: 600 !important; font-size: 14px !important; padding: 10px 6px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important; border-bottom: 2.5px solid var(--accent) !important;
}
.kpi-value { font-size: 22px; font-weight: 800; color: var(--text-primary); font-family: 'Plus Jakarta Sans', sans-serif; }
.kpi-label { font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; border: 1px solid var(--border-color) !important; }
</style>
""", unsafe_allow_html=True)

toggle_icon = "☀️" if theme == "dark" else "🌙"
if st.button(toggle_icon, key="theme_toggle_btn", help="Switch theme"):
    st.session_state.theme = "light" if theme == "dark" else "dark"
    st.rerun()


# ── HELPERS ─────────────────────────────────────────────────
def api_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=4).status_code == 200
    except Exception:
        return False


def call_api(payload):
    try:
        r = requests.post(f"{API_URL}/predict/recovery", json=payload, timeout=30)
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        st.error("FastAPI offline — run: uvicorn step12_fastapi:app --reload --port 8001")
        return None


def tier_color(tier):
    palette = {
        "dark":  {"Critical": "#F87171", "High": "#F87171", "Medium": "#FBBF24", "Low": "#34D399"},
        "light": {"Critical": "#B91C1C", "High": "#B91C1C", "Medium": "#92400E", "Low": "#047857"},
    }
    return palette[st.session_state.get("theme", "light")].get(tier, "#94A3B8")


def gauge_theme():
    if st.session_state.get("theme", "light") == "light":
        return {
            "tickcolor": "#94A3B8", "tickfont": "#475569",
            "track_bg": "rgba(15,23,42,0.07)", "title_font": "#475569",
            "threshold": "#0F172A",
            "steps": ["rgba(5,150,105,0.08)", "rgba(217,119,6,0.08)", "rgba(220,38,38,0.08)"],
        }
    return {
        "tickcolor": "#475569", "tickfont": "#94A3B8",
        "track_bg": "rgba(255,255,255,0.05)", "title_font": "#94A3B8",
        "threshold": "#FFFFFF",
        "steps": ["rgba(16,185,129,0.06)", "rgba(245,158,11,0.06)", "rgba(239,68,68,0.06)"],
    }


def chart_theme():
    if st.session_state.get("theme", "light") == "light":
        return {
            "font_color": "#334155",
            "axis_color": "#475569",
            "title_color": "#1E293B",
            "grid_color": "rgba(15,23,42,0.12)",
            "bar_color": "#2563EB",
        }
    return {
        "font_color": "#CBD5E1",
        "axis_color": "#CBD5E1",
        "title_color": "#F8FAFC",
        "grid_color": "rgba(255,255,255,0.12)",
        "bar_color": "#3B82F6",
    }


def horizontal_bar(df, x_col, y_col, title, color=None):
    """Reusable horizontal bar chart, themed to match the rest of the dashboard."""
    ct = chart_theme()
    fig = go.Figure(go.Bar(
        x=df[x_col], y=df[y_col].astype(str), orientation="h",
        marker_color=color or ct["bar_color"],
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=ct["title_color"], family="Plus Jakarta Sans, sans-serif")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ct["font_color"], family="Inter, sans-serif", size=11),
        xaxis=dict(
            gridcolor=ct["grid_color"],
            zerolinecolor=ct["grid_color"],
            linecolor=ct["grid_color"],
            tickfont=dict(color=ct["axis_color"], size=11),
            title_font=dict(color=ct["axis_color"]),
        ),
        yaxis=dict(
            tickfont=dict(color=ct["axis_color"], size=11),
            title_font=dict(color=ct["axis_color"]),
        ),
        margin=dict(t=40, b=10, l=10, r=10),
        height=max(280, 28 * len(df)),
    )
    return fig


def dataframe_theme():
    if st.session_state.get("theme", "light") == "light":
        return {
            "header_bg": "#E2E8F0",
            "cell_bg": "#FFFFFF",
            "cell_alt_bg": "#F8FAFC",
            "text": "#0F172A",
            "muted": "#334155",
            "border": "#CBD5E1",
        }
    return {
        "header_bg": "#1A1D25",
        "cell_bg": "#0B0F16",
        "cell_alt_bg": "#0F141D",
        "text": "#F8FAFC",
        "muted": "#CBD5E1",
        "border": "#2A2F3A",
    }


def styled_dataframe(df):
    dt = dataframe_theme()
    return (
        df.style
        .set_properties(**{
            "background-color": dt["cell_bg"],
            "color": dt["text"],
            "border-color": dt["border"],
        })
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", dt["header_bg"]),
                    ("color", dt["muted"]),
                    ("border-color", dt["border"]),
                    ("font-weight", "700"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [
                    ("background-color", dt["cell_alt_bg"]),
                    ("color", dt["text"]),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("border-color", dt["border"]),
                ],
            },
        ])
    )


def kpi_row(items):
    """items: list of (label, value) tuples -- rendered as stat-cards."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"""<div class="stat-card" style="text-align:center;">
                        <div class="kpi-value">{value}</div>
                        <div class="kpi-label">{label}</div>
                    </div>""",
                unsafe_allow_html=True,
            )


@st.cache_data(ttl=300)
def load_csv_safe(path_str):
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def missing_file_notice(*paths):
    lines = "\n".join(f"- `{p.relative_to(BASE_DIR) if p.is_relative_to(BASE_DIR) else p}`" for p in paths)
    st.warning(
        "No data found yet for this tab. Run the pipeline first, then refresh:\n\n"
        "1. `python step09_ar_priority_queue.py`\n"
        "2. `python step11_underpayment_report.py`\n\n"
        f"Expected file(s):\n{lines}"
    )


# ── NAVBAR ───────────────────────────────────────────────────
ok = api_health()
status_dot_cls = "status-connected" if ok else "status-disconnected"
status_text    = "<span style='color:var(--success-text);'>API Connected</span>" if ok else "<span style='color:var(--danger-text);'>API Offline</span>"

st.markdown(f"""
<div class="navbar">
    <div class="nav-brand">
        <div class="nav-logo">💰</div>
        <div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:18px; font-weight:800;
                color:var(--text-primary); letter-spacing:-0.03em; line-height:1.1;">
                AR Recovery Engine
            </div>
            <div class="nav-tagline">CMS Medicare PUF 2023 &middot; Underpayment Detection</div>
        </div>
    </div>
    <div class="nav-right">
        <span class="nav-badge">&#128178; Live Demo</span>
        <div class="nav-status">
            <span class="status-dot {status_dot_cls}"></span>
            {status_text}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not ok:
    st.info("ℹ️ FastAPI is only needed for the **Claim Checker** tab — run: `uvicorn step12_fastapi:app --reload --port 8001`")

# ── PAGE HEADER ──────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <div>
        <h1>AR Recovery &amp; Underpayment Intelligence</h1>
        <p>Prioritized AR workqueue, underpayment analytics, and single-claim recovery scoring &mdash; built on the CMS Medicare PUF 2023.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TAB 1 — AR PRIORITY QUEUE
# ============================================================
def render_ar_queue_tab():
    queue_df = load_csv_safe(str(TOP_UNDERPAY_PATH))
    if queue_df is None:
        missing_file_notice(TOP_UNDERPAY_PATH)
        return

    df = queue_df.copy()
    _queue_caption_legacy = (
        "Showing the **Top 500** highest-value underpaid claim groups. "
        "(the full queue can be millions of rows — see `ar_priority_outputs/ar_priority_queue.csv` "
        ""
    )

    st.markdown('<div class="form-section" style="margin-top:0;">Filters</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        provider_opts = sorted(df["Rndrng_Prvdr_Type"].dropna().unique().tolist()) if "Rndrng_Prvdr_Type" in df else []
        sel_provider = st.multiselect("Provider Type", provider_opts)
    with f2:
        state_opts = sorted(df["Rndrng_Prvdr_State_Abrvtn"].dropna().unique().tolist()) if "Rndrng_Prvdr_State_Abrvtn" in df else []
        sel_state = st.multiselect("State", state_opts)
    with f3:
        tier_order = ["Critical", "High", "Medium", "Low"]
        tier_opts = [t for t in tier_order if "priority_tier" in df and t in df["priority_tier"].unique()]
        sel_tier = st.multiselect("Priority Tier", tier_opts)
    with f4:
        payer_opts = sorted(df["payer_type_proxy"].dropna().unique().tolist()) if "payer_type_proxy" in df else []
        sel_payer = st.multiselect("Payer Type", payer_opts)

    f5, f6 = st.columns([1, 2])
    with f5:
        hcpcs_search = st.text_input("HCPCS code contains", "")
    with f6:
        max_recovery = float(df["estimated_recovery"].max()) if "estimated_recovery" in df and len(df) else 0.0
        step = max(1.0, round(max_recovery / 100, 2)) if max_recovery > 0 else 1.0
        min_recovery = st.slider("Min estimated recovery ($)", 0.0, max_recovery, 0.0, step=step)

    filtered = df.copy()
    if sel_provider:
        filtered = filtered[filtered["Rndrng_Prvdr_Type"].isin(sel_provider)]
    if sel_state:
        filtered = filtered[filtered["Rndrng_Prvdr_State_Abrvtn"].isin(sel_state)]
    if sel_tier:
        filtered = filtered[filtered["priority_tier"].isin(sel_tier)]
    if sel_payer:
        filtered = filtered[filtered["payer_type_proxy"].isin(sel_payer)]
    if hcpcs_search:
        filtered = filtered[filtered["HCPCS_Cd"].astype(str).str.contains(hcpcs_search, case=False, na=False)]
    if "estimated_recovery" in filtered:
        filtered = filtered[filtered["estimated_recovery"] >= min_recovery]

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    avg_conf = filtered["confidence_score"].mean() if "confidence_score" in filtered and len(filtered) else None
    kpi_row([
        ("Claims in view", f"{len(filtered):,}"),
        ("Total est. recovery", f"${filtered['estimated_recovery'].sum():,.0f}" if "estimated_recovery" in filtered else "—"),
        ("Critical + High", f"{filtered['priority_tier'].isin(['Critical', 'High']).sum():,}" if "priority_tier" in filtered else "—"),
        ("Avg confidence", f"{avg_conf * 100:.0f}%" if avg_conf is not None else "—"),
    ])

    sort_col_map = {
        "Estimated recovery ($)": "estimated_recovery",
        "Confidence score": "confidence_score",
        "Payment gap %": "Payment_Gap_Pct",
    }
    sort_col_map = {k: v for k, v in sort_col_map.items() if v in filtered.columns}

    s1, s2 = st.columns([3, 1])
    with s1:
        sort_label = st.selectbox("Sort by", list(sort_col_map.keys())) if sort_col_map else None
    with s2:
        sort_dir = st.radio("Order", ["Desc", "Asc"], horizontal=True)

    if sort_label:
        filtered = filtered.sort_values(sort_col_map[sort_label], ascending=(sort_dir == "Asc"))

    display_cols = [c for c in [
        "report_rank", "Rndrng_Prvdr_Type", "Rndrng_Prvdr_State_Abrvtn", "HCPCS_Cd",
        "Place_Of_Srvc", "payer_type_proxy", "estimated_recovery", "Payment_Gap_Pct",
        "confidence_score", "priority_tier",
    ] if c in filtered.columns]

    rename_map = {
        "report_rank": "Rank", "Rndrng_Prvdr_Type": "Provider Type",
        "Rndrng_Prvdr_State_Abrvtn": "State", "HCPCS_Cd": "HCPCS",
        "Place_Of_Srvc": "Setting", "payer_type_proxy": "Payer Type (proxy)",
        "estimated_recovery": "Est. Recovery ($)", "Payment_Gap_Pct": "Gap %",
        "confidence_score": "Confidence", "priority_tier": "Tier",
    }

    st.markdown('<div class="card-title" style="margin-top:24px;">AR Priority Queue</div>', unsafe_allow_html=True)
    display_df = filtered[display_cols].rename(columns=rename_map)
    st.dataframe(
        styled_dataframe(display_df),
        use_container_width=True, height=420, hide_index=True,
    )
    st.caption(
        "Showing the **Top 500** highest-value underpaid claim groups. "
        "Filters and totals below apply to this top-priority review sample."
    )

    if len(filtered) and "estimated_recovery" in filtered:
        top10 = filtered.head(10).copy()
        top10["label"] = top10.get("HCPCS_Cd", "").astype(str) + " · " + top10.get("Rndrng_Prvdr_State_Abrvtn", "").astype(str)
        st.plotly_chart(
            horizontal_bar(top10.iloc[::-1], "estimated_recovery", "label",
                           "Top 10 in Current View — Estimated Recovery ($)"),
            use_container_width=True,
        )

    _queue_disclaimer_legacy = (
        "<div class='disclaimer-bar'>Source: Step 11 <code>top_underpayments.csv</code>. "
        "CMS Medicare PUF 2023 &middot; Portfolio demo — not real recoverable amounts.</div>",
    )
    st.markdown(
        "<div class='disclaimer-bar'>CMS Medicare PUF 2023 &middot; Portfolio demo. "
        "Estimated recovery amounts are prioritization signals, not confirmed recoverable dollars.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 2 — UNDERPAYMENT REPORT (by HCPCS / state / provider / payer)
# ============================================================
def render_report_tab():
    hcpcs_df    = load_csv_safe(str(SUMMARY_HCPCS_PATH))
    state_df    = load_csv_safe(str(SUMMARY_STATE_PATH))
    provider_df = load_csv_safe(str(SUMMARY_PROVIDER_PATH))
    payer_df    = load_csv_safe(str(SUMMARY_PAYER_PATH))
    exec_df     = load_csv_safe(str(EXEC_SUMMARY_PATH))

    if all(d is None for d in [hcpcs_df, state_df, provider_df, payer_df]):
        missing_file_notice(SUMMARY_HCPCS_PATH, SUMMARY_STATE_PATH, SUMMARY_PROVIDER_PATH, SUMMARY_PAYER_PATH)
        return

    if exec_df is not None and "metric" in exec_df.columns:
        exec_map = dict(zip(exec_df["metric"], exec_df["value"]))

        def _num(key, default=0):
            try:
                return float(exec_map.get(key, default))
            except (TypeError, ValueError):
                return default

        kpi_row([
            ("Underpaid rows", f"{int(_num('total_underpaid_rows')):,}"),
            ("Total est. recovery", f"${_num('total_estimated_recovery'):,.0f}"),
            ("Critical tier", f"{int(_num('critical_tier_rows')):,}"),
            ("Top state", str(exec_map.get("top_state_by_recovery", "—"))),
        ])

    c1, c2 = st.columns(2)
    with c1:
        if state_df is not None and "total_estimated_recovery" in state_df.columns:
            top_state = state_df.sort_values("total_estimated_recovery", ascending=False).head(15)
            st.plotly_chart(
                horizontal_bar(top_state.iloc[::-1], "total_estimated_recovery", "provider_state",
                               "Underpayment $ by State (Top 15)"),
                use_container_width=True,
            )
        else:
            st.info("`underpayment_summary_by_state.csv` not found.")
    with c2:
        if hcpcs_df is not None and "total_estimated_recovery" in hcpcs_df.columns:
            top_hcpcs = hcpcs_df.sort_values("total_estimated_recovery", ascending=False).head(15)
            st.plotly_chart(
                horizontal_bar(top_hcpcs.iloc[::-1], "total_estimated_recovery", "HCPCS_Cd",
                               "Underpayment $ by HCPCS Code (Top 15)"),
                use_container_width=True,
            )
        else:
            st.info("`underpayment_summary_by_hcpcs.csv` not found.")

    c3, c4 = st.columns(2)
    with c3:
        if provider_df is not None and "total_estimated_recovery" in provider_df.columns:
            top_prov = provider_df.sort_values("total_estimated_recovery", ascending=False).head(15)
            st.plotly_chart(
                horizontal_bar(top_prov.iloc[::-1], "total_estimated_recovery", "provider_type",
                               "Underpayment $ by Provider Type (Top 15)"),
                use_container_width=True,
            )
        else:
            st.info("`underpayment_summary_by_provider_type.csv` not found.")
    with c4:
        if payer_df is not None and "total_estimated_recovery" in payer_df.columns:
            pay = payer_df.sort_values("total_estimated_recovery", ascending=False)
            st.plotly_chart(
                horizontal_bar(pay.iloc[::-1], "total_estimated_recovery", "payer_type_proxy",
                               "Underpayment $ by Payer Type (Medicare Participation Proxy)"),
                use_container_width=True,
            )
        else:
            st.info("`underpayment_summary_by_payer_type.csv` not found.")

    st.markdown(
        "<div class='disclaimer-bar'>\"Payer type\" reflects Medicare participation status only — "
        "the CMS PUF is Medicare-only, so this is a documented proxy, not a real multi-payer breakdown. "
        "See Step 06's feature definition card for the full limitation.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 3 — CLAIM CHECKER (original Step 13 single-claim form)
# ============================================================
def render_claim_checker_tab():
    st.caption("Enter claim details — the model predicts recovery priority + a recommended action.")

    col_form, col_gap, col_result = st.columns([5, 0.3, 5])

    with col_form:
        with st.form("ar_form"):

            st.markdown('<div class="form-section">Procedure Info</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                proc_cat = st.selectbox("Procedure Category", [
                    "Evaluation_and_Management",
                    "Musculoskeletal_Surgery",
                    "Radiology",
                    "Medicine_Services",
                    "Pathology_Laboratory",
                    "Respiratory_Cardiovascular_Surgery",
                    "Digestive_Surgery",
                    "Urinary_Genital_Surgery",
                    "Nervous_System_Surgery",
                    "Integumentary_Surgery",
                    "Drugs",
                    "HCPCS_G_Codes",
                    "Other",
                ])
                pos = st.selectbox("Place of Service", ["Office (O)", "Facility (F)"])
            with c2:
                payer = st.selectbox("Medicare Participation", [
                    "Medicare_Participating",
                    "Medicare_NonParticipating",
                ])
                review_flag = st.selectbox("Review Flag", ["No (0)", "Yes (1)"])

            st.markdown('<div class="form-section">Utilization</div>', unsafe_allow_html=True)
            u1, u2, u3 = st.columns(3)
            with u1:
                tot_srvcs = st.slider("Total Services", 1, 5000, 50)
            with u2:
                tot_benes = st.slider("Total Beneficiaries", 1, 2000, 30)
            with u3:
                avg_charge = st.slider("Avg Submitted Charge ($)", 0, 5000, 250)

            submitted = st.form_submit_button("💰 Check Recovery Priority", use_container_width=True)

    with col_result:
        if submitted:
            pos_val = "O" if pos.startswith("O") else "F"
            payload = {
                "procedure_category": proc_cat,
                "payer_type_proxy":   payer,
                "place_of_service":   pos_val,
                "tot_srvcs":          float(tot_srvcs),
                "tot_benes":          float(tot_benes),
                "avg_sbmtd_chrg":     float(avg_charge),
                "review_flag":        1 if "1" in review_flag else 0,
            }

            with st.spinner("Analyzing claim..."):
                result = call_api(payload)

            if result:
                prob   = result.get("recovery_probability", 0)
                tier   = result.get("priority_tier", "Low")
                action = result.get("recommended_action", "")
                signal = result.get("estimated_recovery_signal", "")
                factors = result.get("top_risk_factors", [])
                model_name = result.get("model_name", "")

                badge_map = {"Critical": "HIGH", "High": "HIGH", "Medium": "MEDIUM", "Low": "LOW"}
                badge_cls = badge_map.get(tier, "LOW")

                gt = gauge_theme()
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix": "%", "font": {"size": 44, "family": "Plus Jakarta Sans, sans-serif", "color": tier_color(tier)}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": gt["tickcolor"], "tickwidth": 1, "tickfont": {"color": gt["tickfont"]}},
                        "bar":  {"color": tier_color(tier), "thickness": 0.28},
                        "bgcolor": gt["track_bg"],
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,  40],  "color": gt["steps"][0]},
                            {"range": [40, 70],  "color": gt["steps"][1]},
                            {"range": [70, 100], "color": gt["steps"][2]},
                        ],
                        "threshold": {
                            "line": {"color": gt["threshold"], "width": 2},
                            "thickness": 0.8,
                            "value": 60,
                        },
                    },
                    title={"text": "Recovery Priority Score", "font": {"size": 13, "color": gt["title_font"], "family": "Plus Jakarta Sans, sans-serif"}},
                ))
                fig.update_layout(
                    height=240,
                    margin=dict(t=50, b=0, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    f"<div style='text-align:center;margin-top:-8px;margin-bottom:20px'>"
                    f"<span class='badge-{badge_cls}'>{tier.upper()} PRIORITY</span></div>",
                    unsafe_allow_html=True,
                )

                action_icon = {"Critical": "🚨", "High": "⚡", "Medium": "📋", "Low": "👁️"}.get(tier, "📋")
                st.markdown(f"""
                <div class="cause-card">
                    <div style="display:flex; align-items:center; gap:14px; margin-bottom:14px;">
                        <div style="width:38px; height:38px; border-radius:10px; flex-shrink:0;
                            background:rgba(96,165,250,0.12); border:1px solid rgba(96,165,250,0.2);
                            display:flex; align-items:center; justify-content:center; font-size:20px;">
                            {action_icon}
                        </div>
                        <div>
                            <div class="cause-label">{tier} Priority</div>
                            <div class="cause-conf">{signal}</div>
                        </div>
                    </div>
                    <div style="padding-top:14px; border-top:1px solid var(--border-color);">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                            <span style="color:var(--fix-label-text); font-size:14px;">&#10022;</span>
                            <span class="fix-label">Recommended Action</span>
                        </div>
                        <div class="fix-text">{action}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if factors:
                    st.markdown('<div class="card-title">Recovery Signals</div>', unsafe_allow_html=True)
                    for i, factor in enumerate(factors):
                        pct = max(30, 100 - i * 28)
                        bar_cls = "driver-bar-high" if i == 0 else ("driver-bar-high" if i == 1 else "driver-bar-low")
                        st.markdown(f"""
                        <div class="driver-row">
                            <div class="driver-name">&#8679; {factor}</div>
                            <div class="driver-bar-wrap"><div class="{bar_cls}" style="width:{pct}%"></div></div>
                        </div>""", unsafe_allow_html=True)

                st.markdown(
                    f"<div class='disclaimer-bar'>Model: {model_name} &nbsp;&middot;&nbsp; "
                    f"CMS PUF 2023 &nbsp;&middot;&nbsp; Portfolio demo only — not real recoverable amounts</div>",
                    unsafe_allow_html=True,
                )

        else:
            st.markdown("""
<div class="ready-card">
    <div class="ready-icon">💰</div>
    <div class="ready-title">Ready to Analyze</div>
    <div class="ready-subtitle">
        Fill in the claim details and click<br>
        <strong>Check Recovery Priority</strong>
    </div>
    <div class="feature-list">
        <div class="feature-item">
            <div class="feature-icon fi-blue">📊</div>
            <div>
                <div class="feature-title">Recovery Priority Score</div>
                <div class="feature-desc">ML model predicts 0–100% recovery likelihood</div>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon fi-amber">⚡</div>
            <div>
                <div class="feature-title">Priority Tier</div>
                <div class="feature-desc">Critical / High / Medium / Low — ranked by urgency</div>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon fi-green">✅</div>
            <div>
                <div class="feature-title">Recommended Action</div>
                <div class="feature-desc">Tells billing team exactly what to do next</div>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon fi-red">🔍</div>
            <div>
                <div class="feature-title">Recovery Signals</div>
                <div class="feature-desc">Top factors driving the underpayment risk</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── TABS ─────────────────────────────────────────────────────
tab_queue, tab_report, tab_checker = st.tabs([
    "📋 AR Priority Queue", "📊 Underpayment Report", "🔍 Claim Checker",
])

with tab_queue:
    render_ar_queue_tab()

with tab_report:
    render_report_tab()

with tab_checker:
    render_claim_checker_tab()
