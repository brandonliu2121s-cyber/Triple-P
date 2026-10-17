"""Dashboard styling and motion. Owner: UI/Integration.

Motion rules (the stream refreshes every second, so almost nothing animates):
- press feedback on buttons and tabs, 160 ms ease-out
- newly flagged cards (alerts, case detail) fade in once, 200 ms ease-out
- no animation on the live feed table or metrics, which update continuously
- everything motion-based is removed under prefers-reduced-motion
"""
import streamlit as st

CSS = """
<style>
:root {
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
  --card-border: rgba(128, 128, 128, 0.22);
  --card-bg: rgba(128, 128, 128, 0.06);
  --tier-hold: #f5a524;
  --tier-escalate: #e5484d;
}

/* Tighter page, clearer hierarchy */
.block-container { padding-top: 2rem; max-width: 1400px; }
h1 { letter-spacing: -0.02em; }

/* Long ids, emails and URLs have no spaces; let them wrap instead of pushing the layout wide */
[data-testid="stMarkdownContainer"], [data-testid="stAlert"] { overflow-wrap: anywhere; }
[data-testid="stMarkdownContainer"] code { white-space: pre-wrap; }

/* Metric cards: soft border instead of a heavy fill */
[data-testid="stMetric"] {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 14px 16px;
}
[data-testid="stMetricLabel"] { opacity: 0.7; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }  /* no jitter as numbers tick */

/* Press feedback */
.stButton > button,
[data-testid="stBaseButton-secondary"] {
  transition: transform 160ms var(--ease-out), border-color 160ms ease, background-color 160ms ease;
}
.stButton > button:active { transform: scale(0.97); }

/* Tabs: colour only, 160 ms */
[data-testid="stTabs"] button[role="tab"] { transition: color 160ms ease; }

/* Alerts and info boxes: rounded, and fade in when they first mount */
[data-testid="stAlert"] {
  border-radius: 12px;
  border: 1px solid var(--card-border);
  transition: opacity 200ms var(--ease-out), transform 200ms var(--ease-out);
  @starting-style { opacity: 0; transform: translateY(6px); }
}

/* Dataframe sits in a card */
[data-testid="stDataFrame"] {
  border: 1px solid var(--card-border);
  border-radius: 12px;
  overflow: hidden;
}

/* Sidebar spacing */
[data-testid="stSidebar"] .stButton { margin-bottom: 2px; }

@media (prefers-reduced-motion: reduce) {
  .stButton > button, [data-testid="stAlert"] { transition: opacity 160ms ease; transform: none !important; }
}
</style>
"""


def apply() -> None:
    """Inject once, outside the 1 s fragment, so the stylesheet is not re-sent every refresh."""
    st.markdown(CSS, unsafe_allow_html=True)
