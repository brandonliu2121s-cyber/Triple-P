"""GuardianZ entry point and page router. Run: streamlit run app.py   (Owner: UI/Integration)"""
import streamlit as st

from ui import theme

st.set_page_config(page_title="GuardianZ", page_icon="🛡️", layout="wide")
theme.apply()

st.navigation([
    st.Page("views/live.py", title="Live monitor", icon="📡", default=True),
    st.Page("views/analysis.py", title="Analysis", icon="📊"),
]).run()
