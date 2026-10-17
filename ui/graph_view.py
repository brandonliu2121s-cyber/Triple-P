"""Mule graph view via st.graphviz_chart (rendered client-side, no extra deps). Owner: Streaming/Graph."""
from __future__ import annotations

import streamlit as st

from core.graph import MuleGraph
from core.textutil import md_escape


def _dot(s: str) -> str:
    """Escape an id for a quoted Graphviz string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def render_neighborhood(graph: MuleGraph, node_id: str, radius: int = 1) -> None:
    sub = graph.neighborhood(node_id, radius)
    if not sub.nodes:
        st.write(f"{md_escape(node_id)} is not in the account graph.")   # plain text wraps; a code span would not
        return
    lines = ["digraph G { rankdir=LR; node [shape=circle, style=filled, fontsize=9];"]
    for n in sub.nodes:
        color = "#e5484d" if n in graph.mule_ids else "#8da3b8"
        pen = ", penwidth=3" if n == node_id else ""
        lines.append(f'"{_dot(n)}" [fillcolor="{color}"{pen}];')
    for a, b in graph.g.subgraph(sub.nodes).edges:
        lines.append(f'"{_dot(a)}" -> "{_dot(b)}";')
    lines.append("}")
    st.graphviz_chart("\n".join(lines), width="stretch")
    score, signals = graph.score(node_id)
    st.caption(f"Network risk {score:.0f}/100 · {signals}")
