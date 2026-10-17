"""Mule graph view via st.graphviz_chart (rendered client-side, no extra deps). Owner: Streaming/Graph."""
from __future__ import annotations

import streamlit as st

from core.graph import MuleGraph


def render_neighborhood(graph: MuleGraph, node_id: str, radius: int = 1) -> None:
    sub = graph.neighborhood(node_id, radius)
    if not sub.nodes:
        st.write(f"`{node_id}` not in the account graph.")
        return
    lines = ['digraph G { rankdir=TB; pad=0.2; ranksep=0.9; nodesep=0.2;',
             'node [shape=circle, style=filled, fixedsize=true, width=0.85, fontsize=10, fontcolor=white];',
             'edge [arrowsize=0.6, color="#6b7785"];']
    for n in sub.nodes:
        color = "#e5484d" if n in graph.mule_ids else "#5b7a99"
        pen = ", penwidth=3, color=black" if n == node_id else ""
        lines.append(f'"{n}" [fillcolor="{color}"{pen}];')
    for a, b in graph.g.subgraph(sub.nodes).edges:
        lines.append(f'"{a}" -> "{b}";')
    lines.append("}")
    st.graphviz_chart("\n".join(lines), width="content", height=380)
    st.caption("Red = known mule-ring accounts · bold outline = receiving account of the latest held transfer")
    score, signals = graph.score(node_id)
    st.caption(f"Network risk {score:.0f}/100 · {signals}")
