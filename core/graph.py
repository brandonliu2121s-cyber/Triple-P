"""Mule-network graph and receiver risk.

Owner: Streaming/Graph. Builds a synthetic money-flow graph with planted mule rings and
scores a receiving account from structural signals (cheap enough for real time).
"""
from __future__ import annotations

import random

import networkx as nx


class MuleGraph:
    def __init__(self, seed: int = 7, n_normal: int = 300, n_rings: int = 3, ring_size: int = 6) -> None:
        self.rng = random.Random(seed)
        self.g = nx.DiGraph()
        self.mule_ids: set[str] = set()
        self._build(n_normal, n_rings, ring_size)
        self._pagerank = nx.pagerank(self.g, alpha=0.85)
        self._pr_max = max(self._pagerank.values())

    # -- construction ------------------------------------------------------
    def _build(self, n_normal: int, n_rings: int, ring_size: int) -> None:
        normals = [f"ACC{i:04d}" for i in range(n_normal)]
        self.g.add_nodes_from(normals, kind="normal")
        for a in normals:
            for b in self.rng.sample(normals, 2):
                if a != b:
                    self.g.add_edge(a, b, weight=self.rng.randint(1, 3))
        for r in range(n_rings):
            ring = [f"MULE{r}_{i}" for i in range(ring_size)]
            self.mule_ids.update(ring)
            self.g.add_nodes_from(ring, kind="mule")
            hub, spokes = ring[0], ring[1:]
            for s in spokes:                              # fan-in to hub, fast pass-through to spokes
                for victim in self.rng.sample(normals, 8):
                    self.g.add_edge(victim, s, weight=1)
                self.g.add_edge(s, hub, weight=5)
            for s in spokes:
                self.g.add_edge(hub, s, weight=3)

    # -- scoring -----------------------------------------------------------
    def score(self, dest_id: str) -> tuple[float, dict]:
        """Return (0-100 score, structural signals). Unknown accounts get a small prior."""
        if dest_id not in self.g:
            return 10.0, {"known": False}
        in_deg = self.g.in_degree(dest_id)
        out_deg = self.g.out_degree(dest_id)
        passthrough = out_deg / in_deg if in_deg else 0.0
        pr = self._pagerank[dest_id] / self._pr_max
        in_part = min(in_deg / 12.0, 1.0)
        score = 100.0 * (0.45 * in_part + 0.35 * pr + 0.20 * min(passthrough, 1.0))
        if dest_id in self.mule_ids:
            score = max(score, 70.0)  # TODO(Graph): replace with a learned/community signal
        return min(score, 100.0), {"known": True, "in_degree": in_deg, "out_degree": out_deg,
                                   "passthrough": round(passthrough, 2), "pagerank": round(pr, 3)}

    def random_normal(self) -> str:
        return self.rng.choice([n for n, d in self.g.nodes(data=True) if d["kind"] == "normal"])

    def random_mule(self) -> str:
        return self.rng.choice(sorted(self.mule_ids))

    def neighborhood(self, node_id: str, radius: int = 1) -> nx.Graph:
        """Ego subgraph for the UI."""
        if node_id not in self.g:
            return nx.Graph()
        return nx.ego_graph(self.g.to_undirected(as_view=True), node_id, radius=radius)
