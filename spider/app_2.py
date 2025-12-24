# app.py
import json
import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network

# =============================================================================
# App Versioning
# =============================================================================
APP_VERSION = "0.2.4"

VERSION_HISTORY = [
    ("0.2.4", "Fix PyVis rendering: ensure net.set_options gets JSON only (no JS). Add empty-graph guard + lightweight debug."),
    ("0.2.3", "Keep layout modes + re-run layout + community legend + export node positions. (Polinode .xlsx supported)."),
    ("0.2.2", "Add visible app version + version history section."),
    ("0.2.1", "Add Louvain community detection and color-by-community."),
    ("0.2.0", "Initial Polinode/CSV upload + column mapping + basic filters and visualization."),
]

st.set_page_config(page_title=f"Mini Network Mapper v{APP_VERSION}", layout="wide")


# =============================================================================
# Helpers
# =============================================================================
def strip_and_drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]
    return df


def read_table(uploaded_file, sheet_name=None) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    name = (uploaded_file.name or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)
    return pd.read_csv(uploaded_file)


def safe_float(x, default=1.0) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """node -> community_id (Louvain if available, else greedy modularity fallback)."""
    H = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    try:
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(H, seed=42)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(H))

    node_to_comm: Dict[str, int] = {}
    for i, comm in enumerate(communities):
        for n in comm:
            node_to_comm[str(n)] = int(i)
    return node_to_comm


def compute_layout_positions(G: nx.Graph, layout_algo: str, seed: int) -> Dict[str, Dict[str, float]]:
    """
    Returns dict[node] -> {"x": float, "y": float}
    Positions are computed in Python and used to freeze nodes in PyVis.
    """
    H = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    if H.number_of_nodes() == 0:
        return {}

    if layout_algo == "spring":
        pos = nx.spring_layout(H, seed=seed, iterations=120)
    elif layout_algo == "kamada_kawai":
        pos = nx.kamada_kawai_layout(H)
    else:
        pos = nx.spring_layout(H, seed=seed, iterations=120)

    scaled: Dict[str, Dict[str, float]] = {}
    for n, (x, y) in pos.items():
        scaled[str(n)] = {"x": float(x) * 800.0, "y": float(y) * 800.0}
    return scaled


def build_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    directed: bool,
    node_id_col: Optional[str],
    edge_source_col: str,
    edge_target_col: str,
    edge_type_col: Optional[str],
    edge_weight_col: Optional[str],
    edge_label_col: Optional[str],
) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()

    # Nodes (optional)
    if not nodes_df.empty and node_id_col:
        for _, row in nodes_df.iterrows():
            node_id = row.get(node_id_col)
            if pd.isna(node_id):
                continue
            node_id = str(node_id)
            attrs = row.to_dict()
            attrs["id"] = node_id
            G.add_node(node_id, **attrs)

    # Edges (required)
    if edge_source_col not in edges_df.columns or edge_target_col not in edges_df.columns:
        raise ValueError("Edges mapping invalid: source/target columns not found in edges table.")

    for _, row in edges_df.iterrows():
        s = row.get(edge_source_col)
        t = row.get(edge_target_col)
        if pd.isna(s) or pd.isna(t):
            continue
        s = str(s)
        t = str(t)

        if not G.has_node(s):
            G.add_node(s, id=s)
        if not G.has_node(t):
            G.add_node(t, id=t)

        attrs = row.to_dict()

        # Normalize optionals to standard keys used by filters/legend.
        if edge_type_col and edge_type_col in edges_df.columns:
            attrs["type"] = "" if pd.isna(row.get(edge_type_col)) else str(row.get(edge_type_col))
        if edge_label_col and edge_label_col in edges_df.columns:
            attrs["label"] = "" if pd.isna(row.get(edge_label_col)) else str(row.get(edge_label_col))
        if edge_weight_col and edge_weight_col in edges_df.columns:
            attrs["weight"] = safe_float(row.get(edge_weight_col), default=1.0)
        else:
            attrs["weight"] = safe_float(row.get("weight", 1.0), default=1.0)

        G.add_edge(s, t, **attrs)

    return G


def pyvis_options_json(physics_enabled: bool) -> str:
    """
    PyVis requires a JSON string here. DO NOT pass JavaScript (e.g., 'var options = ...').
    """
    options: Dict[str, Any] = {
        "nodes": {"borderWidth": 0},
        "edges": {"smooth": {"type": "dynamic"}},
        "interaction": {"hover": True, "tooltipDelay": 100, "hideEdgesOnDrag": True},
        "physics": {
            "enabled": bool(physics_enabled),
            "stabilization": {"enabled": True, "iterations": 200, "updateInterval": 25},
        },
    }
    return json.dumps(options)


def to_pyvis(
    G: nx.Graph,
    height: str,
    physics_enabled: bool,
    node_color_attr: Optional[str],
    node_size_mode: str,
    label_attr: Optional[str],
    edge_label_attr: Optional[str],
    edge_color_attr: Optional[str],
    max_nodes: int,
    positions: Optional[Dict[str, Dict[str, float]]],
) -> Network:
    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="#111111",
        directed=isinstance(G, nx.DiGraph),
    )

    # If we're using fixed positions, we want physics off.
    if positions:
        physics_enabled = False

    net.toggle_physics(physics_enabled)

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G) if node_size_mode == "betweenness" else {}

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
        H = G.subgraph(nodes).copy()
    else:
        H = G

    category_to_color: Dict[str, str] = {}
    color_idx = 0

    # Nodes
    for n, attrs in H.nodes(data=True):
        n_str = str(n)

        if label_attr and label_attr in attrs and pd.notna(attrs.get(label_attr)):
            label = str(attrs.get(label_attr))
        else:
            label = n_str

        color = "#4c78a8"
        if node_color_attr and node_color_attr in attrs and pd.notna(attrs.get(node_color_attr)):
            cat = str(attrs.get(node_color_attr))
            if cat not in category_to_color:
                category_to_color[cat] = palette[color_idx % len(palette)]
                color_idx += 1
            color = category_to_color[cat]

        if node_size_mode == "degree":
            size = 8 + 3 * math.log1p(degree.get(n, 0))
        elif node_size_mode == "betweenness":
            size = 8 + 40 * betweenness.get(n, 0)
        else:
            size = 10

        title_lines = [f"<b>{label}</b>"]
        for k, v in attrs.items():
            if k == "id":
                continue
            title_lines.append(f"{k}: {v}")
        title = "<br/>".join(title_lines)

        if positions and n_str in positions:
            net.add_node(
                n_str,
                label=label,
                color=color,
                size=size,
                title=title,
                x=positions[n_str]["x"],
                y=positions[n_str]["y"],
                physics=False,
            )
        else:
            net.add_node(n_str, label=label, color=color, size=size, title=title)

    # Edges
    for u, v, attrs in H.edges(data=True):
        u_str = str(u)
        v_str = str(v)

        e_label = ""
        if edge_label_attr and edge_label_attr in attrs and pd.notna(attrs.get(edge_label_attr)):
            e_label = str(attrs.get(edge_label_attr))

        e_color = "#999999"
        if edge_color_attr and edge_color_attr in attrs and pd.notna(attrs.get(edge_color_attr)):
            e_cat = str(attrs.get(edge_color_attr))
            if e_cat not in category_to_color:
                category_to_color[e_cat] = palette[color_idx % len(palette)]
                color_idx += 1
            e_color = category_to_color[e_cat]

        w = attrs.get("weight", 1.0)
        try:
            width = 1 + 2 * math.log1p(float(w))
        except Exception:
            width = 2

        net.add_edge(u_str, v_str, label=e_label, color=e_color, width=width, title=str(attrs))

    # CRITICAL: JSON ONLY
    net.set_options(pyvis_options_json(physics_enabled))

    return net


# =============================================================================
# Session State
# =============================================================================
if "layout_seed" not in st.session_state:
    st.session_state.layout_seed = int(time.time()) % 1_000_000


# =============================================================================
# UI
# =============================================================================
st.title("Mini Network Mapper")
st.caption(f"Version: **v{APP_VERSION}**")

with st.expander("Version history", expanded=False):
    for v, note in VERSION_HISTORY:
        st.markdown(f"- **{v}** — {note}")

with st.sidebar:
    st.header("Upload")

    st.markdown("**Option A (recommended):** Polinode-style workbook (`.xlsx`) with sheets `Nodes` and `Edges`.")
    workbook = st.file_uploader("Upload workbook (.xlsx)", type=["xlsx", "xls"], key="workbook")

    st.markdown("---")
    st.markdown("**Option B:** CSVs")
    edges_file = st.file_uploader("Upload edges.csv", type=["csv"], key="edges")
    nodes_file = st.file_uploader("Upload nodes.csv (optional)", type=["csv"], key="nodes")

    st.divider()
    st.header("Display settings")
    directed = st.checkbox("Directed graph", value=False)

    layout_mode = st.radio(
        "Layout mode",
        ["Auto (best)", "Physics (vis.js)", "Python layout (frozen)"],
        index=0,
        help="Auto uses Python frozen layout for big graphs (no drifting), physics for small graphs.",
    )

    physics_checkbox = st.checkbox("Physics layout (may animate)", value=True)
    detect_comm = st.checkbox("Detect communities (Louvain)", value=False)

    height = st.selectbox("Canvas height", ["650px", "750px", "900px"], index=0)
    max_nodes = st.slider("Max nodes to render (performance)", 50, 5000, 1500, step=50)

    st.divider()
    st.header("Styling")
    node_size_mode = st.selectbox("Node size by", ["degree", "betweenness", "fixed"], index=0)
    python_layout_algo = st.selectbox("Python layout algorithm", ["spring", "kamada_kawai"], index=0)

    st.divider()
    st.header("Layout actions")
    if st.button("Re-run layout"):
        st.session_state.layout_seed = (st.session_state.layout_seed + 1) % 1_000_000

    st.divider()
    debug_mode = st.checkbox("Debug mode (show tables + counts)", value=False)

# Load data
if workbook is not None:
    nodes_df_raw = strip_and_drop_unnamed(read_table(workbook, sheet_name="Nodes"))
    edges_df_raw = strip_and_drop_unnamed(read_table(workbook, sheet_name="Edges"))
    source_mode = "workbook"
else:
    if edges_file is None:
        st.info("Upload either a workbook (.xlsx) or an edges.csv to begin.")
        st.stop()
    edges_df_raw = strip_and_drop_unnamed(read_table(edges_file))
    nodes_df_raw = strip_and_drop_unnamed(read_table(nodes_file)) if nodes_file else pd.DataFrame()
    source_mode = "csv"

if debug_mode:
    st.subheader("Debug: Raw tables")
    st.write(f"Source mode: **{source_mode}**")
    st.write("Edges preview:")
    st.dataframe(edges_df_raw.head(25), use_container_width=True)
    if not nodes_df_raw.empty:
        st.write("Nodes preview:")
        st.dataframe(nodes_df_raw.head(25), use_container_width=True)

# Column Mapping
st.subheader("Column Mapping")
st.caption("Pick which columns to use (supports Polinode exports and most other formats).")

mcol1, mcol2 = st.columns([0.5, 0.5], gap="large")

with mcol1:
    st.markdown("### Edges (required)")
    e_cols = list(edges_df_raw.columns)
    if not e_cols:
        st.error("Edges table has no columns. Check your file.")
        st.stop()

    def pick_default(cols: List[str], candidates: List[str]) -> str:
        for c in candidates:
            if c in cols:
                return c
        return cols[0]

    default_source = pick_default(e_cols, ["Source", "source", "From", "from", "src", "Src"])
    default_target = pick_default(e_cols, ["Target", "target", "To", "to", "dst", "Dst"])
    default_type = pick_default(e_cols, ["Type", "type", "Edge Type", "edge_type", "relationship", "Relationship"])
    default_weight = pick_default(e_cols, ["Amount", "amount", "Weight", "weight", "Value", "value", "count", "Count"])
    default_label = pick_default(e_cols, ["label", "Label", "Purpose", "purpose", "Description", "description"])

    edge_source_col = st.selectbox("Edge source column", e_cols, index=e_cols.index(default_source))
    edge_target_col = st.selectbox("Edge target column", e_cols, index=e_cols.index(default_target))

    edge_type_col_sel = st.selectbox(
        "Edge type column (optional)",
        ["(none)"] + e_cols,
        index=(["(none)"] + e_cols).index(default_type) if default_type in e_cols else 0,
    )
    edge_type_col = None if edge_type_col_sel == "(none)" else edge_type_col_sel

    edge_weight_col_sel = st.selectbox(
        "Edge weight column (optional)",
        ["(none)"] + e_cols,
        index=(["(none)"] + e_cols).index(default_weight) if default_weight in e_cols else 0,
    )
    edge_weight_col = None if edge_weight_col_sel == "(none)" else edge_weight_col_sel

    edge_label_col_sel = st.selectbox(
        "Edge label column (optional)",
        ["(none)"] + e_cols,
        index=(["(none)"] + e_cols).index(default_label) if default_label in e_cols else 0,
    )
    edge_label_col = None if edge_label_col_sel == "(none)" else edge_label_col_sel

with mcol2:
    st.markdown("### Nodes (optional)")
    if nodes_df_raw.empty:
        st.info("No nodes table found/uploaded — nodes will be created from edge endpoints.")
        node_id_col = None
    else:
        n_cols = list(nodes_df_raw.columns)
        default_node_id = "Name" if "Name" in n_cols else ("id" if "id" in n_cols else n_cols[0])
        node_id_col = st.selectbox("Which column is the node id?", n_cols, index=n_cols.index(default_node_id))

# Build graph
try:
    G = build_graph(
        nodes_df=nodes_df_raw,
        edges_df=edges_df_raw,
        directed=directed,
        node_id_col=node_id_col,
        edge_source_col=edge_source_col,
        edge_target_col=edge_target_col,
        edge_type_col=edge_type_col,
        edge_weight_col=edge_weight_col,
        edge_label_col=edge_label_col,
    )
except Exception as e:
    st.error(f"Could not build graph: {e}")
    st.stop()

# Communities
if detect_comm and G.number_of_nodes() > 0 and G.number_of_edges() > 0:
    node_to_comm = detect_communities(G)
    nx.set_node_attributes(G, node_to_comm, "_community")

# Collect attrs
all_node_attrs = sorted({k for _, a in G.nodes(data=True) for k in a.keys()})
all_edge_attrs = sorted({k for _, _, a in G.edges(data=True) for k in a.keys()})

# Main columns
col1, col2 = st.columns([0.35, 0.65], gap="large")

with col1:
    st.subheader("Filters")

    color_options = ["(none)"] + [("(community)" if a == "_community" else a) for a in all_node_attrs]
    node_color_choice = st.selectbox("Color nodes by attribute", color_options, index=0)
    if node_color_choice == "(none)":
        node_color_attr = None
    elif node_color_choice == "(community)":
        node_color_attr = "_community"
    else:
        node_color_attr = node_color_choice

    label_attr_sel = st.selectbox("Node label attribute", ["(id)"] + all_node_attrs, index=0)
    label_attr = None if label_attr_sel == "(id)" else label_attr_sel

    edge_label_attr_sel = st.selectbox("Edge label attribute", ["(none)"] + all_edge_attrs, index=0)
    edge_label_attr = None if edge_label_attr_sel == "(none)" else edge_label_attr_sel

    edge_color_attr_sel = st.selectbox("Color edges by attribute", ["(none)"] + all_edge_attrs, index=0)
    edge_color_attr = None if edge_color_attr_sel == "(none)" else edge_color_attr_sel

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 0
    min_deg = st.slider("Minimum degree", 0, max_deg, 0)

    # Edge type filter (based on normalized attrs["type"])
    type_values = sorted({str(a.get("type")) for _, _, a in G.edges(data=True) if a.get("type") not in [None, ""]})
    if type_values:
        selected_types = st.multiselect("Edge type filter", type_values, default=type_values)
    else:
        selected_types = []

    search = st.text_input("Search node (by id or label text)", value="").strip().lower()

    st.divider()
    st.subheader("Stats")
    st.write(f"**Source mode:** {source_mode}")
    st.write(f"**Nodes:** {G.number_of_nodes():,}")
    st.write(f"**Edges:** {G.number_of_edges():,}")
    if G.number_of_nodes() > 0:
        st.write(f"**Density:** {nx.density(G):.4f}")

    # Community legend
    if "_community" in all_node_attrs:
        st.divider()
        st.subheader("Community legend")
        comm_vals = [a.get("_community") for _, a in G.nodes(data=True) if a.get("_community") is not None]
        if comm_vals:
            legend = (
                pd.Series(comm_vals, name="community")
                .value_counts()
                .rename_axis("community_id")
                .reset_index(name="node_count")
                .sort_values("node_count", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(legend, use_container_width=True, height=240)

            st.download_button(
                "Download community_legend.csv",
                data=legend.to_csv(index=False).encode("utf-8"),
                file_name="community_legend.csv",
                mime="text/csv",
            )

with col2:
    st.subheader("Network")

    # Filter nodes
    keep_nodes = set()
    degrees = dict(G.degree())
    for n, attrs in G.nodes(data=True):
        if degrees.get(n, 0) < min_deg:
            continue
        if search:
            hay = str(n).lower()
            if label_attr and label_attr in attrs and pd.notna(attrs.get(label_attr)):
                hay += " " + str(attrs.get(label_attr)).lower()
            if search not in hay:
                continue
        keep_nodes.add(str(n))

    H = G.subgraph(keep_nodes).copy()

    # Filter edges by type
    if selected_types:
        remove_edges = []
        for u, v, attrs in H.edges(data=True):
            t = attrs.get("type", None)
            if t in [None, ""]:
                continue
            if str(t) not in selected_types:
                remove_edges.append((u, v))
        H.remove_edges_from(remove_edges)

        # Remove isolates created by edge filtering
        isolates = [n for n in H.nodes() if H.degree(n) == 0]
        H.remove_nodes_from(isolates)

    # Guard: if filters remove everything, show why.
    if H.number_of_nodes() == 0 or H.number_of_edges() == 0:
        st.warning(
            "No graph to render (your current filters likely removed everything). "
            "Try setting **Minimum degree** back to 0, clearing **Edge type filter**, and clearing **Search**."
        )
        if debug_mode:
            st.write(
                {
                    "G_nodes": G.number_of_nodes(),
                    "G_edges": G.number_of_edges(),
                    "H_nodes": H.number_of_nodes(),
                    "H_edges": H.number_of_edges(),
                    "min_degree": min_deg,
                    "selected_types_count": len(selected_types),
                    "search": search,
                }
            )
        st.stop()

    # Layout behavior + physics fix
    N = H.number_of_nodes()
    positions = None

    if layout_mode == "Physics (vis.js)":
        physics_enabled = bool(physics_checkbox)
        if N > 250 and physics_enabled:
            st.warning("Physics on a large graph may drift. Consider **Python layout (frozen)** or Auto.")
    elif layout_mode == "Python layout (frozen)":
        physics_enabled = False
        positions = compute_layout_positions(H, python_layout_algo, seed=st.session_state.layout_seed)
    else:  # Auto (best)
        if N > 250:
            physics_enabled = False
            positions = compute_layout_positions(H, python_layout_algo, seed=st.session_state.layout_seed)
        else:
            physics_enabled = bool(physics_checkbox)

    net = to_pyvis(
        H,
        height=height,
        physics_enabled=physics_enabled,
        node_color_attr=node_color_attr,
        node_size_mode=node_size_mode,
        label_attr=label_attr,
        edge_label_attr=edge_label_attr,
        edge_color_attr=edge_color_attr,
        max_nodes=max_nodes,
        positions=positions,
    )

    st.components.v1.html(net.generate_html(), height=int(height.replace("px", "")) + 30, scrolling=True)

    with st.expander("Downloads"):
        out_nodes = pd.DataFrame([{"id": str(n), **a} for n, a in H.nodes(data=True)])
        out_edges = pd.DataFrame([{"source": str(u), "target": str(v), **a} for u, v, a in H.edges(data=True)])

        st.download_button(
            "Download nodes_filtered.csv",
            data=out_nodes.to_csv(index=False).encode("utf-8"),
            file_name="nodes_filtered.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download edges_filtered.csv",
            data=out_edges.to_csv(index=False).encode("utf-8"),
            file_name="edges_filtered.csv",
            mime="text/csv",
        )

        # Positions export (only reliable in Python layout mode / Auto when it chooses Python)
        if positions:
            pos_df = pd.DataFrame([{"id": n, "x": positions[n]["x"], "y": positions[n]["y"]} for n in positions.keys()])
            st.download_button(
                "Download node_positions.csv",
                data=pos_df.to_csv(index=False).encode("utf-8"),
                file_name="node_positions.csv",
                mime="text/csv",
            )
        else:
            st.info("Node positions export is available when using **Python layout (frozen)** (or Auto on larger graphs).")
