# app.py
import time
import math
from typing import Optional, Dict, Any, List

import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go

# =============================================================================
# App Versioning
# =============================================================================
APP_VERSION = "0.3.0"

VERSION_HISTORY = [
    ("0.3.0", "Switch renderer to Plotly (more reliable than PyVis in Streamlit). Keep Polinode xlsx/CSV, mapping UI, Louvain communities, legend, rerun layout, export positions."),
    ("0.2.4", "Attempt PyVis JSON-only set_options + empty-graph guard + versioning."),
    ("0.2.3", "Layout modes, rerun layout, legend, export positions."),
    ("0.2.2", "Add visible app version + version history."),
    ("0.2.1", "Add Louvain community detection."),
    ("0.2.0", "Initial upload + mapping + basic filters."),
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
        comms = louvain_communities(H, seed=42)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(H))

    node_to_comm: Dict[str, int] = {}
    for i, cset in enumerate(comms):
        for n in cset:
            node_to_comm[str(n)] = int(i)
    return node_to_comm


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

        # Normalize to standard keys
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


def compute_positions(G: nx.Graph, algo: str, seed: int) -> Dict[str, Dict[str, float]]:
    H = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    if H.number_of_nodes() == 0:
        return {}

    if algo == "spring":
        pos = nx.spring_layout(H, seed=seed, iterations=150)
    elif algo == "kamada_kawai":
        pos = nx.kamada_kawai_layout(H)
    else:
        pos = nx.spring_layout(H, seed=seed, iterations=150)

    out: Dict[str, Dict[str, float]] = {}
    for n, (x, y) in pos.items():
        out[str(n)] = {"x": float(x), "y": float(y)}
    return out


def pick_default(cols: List[str], candidates: List[str]) -> str:
    for c in candidates:
        if c in cols:
            return c
    return cols[0] if cols else ""


def compute_node_sizes(G: nx.Graph, mode: str) -> Dict[str, float]:
    if mode == "betweenness":
        b = nx.betweenness_centrality(G)
        return {str(k): 8 + 40 * float(v) for k, v in b.items()}
    if mode == "degree":
        deg = dict(G.degree())
        return {str(k): 8 + 3 * math.log1p(float(v)) for k, v in deg.items()}
    return {str(k): 10.0 for k in G.nodes()}


def build_plotly_figure(
    H: nx.Graph,
    positions: Dict[str, Dict[str, float]],
    node_color_attr: Optional[str],
    label_attr: Optional[str],
    node_sizes: Dict[str, float],
):
    # Edges as line segments
    edge_x = []
    edge_y = []
    for u, v in H.edges():
        u = str(u); v = str(v)
        if u not in positions or v not in positions:
            continue
        edge_x += [positions[u]["x"], positions[v]["x"], None]
        edge_y += [positions[u]["y"], positions[v]["y"], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
        name="edges",
    )

    # Nodes
    node_x, node_y, node_text, node_ids = [], [], [], []
    color_vals = []
    for n, attrs in H.nodes(data=True):
        n = str(n)
        if n not in positions:
            continue
        node_x.append(positions[n]["x"])
        node_y.append(positions[n]["y"])
        node_ids.append(n)

        # label
        if label_attr and label_attr in attrs and pd.notna(attrs.get(label_attr)):
            label = str(attrs.get(label_attr))
        else:
            label = n

        # hover text
        lines = [f"<b>{label}</b>"]
        for k, v in attrs.items():
            if k == "id":
                continue
            lines.append(f"{k}: {v}")
        node_text.append("<br>".join(lines))

        # color attribute
        if node_color_attr and node_color_attr in attrs and pd.notna(attrs.get(node_color_attr)):
            color_vals.append(str(attrs.get(node_color_attr)))
        else:
            color_vals.append("")

    # Make colors categorical without hardcoding a palette:
    # Plotly will assign colors across categories automatically in separate traces.
    # We’ll split nodes by category into traces (better legend, better consistency).
    df = pd.DataFrame({
        "id": node_ids,
        "x": node_x,
        "y": node_y,
        "hover": node_text,
        "cat": color_vals,
        "size": [node_sizes.get(i, 10.0) for i in node_ids],
    })

    traces = [edge_trace]

    if node_color_attr:
        for cat, g in df.groupby("cat", dropna=False):
            name = str(cat) if cat != "" else "(none)"
            traces.append(
                go.Scatter(
                    x=g["x"], y=g["y"],
                    mode="markers",
                    marker=dict(size=g["size"]),
                    text=g["hover"],
                    hoverinfo="text",
                    name=name,
                )
            )
    else:
        traces.append(
            go.Scatter(
                x=df["x"], y=df["y"],
                mode="markers",
                marker=dict(size=df["size"]),
                text=df["hover"],
                hoverinfo="text",
                name="nodes",
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        dragmode="pan",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


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

    st.markdown("**Option A (recommended):** Polinode workbook (`.xlsx`) with sheets `Nodes` and `Edges`.")
    workbook = st.file_uploader("Upload workbook (.xlsx)", type=["xlsx", "xls"], key="workbook")

    st.markdown("---")
    st.markdown("**Option B:** CSVs")
    edges_file = st.file_uploader("Upload edges.csv", type=["csv"], key="edges")
    nodes_file = st.file_uploader("Upload nodes.csv (optional)", type=["csv"], key="nodes")

    st.divider()
    st.header("Graph settings")
    directed = st.checkbox("Directed graph", value=False)
    detect_comm = st.checkbox("Detect communities (Louvain)", value=False)

    layout_algo = st.selectbox("Layout algorithm", ["spring", "kamada_kawai"], index=0)

    if st.button("Re-run layout"):
        st.session_state.layout_seed = (st.session_state.layout_seed + 1) % 1_000_000

    st.divider()
    st.header("Performance")
    max_nodes = st.slider("Max nodes to render", 50, 5000, 1500, step=50)
    debug_mode = st.checkbox("Debug mode", value=False)

    st.divider()
    st.header("Styling")
    node_size_mode = st.selectbox("Node size by", ["degree", "betweenness", "fixed"], index=0)

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

if edges_df_raw.empty:
    st.error("Edges table is empty.")
    st.stop()

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
st.caption("Pick which columns to use (works with Polinode exports and most other formats).")

mcol1, mcol2 = st.columns([0.5, 0.5], gap="large")

with mcol1:
    st.markdown("### Edges (required)")
    e_cols = list(edges_df_raw.columns)

    default_source = pick_default(e_cols, ["Source", "source", "From", "from", "src", "Src"])
    default_target = pick_default(e_cols, ["Target", "target", "To", "to", "dst", "Dst"])
    default_type = pick_default(e_cols, ["Type", "type", "Edge Type", "edge_type", "relationship", "Relationship"])
    default_weight = pick_default(e_cols, ["Amount", "amount", "Weight", "weight", "Value", "value"])
    default_label = pick_default(e_cols, ["label", "Label", "Purpose", "purpose", "Description", "description"])

    edge_source_col = st.selectbox("Edge source column", e_cols, index=e_cols.index(default_source))
    edge_target_col = st.selectbox("Edge target column", e_cols, index=e_cols.index(default_target))

    edge_type_col_sel = st.selectbox("Edge type column (optional)", ["(none)"] + e_cols,
                                     index=(["(none)"] + e_cols).index(default_type) if default_type in e_cols else 0)
    edge_type_col = None if edge_type_col_sel == "(none)" else edge_type_col_sel

    edge_weight_col_sel = st.selectbox("Edge weight column (optional)", ["(none)"] + e_cols,
                                       index=(["(none)"] + e_cols).index(default_weight) if default_weight in e_cols else 0)
    edge_weight_col = None if edge_weight_col_sel == "(none)" else edge_weight_col_sel

    edge_label_col_sel = st.selectbox("Edge label column (optional)", ["(none)"] + e_cols,
                                      index=(["(none)"] + e_cols).index(default_label) if default_label in e_cols else 0)
    edge_label_col = None if edge_label_col_sel == "(none)" else edge_label_col_sel

with mcol2:
    st.markdown("### Nodes (optional)")
    if nodes_df_raw.empty:
        st.info("No nodes table found/uploaded — nodes will be created from edge endpoints.")
        node_id_col = None
    else:
        n_cols = list(nodes_df_raw.columns)
        default_node_id = "Name" if "Name" in n_cols else (n_cols[0] if n_cols else None)
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

# Add communities
if detect_comm and G.number_of_nodes() > 0 and G.number_of_edges() > 0:
    node_to_comm = detect_communities(G)
    nx.set_node_attributes(G, node_to_comm, "_community")

all_node_attrs = sorted({k for _, a in G.nodes(data=True) for k in a.keys()})
all_edge_attrs = sorted({k for _, _, a in G.edges(data=True) for k in a.keys()})

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

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 0
    min_deg = st.slider("Minimum degree", 0, max_deg, 0)

    type_values = sorted({str(a.get("type")) for _, _, a in G.edges(data=True) if a.get("type") not in [None, ""]})
    if type_values:
        selected_types = st.multiselect("Edge type filter", type_values, default=type_values)
    else:
        selected_types = []

    search = st.text_input("Search node (by id/label)", value="").strip().lower()

    st.divider()
    st.subheader("Stats")
    st.write(f"**Nodes:** {G.number_of_nodes():,}")
    st.write(f"**Edges:** {G.number_of_edges():,}")
    if G.number_of_nodes() > 0:
        st.write(f"**Density:** {nx.density(G):.4f}")

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
    deg = dict(G.degree())
    for n, attrs in G.nodes(data=True):
        n = str(n)
        if deg.get(n, 0) < min_deg:
            continue
        if search:
            hay = n.lower()
            if label_attr and label_attr in attrs and pd.notna(attrs.get(label_attr)):
                hay += " " + str(attrs.get(label_attr)).lower()
            if search not in hay:
                continue
        keep_nodes.add(n)

    H = G.subgraph(keep_nodes).copy()

    # Edge type filtering
    if selected_types:
        remove_edges = []
        for u, v, attrs in H.edges(data=True):
            t = attrs.get("type", None)
            if t in [None, ""]:
                continue
            if str(t) not in selected_types:
                remove_edges.append((u, v))
        H.remove_edges_from(remove_edges)

        isolates = [n for n in H.nodes() if H.degree(n) == 0]
        H.remove_nodes_from(isolates)

    # Performance: node cap
    if H.number_of_nodes() > max_nodes:
        keep = list(H.nodes())[:max_nodes]
        H = H.subgraph(keep).copy()

    if H.number_of_nodes() == 0 or H.number_of_edges() == 0:
        st.warning("No graph to render after filters. Set Minimum degree to 0, clear Edge type filter, and clear Search.")
        if debug_mode:
            st.write({"H_nodes": H.number_of_nodes(), "H_edges": H.number_of_edges(), "min_deg": min_deg, "search": search})
        st.stop()

    # Layout + positions (always compute so we can export)
    positions = compute_positions(H, layout_algo, seed=st.session_state.layout_seed)

    node_sizes = compute_node_sizes(H, node_size_mode)

    fig = build_plotly_figure(
        H=H,
        positions=positions,
        node_color_attr=node_color_attr,
        label_attr=label_attr,
        node_sizes=node_sizes,
    )
    st.plotly_chart(fig, use_container_width=True)

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

        pos_df = pd.DataFrame([{"id": n, "x": positions[n]["x"], "y": positions[n]["y"]} for n in positions.keys()])
        st.download_button(
            "Download node_positions.csv",
            data=pos_df.to_csv(index=False).encode("utf-8"),
            file_name="node_positions.csv",
            mime="text/csv",
        )
