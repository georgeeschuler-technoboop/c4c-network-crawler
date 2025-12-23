import math
import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network

st.set_page_config(page_title="Mini Network Mapper", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def read_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    return pd.read_csv(uploaded_file)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, directed: bool) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes (optional)
    if not nodes_df.empty:
        if "id" not in nodes_df.columns:
            raise ValueError("nodes.csv must contain an 'id' column.")
        for _, row in nodes_df.iterrows():
            node_id = row["id"]
            attrs = row.to_dict()
            G.add_node(node_id, **attrs)

    # Add edges (required)
    required = {"source", "target"}
    if not required.issubset(set(edges_df.columns)):
        raise ValueError("edges.csv must contain columns: source, target (plus optional weight/type/label).")

    for _, row in edges_df.iterrows():
        s = row["source"]
        t = row["target"]
        attrs = row.to_dict()

        # Ensure nodes exist even if nodes.csv not provided
        if not G.has_node(s):
            G.add_node(s, id=s)
        if not G.has_node(t):
            G.add_node(t, id=t)

        # Optional weight
        w = row["weight"] if "weight" in row and not pd.isna(row["weight"]) else 1.0
        attrs["weight"] = float(w)

        G.add_edge(s, t, **attrs)

    return G

def detect_communities(G: nx.Graph) -> dict:
    """
    Returns dict[node] -> community_id (int)
    Uses Louvain if available in NetworkX; otherwise greedy modularity fallback.
    """
    H = G.to_undirected() if isinstance(G, nx.DiGraph) else G

    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(H, seed=42)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(H))

    node_to_comm = {}
    for i, comm in enumerate(communities):
        for n in comm:
            node_to_comm[n] = i
    return node_to_comm

def to_pyvis(
    G: nx.Graph,
    height: str,
    physics: bool,
    node_color_attr: str | None,
    node_size_mode: str,
    label_attr: str | None,
    edge_label_attr: str | None,
    edge_color_attr: str | None,
    max_nodes: int,
):
    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="#111111",
        directed=isinstance(G, nx.DiGraph),
    )
    net.toggle_physics(physics)

    # Metrics for sizing
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G) if node_size_mode == "betweenness" else {}

    # Simple categorical palette (cycled)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Optionally downsample for large graphs
    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
        H = G.subgraph(nodes).copy()
    else:
        H = G

    # Map categories to colors
    category_to_color = {}
    color_idx = 0

    for n, attrs in H.nodes(data=True):
        # Label
        if label_attr and label_attr in attrs and pd.notna(attrs.get(label_attr)):
            label = str(attrs.get(label_attr))
        else:
            label = str(n)

        # Node color
        color = "#4c78a8"
        if node_color_attr and node_color_attr in attrs and pd.notna(attrs.get(node_color_attr)):
            cat = str(attrs.get(node_color_attr))
            if cat not in category_to_color:
                category_to_color[cat] = palette[color_idx % len(palette)]
                color_idx += 1
            color = category_to_color[cat]

        # Size
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

        net.add_node(n, label=label, color=color, size=size, title=title)

    for u, v, attrs in H.edges(data=True):
        # Edge label
        e_label = ""
        if edge_label_attr and edge_label_attr in attrs and pd.notna(attrs.get(edge_label_attr)):
            e_label = str(attrs.get(edge_label_attr))

        # Edge color (optional categorical)
        e_color = "#999999"
        if edge_color_attr and edge_color_attr in attrs and pd.notna(attrs.get(edge_color_attr)):
            e_cat = str(attrs.get(edge_color_attr))
            if e_cat not in category_to_color:
                category_to_color[e_cat] = palette[color_idx % len(palette)]
                color_idx += 1
            e_color = category_to_color[e_cat]

        # Edge width by weight
        w = attrs.get("weight", 1.0)
        try:
            width = 1 + 2 * math.log1p(float(w))
        except Exception:
            width = 2

        net.add_edge(u, v, label=e_label, color=e_color, width=width, title=str(attrs))

    net.set_options("""
    var options = {
      "nodes": { "borderWidth": 0 },
      "edges": { "smooth": { "type": "dynamic" } },
      "interaction": { "hover": true, "tooltipDelay": 100, "hideEdgesOnDrag": true },
      "physics": {
        "enabled": true,
        "stabilization": { "iterations": 200 }
      }
    }
    """)

    return net

# ----------------------------
# UI
# ----------------------------
st.title("Mini Network Mapper (Polinode/Gephi-lite)")

with st.sidebar:
    st.header("1) Upload data")

    st.markdown("**edges.csv required** with columns: `source`, `target` (optional: `weight`, `type`, `label`).")
    edges_file = st.file_uploader("Upload edges.csv", type=["csv"], key="edges")

    st.markdown("**nodes.csv optional** with column: `id` (optional: `label`, `group`, etc.).")
    nodes_file = st.file_uploader("Upload nodes.csv (optional)", type=["csv"], key="nodes")

    st.divider()
    st.header("2) Display settings")
    directed = st.checkbox("Directed graph", value=False)
    physics = st.checkbox("Physics layout", value=True)
    detect_comm = st.checkbox("Detect communities (Louvain)", value=False)
    height = st.selectbox("Canvas height", ["650px", "750px", "900px"], index=0)
    max_nodes = st.slider("Max nodes to render (for performance)", 50, 5000, 1500, step=50)

    st.divider()
    st.header("3) Styling")
    node_size_mode = st.selectbox("Node size by", ["degree", "betweenness", "fixed"], index=0)

# Load data
edges_df = normalize_columns(read_csv(edges_file))
nodes_df = normalize_columns(read_csv(nodes_file)) if nodes_file else pd.DataFrame()

if edges_file is None:
    st.info("Upload an edges.csv to begin.")
    st.stop()

# Build graph
try:
    G = build_graph(nodes_df, edges_df, directed=directed)
except Exception as e:
    st.error(f"Could not build graph: {e}")
    st.stop()

# Community detection (adds node attribute: _community)
if detect_comm and G.number_of_nodes() > 0 and G.number_of_edges() > 0:
    node_to_comm = detect_communities(G)
    nx.set_node_attributes(G, node_to_comm, "_community")

# Gather attrs
all_node_attrs = set()
for _, attrs in G.nodes(data=True):
    all_node_attrs.update(attrs.keys())
all_node_attrs = sorted(all_node_attrs)

all_edge_attrs = set()
for _, _, attrs in G.edges(data=True):
    all_edge_attrs.update(attrs.keys())
all_edge_attrs = sorted(all_edge_attrs)

col1, col2 = st.columns([0.35, 0.65], gap="large")

with col1:
    st.subheader("Filters")
    st.caption("Lightweight filters so you can explore without re-exporting data.")

    # Color nodes by attribute (including computed community)
    color_options = ["(none)"] + [("(community)" if a == "_community" else a) for a in all_node_attrs]
    node_color_choice = st.selectbox("Color nodes by attribute", color_options, index=0)

    if node_color_choice == "(none)":
        node_color_attr = None
    elif node_color_choice == "(community)":
        node_color_attr = "_community"
    else:
        node_color_attr = node_color_choice

    label_attr = st.selectbox("Node label attribute", ["(id)"] + all_node_attrs, index=0)
    label_attr = None if label_attr == "(id)" else label_attr

    edge_label_attr = st.selectbox("Edge label attribute", ["(none)"] + all_edge_attrs, index=0)
    edge_label_attr = None if edge_label_attr == "(none)" else edge_label_attr

    edge_color_attr = st.selectbox("Color edges by attribute", ["(none)"] + all_edge_attrs, index=0)
    edge_color_attr = None if edge_color_attr == "(none)" else edge_color_attr

    # Filter by minimum degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 0
    min_deg = st.slider("Minimum degree", 0, max_deg, 0)

    # Optional edge-type filter if present
    edge_type_values = []
    if "type" in edges_df.columns:
        edge_type_values = sorted([str(x) for x in edges_df["type"].dropna().unique()])
    selected_types = st.multiselect("Edge type filter (if `type` exists)", edge_type_values, default=edge_type_values)

    search = st.text_input("Search node (by id or label text)", value="").strip().lower()

    st.divider()
    st.subheader("Stats")
    st.write(f"**Nodes:** {G.number_of_nodes():,}")
    st.write(f"**Edges:** {G.number_of_edges():,}")
    if G.number_of_nodes() > 0:
        st.write(f"**Density:** {nx.density(G):.4f}")

    if detect_comm and "_community" in all_node_attrs:
        comm_vals = [attrs.get("_community") for _, attrs in G.nodes(data=True) if attrs.get("_community") is not None]
        if comm_vals:
            st.write(f"**Communities:** {len(set(comm_vals)):,}")

with col2:
    st.subheader("Network")

    # Apply filters to create subgraph
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

        keep_nodes.add(n)

    H = G.subgraph(keep_nodes).copy()

    # Edge type filter
    if selected_types and "type" in edges_df.columns:
        remove_edges = []
        for u, v, attrs in H.edges(data=True):
            t = attrs.get("type", None)
            if t is None:
                continue
            if str(t) not in selected_types:
                remove_edges.append((u, v))
        H.remove_edges_from(remove_edges)

        # Remove isolates created by edge filtering
        isolates = [n for n in H.nodes() if H.degree(n) == 0]
        H.remove_nodes_from(isolates)

    net = to_pyvis(
        H,
        height=height,
        physics=physics,
        node_color_attr=node_color_attr,
        node_size_mode=node_size_mode,
        label_attr=label_attr,
        edge_label_attr=edge_label_attr,
        edge_color_attr=edge_color_attr,
        max_nodes=max_nodes,
    )

    html = net.generate_html()
    st.components.v1.html(html, height=int(height.replace("px", "")) + 30, scrolling=True)

    with st.expander("Download filtered subgraph as CSV"):
        out_nodes = pd.DataFrame([{"id": n, **attrs} for n, attrs in H.nodes(data=True)])
        out_edges = pd.DataFrame([{"source": u, "target": v, **attrs} for u, v, attrs in H.edges(data=True)])

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