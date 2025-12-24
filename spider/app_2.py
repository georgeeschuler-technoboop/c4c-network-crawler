import math
import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network

st.set_page_config(page_title="Mini Network Mapper", layout="wide")


# ----------------------------
# Helpers
# ----------------------------
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


def detect_communities(G: nx.Graph) -> dict:
    """node -> community_id (Louvain if available, else greedy modularity fallback)"""
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


def safe_float(x, default=1.0) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def build_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    directed: bool,
    node_id_col: str | None,
    edge_source_col: str,
    edge_target_col: str,
    edge_type_col: str | None,
    edge_weight_col: str | None,
    edge_label_col: str | None,
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
        raise ValueError("Edges mapping invalid: source/target columns not found.")

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

        # Normalize optional fields onto standard keys
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

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G) if node_size_mode == "betweenness" else {}

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
        H = G.subgraph(nodes).copy()
    else:
        H = G

    category_to_color = {}
    color_idx = 0

    for n, attrs in H.nodes(data=True):
        if label_attr and label_attr in attrs and pd.notna(attrs.get(label_attr)):
            label = str(attrs.get(label_attr))
        else:
            label = str(n)

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

        net.add_node(n, label=label, color=color, size=size, title=title)

    for u, v, attrs in H.edges(data=True):
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

        net.add_edge(u, v, label=e_label, color=e_color, width=width, title=str(attrs))

    net.set_options("""
    var options = {
      "nodes": { "borderWidth": 0 },
      "edges": { "smooth": { "type": "dynamic" } },
      "interaction": { "hover": true, "tooltipDelay": 100, "hideEdgesOnDrag": true },
      "physics": { "enabled": true, "stabilization": { "iterations": 200 } }
    }
    """)

    return net


# ----------------------------
# UI
# ----------------------------
st.title("Mini Network Mapper (Polinode/Gephi-lite)")

with st.sidebar:
    st.header("Upload")

    st.markdown("**Option A (recommended):** Upload a single Polinode-style workbook (`.xlsx`) with sheets `Nodes` and `Edges`.")
    workbook = st.file_uploader("Upload workbook (.xlsx)", type=["xlsx", "xls"], key="workbook")

    st.markdown("---")
    st.markdown("**Option B:** Upload CSVs")
    edges_file = st.file_uploader("Upload edges.csv", type=["csv"], key="edges")
    nodes_file = st.file_uploader("Upload nodes.csv (optional)", type=["csv"], key="nodes")

    st.divider()
    st.header("Display settings")
    directed = st.checkbox("Directed graph", value=False)
    physics = st.checkbox("Physics layout", value=True)
    detect_comm = st.checkbox("Detect communities (Louvain)", value=False)
    height = st.selectbox("Canvas height", ["650px", "750px", "900px"], index=0)
    max_nodes = st.slider("Max nodes to render (performance)", 50, 5000, 1500, step=50)

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

# Column Mapping UI
st.subheader("Column Mapping")
st.caption("Pick which columns to use (works with Polinode exports and almost anything else).")

mcol1, mcol2 = st.columns([0.5, 0.5], gap="large")

with mcol1:
    st.markdown("### Edges (required)")
    e_cols = list(edges_df_raw.columns)
    if not e_cols:
        st.error("Edges table has no columns. Check your file.")
        st.stop()

    def pick_default(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return cols[0]

    # Polinode defaults are Source/Target/Type/Amount
    default_source = pick_default(e_cols, ["Source", "source", "From", "from"])
    default_target = pick_default(e_cols, ["Target", "target", "To", "to"])
    default_type = pick_default(e_cols, ["Type", "type", "Edge Type", "edge_type"])
    default_weight = pick_default(e_cols, ["Amount", "amount", "Weight", "weight", "Value", "value"])
    default_label = pick_default(e_cols, ["label", "Label", "Purpose", "purpose"])

    edge_source_col = st.selectbox("Edge source column", e_cols, index=e_cols.index(default_source))
    edge_target_col = st.selectbox("Edge target column", e_cols, index=e_cols.index(default_target))

    edge_type_col = st.selectbox("Edge type column (optional)", ["(none)"] + e_cols,
                                 index=(["(none)"] + e_cols).index(default_type) if default_type in e_cols else 0)
    edge_type_col = None if edge_type_col == "(none)" else edge_type_col

    edge_weight_col = st.selectbox("Edge weight column (optional)", ["(none)"] + e_cols,
                                   index=(["(none)"] + e_cols).index(default_weight) if default_weight in e_cols else 0)
    edge_weight_col = None if edge_weight_col == "(none)" else edge_weight_col

    edge_label_col = st.selectbox("Edge label column (optional)", ["(none)"] + e_cols,
                                  index=(["(none)"] + e_cols).index(default_label) if default_label in e_cols else 0)
    edge_label_col = None if edge_label_col == "(none)" else edge_label_col

with mcol2:
    st.markdown("### Nodes (optional)")
    if nodes_df_raw.empty:
        st.info("No nodes table found/uploaded — nodes will be created from edge endpoints.")
        node_id_col = None
    else:
        n_cols = list(nodes_df_raw.columns)
        # Polinode default is Name
        default_node_id = "Name" if "Name" in n_cols else n_cols[0]
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

# Attributes
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

    label_attr = st.selectbox("Node label attribute", ["(id)"] + all_node_attrs, index=0)
    label_attr = None if label_attr == "(id)" else label_attr

    edge_label_attr = st.selectbox("Edge label attribute", ["(none)"] + all_edge_attrs, index=0)
    edge_label_attr = None if edge_label_attr == "(none)" else edge_label_attr

    edge_color_attr = st.selectbox("Color edges by attribute", ["(none)"] + all_edge_attrs, index=0)
    edge_color_attr = None if edge_color_attr == "(none)" else edge_color_attr

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 0
    min_deg = st.slider("Minimum degree", 0, max_deg, 0)

    type_values = sorted({str(a.get("type")) for _, _, a in G.edges(data=True) if a.get("type") not in [None, ""]})
    selected_types = st.multiselect("Edge type filter (if available)", type_values, default=type_values)

    search = st.text_input("Search node (by id or label text)", value="").strip().lower()

    st.divider()
    st.subheader("Stats")
    st.write(f"**Source mode:** {source_mode}")
    st.write(f"**Nodes:** {G.number_of_nodes():,}")
    st.write(f"**Edges:** {G.number_of_edges():,}")
    if G.number_of_nodes() > 0:
        st.write(f"**Density:** {nx.density(G):.4f}")
    if "_community" in all_node_attrs:
        comm_vals = [a.get("_community") for _, a in G.nodes(data=True) if a.get("_community") is not None]
        if comm_vals:
            st.write(f"**Communities:** {len(set(comm_vals)):,}")

with col2:
    st.subheader("Network")

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

    st.components.v1.html(net.generate_html(), height=int(height.replace("px", "")) + 30, scrolling=True)

    with st.expander("Download filtered subgraph as CSV"):
        out_nodes = pd.DataFrame([{"id": n, **a} for n, a in H.nodes(data=True)])
        out_edges = pd.DataFrame([{"source": u, "target": v, **a} for u, v, a in H.edges(data=True)])

        st.download_button("Download nodes_filtered.csv",
                           data=out_nodes.to_csv(index=False).encode("utf-8"),
                           file_name="nodes_filtered.csv",
                           mime="text/csv")
        st.download_button("Download edges_filtered.csv",
                           data=out_edges.to_csv(index=False).encode("utf-8"),
                           file_name="edges_filtered.csv",
                           mime="text/csv")
