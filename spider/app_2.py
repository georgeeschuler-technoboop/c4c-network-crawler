# app.py
import json
import math
import time
from typing import Optional, Dict, Any, List

import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network

# =============================================================================
# App Versioning
# =============================================================================
APP_VERSION = "0.2.7"

VERSION_HISTORY = [
    ("0.2.7", "Add animation presets (Gentle, Medium, Wild) and pause/resume control to PyVis visualization."),
    ("0.2.6", "Fix blank renders by using PyVis inline assets (cdn_resources='in_line'). Add filter-by-community. Improve PyVis physics options."),
    ("0.2.5", "Stabilize PyVis options: JSON-only set_options, empty-graph guard, debug mode."),
    ("0.2.4", "Add visible version + version history in UI; keep layout modes + re-run layout + exports."),
    ("0.2.1", "Add Louvain community detection and color-by-community."),
    ("0.2.0", "Initial upload + mapping + basic filters and visualization."),
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


def pick_default(cols: List[str], candidates: List[str]) -> str:
    for c in candidates:
        if c in cols:
            return c
    return cols[0] if cols else ""


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


def compute_layout_positions(G: nx.Graph, layout_algo: str, seed: int) -> Dict[str, Dict[str, float]]:
    """
    Returns dict[node] -> {"x": float, "y": float}
    (Used when we want a frozen, no-drift layout.)
    """
    H = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    if H.number_of_nodes() == 0:
        return {}

    if layout_algo == "spring":
        pos = nx.spring_layout(H, seed=seed, iterations=140)
    elif layout_algo == "kamada_kawai":
        pos = nx.kamada_kawai_layout(H)
    else:
        pos = nx.spring_layout(H, seed=seed, iterations=140)

    scaled: Dict[str, Dict[str, float]] = {}
    for n, (x, y) in pos.items():
        scaled[str(n)] = {"x": float(x) * 900.0, "y": float(y) * 900.0}
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

        # Normalize optionals to standard keys used by UI
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


def pyvis_options_json(physics_enabled: bool, solver: str, damping: float, spring_len: int, stabilization_enabled: bool = True) -> str:
    """
    PyVis expects JSON only. (Do not pass JS like `var options = {...}`.)
    """
    options: Dict[str, Any] = {
        "interaction": {
            "hover": True,
            "tooltipDelay": 80,
            "hideEdgesOnDrag": True,
            "hideNodesOnDrag": False,
            "navigationButtons": True,
            "keyboard": True,
        },
        "nodes": {
            "borderWidth": 0,
            "font": {"size": 14, "face": "arial"},
        },
        "edges": {
            "smooth": {"type": "dynamic"},
            "color": {"opacity": 0.55},
        },
        "physics": {
            "enabled": bool(physics_enabled),
            "solver": solver,  # "barnesHut" or "forceAtlas2Based"
            "stabilization": {"enabled": bool(stabilization_enabled), "iterations": 250, "updateInterval": 25},
            "barnesHut": {
                "gravitationalConstant": -2200,
                "centralGravity": 0.25,
                "springLength": int(spring_len),
                "springConstant": 0.03,
                "damping": float(damping),
                "avoidOverlap": 0.4,
            },
            "forceAtlas2Based": {
                "gravitationalConstant": -40,
                "centralGravity": 0.02,
                "springLength": int(spring_len),
                "springConstant": 0.08,
                "damping": float(damping),
                "avoidOverlap": 0.5,
            },
        },
        "layout": {"improvedLayout": True},
    }
    return json.dumps(options)


def insert_pause_resume_button(html: str, physics_active: bool, continuous_simulation: bool) -> str:
    """
    Inject a floating Pause/Resume button into the PyVis HTML when physics is active
    and continuous simulation is enabled (stabilization disabled).
    """
    # Only inject button when physics is active and continuous simulation is on
    if not physics_active or not continuous_simulation:
        return html
    
    # JavaScript code for the pause/resume button
    button_script = """
<script type="text/javascript">
(function() {
    // Wait for the network to be initialized
    window.addEventListener('load', function() {
        // Try to find the network instance - PyVis typically assigns it to 'network' variable
        var checkNetwork = setInterval(function() {
            if (typeof network !== 'undefined' && network) {
                clearInterval(checkNetwork);
                initPauseButton();
            }
        }, 100);
        
        function initPauseButton() {
            var isPaused = false;
            
            // Create the button
            var btn = document.createElement('button');
            btn.id = 'pauseResumeBtn';
            btn.innerHTML = 'Pause';
            btn.style.position = 'fixed';
            btn.style.top = '20px';
            btn.style.right = '20px';
            btn.style.zIndex = '9999';
            btn.style.padding = '10px 20px';
            btn.style.backgroundColor = '#4c78a8';
            btn.style.color = 'white';
            btn.style.border = 'none';
            btn.style.borderRadius = '5px';
            btn.style.cursor = 'pointer';
            btn.style.fontSize = '14px';
            btn.style.fontFamily = 'Arial, sans-serif';
            btn.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
            
            // Add hover effect
            btn.onmouseover = function() {
                this.style.backgroundColor = '#3d5f85';
            };
            btn.onmouseout = function() {
                this.style.backgroundColor = '#4c78a8';
            };
            
            // Button click handler
            btn.onclick = function() {
                try {
                    if (isPaused) {
                        // Resume simulation
                        if (typeof network.startSimulation === 'function') {
                            network.startSimulation();
                        } else if (typeof network.physics !== 'undefined') {
                            network.physics.physicsEnabled = true;
                            if (typeof network.stabilize === 'function') {
                                network.stabilize();
                            }
                        }
                        btn.innerHTML = 'Pause';
                        isPaused = false;
                    } else {
                        // Pause simulation
                        if (typeof network.stopSimulation === 'function') {
                            network.stopSimulation();
                        } else if (typeof network.physics !== 'undefined') {
                            network.physics.physicsEnabled = false;
                        }
                        btn.innerHTML = 'Resume';
                        isPaused = true;
                    }
                } catch (e) {
                    console.error('Error toggling simulation:', e);
                }
            };
            
            // Append button to body
            document.body.appendChild(btn);
        }
    });
})();
</script>
"""
    
    # Insert the script before the closing </body> tag
    if "</body>" in html:
        html = html.replace("</body>", button_script + "\n</body>")
    else:
        # Fallback: append to end of HTML
        html = html + button_script
    
    return html


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
    solver: str,
    damping: float,
    spring_len: int,
    continuous_simulation: bool = False,
) -> Network:
    # CRITICAL: inline assets prevent blank render on Streamlit Cloud / CSP environments
    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="#111111",
        directed=isinstance(G, nx.DiGraph),
        cdn_resources="in_line",
    )

    # If positions are provided, force physics off to prevent drift
    if positions:
        physics_enabled = False

    net.toggle_physics(physics_enabled)

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G) if node_size_mode == "betweenness" else {}

    # A clean categorical palette (repeatable)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
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

        # label
        if label_attr and label_attr in attrs and pd.notna(attrs.get(label_attr)):
            label = str(attrs.get(label_attr))
        else:
            label = n_str

        # color
        color = "#4c78a8"
        if node_color_attr and node_color_attr in attrs and pd.notna(attrs.get(node_color_attr)):
            cat = str(attrs.get(node_color_attr))
            if cat not in category_to_color:
                category_to_color[cat] = palette[color_idx % len(palette)]
                color_idx += 1
            color = category_to_color[cat]

        # size
        if node_size_mode == "degree":
            size = 10 + 3 * math.log1p(degree.get(n, 0))
        elif node_size_mode == "betweenness":
            size = 10 + 45 * betweenness.get(n, 0)
        else:
            size = 12

        # tooltip
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

        # edge label
        e_label = ""
        if edge_label_attr and edge_label_attr in attrs and pd.notna(attrs.get(edge_label_attr)):
            e_label = str(attrs.get(edge_label_attr))

        # edge color
        e_color = "#999999"
        if edge_color_attr and edge_color_attr in attrs and pd.notna(attrs.get(edge_color_attr)):
            e_cat = str(attrs.get(edge_color_attr))
            if e_cat not in category_to_color:
                category_to_color[e_cat] = palette[color_idx % len(palette)]
                color_idx += 1
            e_color = category_to_color[e_cat]

        # width by weight
        w = attrs.get("weight", 1.0)
        try:
            width = 1 + 2 * math.log1p(float(w))
        except Exception:
            width = 2

        net.add_edge(u_str, v_str, label=e_label, color=e_color, width=width, title=str(attrs))

    # JSON-only options (no JS)
    # When continuous_simulation is True, disable stabilization to keep animation running
    stabilization_enabled = not continuous_simulation
    net.set_options(pyvis_options_json(physics_enabled, solver=solver, damping=damping, spring_len=spring_len, stabilization_enabled=stabilization_enabled))
    return net


# =============================================================================
# Session State
# =============================================================================
if "layout_seed" not in st.session_state:
    st.session_state.layout_seed = int(time.time()) % 1_000_000

# Initialize physics control states
if "solver" not in st.session_state:
    st.session_state.solver = "barnesHut"
if "damping" not in st.session_state:
    st.session_state.damping = 0.35
if "spring_len" not in st.session_state:
    st.session_state.spring_len = 120
if "continuous_simulation" not in st.session_state:
    st.session_state.continuous_simulation = False
if "animation_intensity" not in st.session_state:
    st.session_state.animation_intensity = 1.0


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

    layout_mode = st.radio(
        "Layout mode",
        ["Auto (best)", "Physics (PyVis)", "Frozen (Python layout)"],
        index=0,
        help="Auto: physics for small graphs, frozen for big graphs. Frozen avoids drift and enables position export.",
    )

    physics_checkbox = st.checkbox("Enable physics (animated)", value=True)
    layout_algo = st.selectbox("Python layout algorithm", ["spring", "kamada_kawai"], index=0)

    st.divider()
    st.header("PyVis physics tuning")
    
    # Animation presets
    st.subheader("Animation presets")
    preset_options = ["(none)", "Gentle", "Medium", "Wild"]
    selected_preset = st.radio(
        "Choose a preset or customize below:",
        preset_options,
        index=0,
        help="Presets configure physics settings for different animation styles."
    )
    
    # Apply preset if selected
    if selected_preset == "Gentle":
        st.session_state.solver = "barnesHut"
        st.session_state.damping = 0.7
        st.session_state.spring_len = 200
        st.session_state.animation_intensity = 0.7
        st.session_state.continuous_simulation = False
    elif selected_preset == "Medium":
        st.session_state.solver = "barnesHut"
        st.session_state.damping = 0.45
        st.session_state.spring_len = 120
        st.session_state.animation_intensity = 1.0
        st.session_state.continuous_simulation = False
    elif selected_preset == "Wild":
        st.session_state.solver = "forceAtlas2Based"
        st.session_state.damping = 0.25
        st.session_state.spring_len = 60
        st.session_state.animation_intensity = 1.5
        st.session_state.continuous_simulation = True
    
    st.markdown("---")
    st.caption("**Manual controls** (override presets):")
    
    solver = st.selectbox(
        "Physics solver", 
        ["barnesHut", "forceAtlas2Based"], 
        index=0 if st.session_state.solver == "barnesHut" else 1,
        key="solver_widget"
    )
    damping = st.slider(
        "Damping (higher = settles faster)", 
        0.05, 0.95, 
        st.session_state.damping, 
        0.05,
        key="damping_widget"
    )
    spring_len = st.slider(
        "Spring length", 
        30, 300, 
        st.session_state.spring_len, 
        10,
        key="spring_len_widget"
    )
    continuous_simulation = st.checkbox(
        "Continuous simulation (disable stabilization)",
        value=st.session_state.continuous_simulation,
        help="Keep the animation running continuously. Enables Pause/Resume button.",
        key="continuous_sim_widget"
    )
    
    # Update session state from widgets (when manually changed)
    if solver != st.session_state.solver:
        st.session_state.solver = solver
    if damping != st.session_state.damping:
        st.session_state.damping = damping
    if spring_len != st.session_state.spring_len:
        st.session_state.spring_len = spring_len
    if continuous_simulation != st.session_state.continuous_simulation:
        st.session_state.continuous_simulation = continuous_simulation


    st.divider()
    st.header("Performance")
    max_nodes = st.slider("Max nodes to render", 50, 5000, 1500, step=50)
    debug_mode = st.checkbox("Debug mode", value=False)

    st.divider()
    st.header("Styling")
    node_size_mode = st.selectbox("Node size by", ["degree", "betweenness", "fixed"], index=0)

    st.divider()
    st.header("Layout actions")
    if st.button("Re-run layout (new seed)"):
        st.session_state.layout_seed = (st.session_state.layout_seed + 1) % 1_000_000

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
st.caption("Pick which columns to use (supports Polinode exports and most other formats).")

mcol1, mcol2 = st.columns([0.5, 0.5], gap="large")

with mcol1:
    st.markdown("### Edges (required)")
    e_cols = list(edges_df_raw.columns)

    default_source = pick_default(e_cols, ["Source", "source", "From", "from", "src", "Src"])
    default_target = pick_default(e_cols, ["Target", "target", "To", "to", "dst", "Dst"])
    default_type = pick_default(e_cols, ["Type", "type", "Edge Type", "edge_type", "relationship", "Relationship"])
    default_weight = pick_default(e_cols, ["Amount", "amount", "Weight", "weight", "Value", "value", "count", "Count"])
    default_label = pick_default(e_cols, ["label", "Label", "Purpose", "purpose", "Description", "description"])

    edge_source_col = st.selectbox("Edge source column", e_cols, index=e_cols.index(default_source))
    edge_target_col = st.selectbox("Edge target column", e_cols, index=e_cols.index(default_target))

    edge_type_col_sel = st.selectbox(
        "Edge type column (optional)", ["(none)"] + e_cols,
        index=(["(none)"] + e_cols).index(default_type) if default_type in e_cols else 0
    )
    edge_type_col = None if edge_type_col_sel == "(none)" else edge_type_col_sel

    edge_weight_col_sel = st.selectbox(
        "Edge weight column (optional)", ["(none)"] + e_cols,
        index=(["(none)"] + e_cols).index(default_weight) if default_weight in e_cols else 0
    )
    edge_weight_col = None if edge_weight_col_sel == "(none)" else edge_weight_col_sel

    edge_label_col_sel = st.selectbox(
        "Edge label column (optional)", ["(none)"] + e_cols,
        index=(["(none)"] + e_cols).index(default_label) if default_label in e_cols else 0
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

all_node_attrs = sorted({k for _, a in G.nodes(data=True) for k in a.keys()})
all_edge_attrs = sorted({k for _, _, a in G.edges(data=True) for k in a.keys()})

# Main layout
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

    # Edge type filter
    type_values = sorted({str(a.get("type")) for _, _, a in G.edges(data=True) if a.get("type") not in [None, ""]})
    if type_values:
        selected_types = st.multiselect("Edge type filter", type_values, default=type_values)
    else:
        selected_types = []

    search = st.text_input("Search node (by id or label text)", value="").strip().lower()

    # Community filter (your requested feature)
    community_filter: Optional[List[int]] = None
    if "_community" in all_node_attrs:
        comm_vals = sorted({a.get("_community") for _, a in G.nodes(data=True) if a.get("_community") is not None})
        if comm_vals:
            community_filter = st.multiselect(
                "Filter by community (isolate selected)",
                comm_vals,
                default=[],
            )

    st.divider()
    st.subheader("Stats")
    st.write(f"**Nodes:** {G.number_of_nodes():,}")
    st.write(f"**Edges:** {G.number_of_edges():,}")
    if G.number_of_nodes() > 0:
        st.write(f"**Density:** {nx.density(G):.4f}")

    # Community legend
    if "_community" in all_node_attrs:
        st.divider()
        st.subheader("Community legend")
        comm_list = [a.get("_community") for _, a in G.nodes(data=True) if a.get("_community") is not None]
        if comm_list:
            legend = (
                pd.Series(comm_list, name="community")
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

        # community filter
        if "_community" in attrs and community_filter and len(community_filter) > 0:
            if attrs.get("_community") not in community_filter:
                continue

        # text search
        if search:
            hay = n.lower()
            if label_attr and label_attr in attrs and pd.notna(attrs.get(label_attr)):
                hay += " " + str(attrs.get(label_attr)).lower()
            if search not in hay:
                continue

        keep_nodes.add(n)

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

        isolates = [n for n in H.nodes() if H.degree(n) == 0]
        H.remove_nodes_from(isolates)

    # Performance cap (keep deterministic order)
    if H.number_of_nodes() > max_nodes:
        keep = list(H.nodes())[:max_nodes]
        H = H.subgraph(keep).copy()

    if H.number_of_nodes() == 0 or H.number_of_edges() == 0:
        st.warning(
            "No graph to render after filters. Try: Minimum degree = 0, clear Search, clear Edge type filter, clear Community filter."
        )
        if debug_mode:
            st.write(
                {
                    "G_nodes": G.number_of_nodes(),
                    "G_edges": G.number_of_edges(),
                    "H_nodes": H.number_of_nodes(),
                    "H_edges": H.number_of_edges(),
                }
            )
        st.stop()

    # Layout mode behavior
    N = H.number_of_nodes()
    positions = None

    if layout_mode == "Physics (PyVis)":
        physics_enabled = bool(physics_checkbox)
        # if huge, warn
        if N > 800 and physics_enabled:
            st.warning("Physics on very large graphs can be slow/drifty. Try Frozen or Auto.")
    elif layout_mode == "Frozen (Python layout)":
        physics_enabled = False
        positions = compute_layout_positions(H, layout_algo, seed=st.session_state.layout_seed)
    else:  # Auto
        if N > 350:
            physics_enabled = False
            positions = compute_layout_positions(H, layout_algo, seed=st.session_state.layout_seed)
        else:
            physics_enabled = bool(physics_checkbox)

    net = to_pyvis(
        H,
        height="750px",
        physics_enabled=physics_enabled,
        node_color_attr=node_color_attr,
        node_size_mode=node_size_mode,
        label_attr=label_attr,
        edge_label_attr=edge_label_attr,
        edge_color_attr=edge_color_attr,
        max_nodes=max_nodes,
        positions=positions,
        solver=st.session_state.solver,
        damping=st.session_state.damping,
        spring_len=st.session_state.spring_len,
        continuous_simulation=st.session_state.continuous_simulation,
    )

    # Generate HTML and inject pause/resume button if needed
    pyvis_html = net.generate_html()
    pyvis_html = insert_pause_resume_button(
        pyvis_html, 
        physics_enabled, 
        st.session_state.continuous_simulation
    )
    
    st.components.v1.html(pyvis_html, height=780, scrolling=True)

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

        if positions:
            pos_df = pd.DataFrame([{"id": n, "x": positions[n]["x"], "y": positions[n]["y"]} for n in positions.keys()])
            st.download_button(
                "Download node_positions.csv",
                data=pos_df.to_csv(index=False).encode("utf-8"),
                file_name="node_positions.csv",
                mime="text/csv",
            )
        else:
            st.info("Node positions export is available when using **Frozen (Python layout)** or Auto when it freezes.")
