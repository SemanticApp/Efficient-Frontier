import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from typing import List, Dict, Tuple, Set

st.set_page_config(page_title="Project Portfolio — Efficient Frontier (with Dependencies)", layout="wide")

st.title("Project Portfolio — Efficient Frontier Analyzer (with Groups & Dependencies)")
st.markdown("""
**Upload or edit** a project table, classify projects into groups (Foundational / Tactical / Infrastructure), and declare dependencies.
The app computes efficient frontiers (analytical) and finds unconstrained and constrained max-Sharpe portfolios.  
Dependencies are enforced: selecting a project will automatically include its prerequisite infrastructure projects in the candidate portfolio.
""")


#
# ---------- Sample data
#
SAMPLE_CSV = """Project,ExpectedReturn,Risk,Group,Dependencies
CMS Upgrade,0.12,0.25,Infrastructure,
CMS Modules + Branding,0.08,0.20,Foundational,CMS Upgrade
Data Infrastructure,0.10,0.22,Infrastructure,
Provider Cleanup,0.09,0.18,Foundational,Data Infrastructure
AI for Keywords,0.20,0.35,Tactical,CMS Upgrade,Provider Cleanup
Open Scheduling,0.18,0.30,Tactical,CMS Upgrade
Insurance Info,0.06,0.10,Tactical,
Campaign Landing Pages,0.07,0.12,Tactical,
Chatbot + LiveChat,0.14,0.20,Tactical,CMS Upgrade
Content Hub,0.10,0.20,Foundational,
Clinical Trials Module,0.13,0.28,Foundational,Data Infrastructure
"""

#
# ---------- Utilities
#
def parse_dependencies_field(val) -> List[str]:
    # Accept multiple formats: comma separated string, or already list
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val)
    if s.strip() == "":
        return []
    # allow comma-separated or semicolon separated
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [p for p in parts if p]

def build_dep_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    mapping = {}
    for _, row in df.iterrows():
        project = str(row["Project"])
        deps = parse_dependencies_field(row.get("Dependencies", ""))
        mapping[project] = deps
    return mapping

def resolve_dependencies(selected: List[str], dep_map: Dict[str, List[str]]) -> List[str]:
    # Return a list including selected and all recursive dependencies
    included: Set[str] = set()
    stack = list(selected)
    while stack:
        p = stack.pop()
        if p in included:
            continue
        included.add(p)
        for d in dep_map.get(p, []):
            if d not in included:
                stack.append(d)
    return list(included)

def safe_invert(mat: np.ndarray, eps=1e-8) -> np.ndarray:
    # regularize small eigenvalues
    try:
        inv = np.linalg.inv(mat)
        return inv
    except np.linalg.LinAlgError:
        # regularize diagonal
        reg = np.eye(mat.shape[0]) * eps
        return np.linalg.inv(mat + reg)

def compute_analytical_frontier(mu: np.ndarray, cov: np.ndarray, points=100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inv_cov = safe_invert(cov)
    ones = np.ones(len(mu))
    A = ones @ inv_cov @ ones
    B = ones @ inv_cov @ mu
    C = mu @ inv_cov @ mu
    denom = A * C - B * B
    if abs(denom) < 1e-12:
        # degenerate; return empty arrays
        return np.array([]), np.array([]), np.array([[]])
    min_ret = mu.min() * 0.8
    max_ret = mu.max() * 1.2
    target_returns = np.linspace(min_ret, max_ret, points)
    risks = []
    weights = []
    for r in target_returns:
        lam = (C - B * r) / denom
        gamma = (A * r - B) / denom
        w = inv_cov @ (lam * ones + gamma * mu)
        var = w.T @ cov @ w
        risks.append(np.sqrt(max(var, 0.0)))
        weights.append(w)
    return target_returns, np.array(risks), np.array(weights)

def unconstrained_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float):
    inv_cov = safe_invert(cov)
    excess = mu - rf
    w = inv_cov @ excess
    if np.sum(w) != 0:
        w = w / np.sum(w)
    ret = w @ mu
    vol = np.sqrt(max(w.T @ cov @ w, 0.0))
    sr = (ret - rf) / vol if vol > 0 else np.nan
    return w, ret, vol, sr

def constrained_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float):
    # try scipy minimize; fallback to None if not available
    try:
        from scipy.optimize import minimize
    except Exception:
        return None
    n = len(mu)
    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(max(w.T @ cov @ w, 1e-12))
        return - (ret - rf) / vol
    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0) for _ in range(n)]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    res = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons)
    if res.success:
        w = res.x
        ret = w @ mu
        vol = np.sqrt(max(w.T @ cov @ w, 0.0))
        sr = (ret - rf) / vol if vol > 0 else np.nan
        return w, ret, vol, sr
    else:
        return None

#
# ---------- Sidebar controls & uploads
#
with st.sidebar:
    st.header("Inputs & Options")
    uploaded = st.file_uploader("Upload CSV (Project,ExpectedReturn,Risk,Group,Dependencies) — or leave empty for sample", type=["csv", "txt"])
    cov_upload = st.file_uploader("Optional: Covariance matrix CSV (square)", type=["csv", "txt"])
    rf = st.number_input("Risk-free rate (rf)", value=0.0, step=0.001, format="%.4f")
    corr_assumption = st.slider("Assume constant correlation (if no cov supplied)", -0.5, 0.9, 0.0)
    show_groups = st.multiselect("Show groups on plot (select to overlay)", ["Foundational", "Tactical", "Infrastructure"], default=["Foundational", "Tactical", "Infrastructure"])
    use_constrained = st.checkbox("Attempt constrained (no short selling) optimization if SciPy available", value=True)
    st.markdown("---")
    st.markdown("**Dependency enforcement**")
    auto_include_deps = st.checkbox("Automatically include dependency projects when a project is selected", value=True)
    st.markdown("Dependencies should exactly match project names and be comma-separated.")

#
# ---------- Load projects dataframe
#
if uploaded is None:
    st.info("No CSV uploaded — using sample data. You can edit it below.")
    df = pd.read_csv(io.StringIO(SAMPLE_CSV))
else:
    df = pd.read_csv(uploaded)

# Ensure expected columns exist; if not, create defaults
for col in ["Group", "Dependencies"]:
    if col not in df.columns:
        df[col] = ""

# Normalize columns and types
df["Project"] = df["Project"].astype(str)
df["ExpectedReturn"] = pd.to_numeric(df["ExpectedReturn"], errors="coerce")
df["Risk"] = pd.to_numeric(df["Risk"], errors="coerce")
df["Group"] = df["Group"].fillna("").astype(str)
df["Dependencies"] = df["Dependencies"].fillna("").astype(str)

# Provide an editable table
st.subheader("Project table (editable)")
edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# Use edited data for everything from here on
df = edited.copy()

# Build dependency map and validate project names used in Dependencies
dep_map = build_dep_map(df)
all_projects = set(df["Project"].tolist())
invalid_deps = {}
for p, deps in dep_map.items():
    invalid = [d for d in deps if d not in all_projects]
    if invalid:
        invalid_deps[p] = invalid

if invalid_deps:
    st.warning("Some dependencies reference unknown projects. Please correct them in the table.")
    for p, invalid in invalid_deps.items():
        st.write(f"- {p}: unknown deps -> {invalid}")

#
# ---------- Covariance matrix
#
n = len(df)
if cov_upload is not None:
    cov_df = pd.read_csv(cov_upload, index_col=0)
    # basic validation
    if cov_df.shape[0] != cov_df.shape[1] or cov_df.shape[0] != n:
        st.error("Covariance matrix must be square and match the number of projects.")
        st.stop()
    # reorder to match df project order if necessary
    try:
        cov_df = cov_df.loc[df["Project"], df["Project"]]
    except Exception:
        st.warning("Covariance matrix headers do not match project names exactly; using raw matrix order.")
    cov = cov_df.values
else:
    sigma = df["Risk"].values
    corr = corr_assumption
    cov = np.outer(sigma, sigma) * corr
    np.fill_diagonal(cov, sigma ** 2)

mu = df["ExpectedReturn"].values

#
# ---------- Group selection & candidate pool
#
st.subheader("Select candidate projects to analyze")
# project selection multiselect (but we'll implement dependency auto-inclusion)
project_options = list(df["Project"])
selected_projects = st.multiselect("Pick projects to consider in the portfolio (you can pick none to use all)", project_options, default=project_options)

if len(selected_projects) == 0:
    st.info("No selection — all projects will be used as candidates.")
    selected_projects = project_options

if auto_include_deps:
    resolved = resolve_dependencies(selected_projects, dep_map)
    if set(resolved) != set(selected_projects):
        st.info(f"Dependencies added: {set(resolved) - set(selected_projects)}")
    selected_projects = sorted(resolved)

# Filter df to selected projects
mask = df["Project"].isin(selected_projects)
df_sel = df[mask].reset_index(drop=True)
sel_indices = [df.index[df["Project"] == p][0] for p in df_sel["Project"]]  # indices into df original
cov_sel = cov[np.ix_(sel_indices, sel_indices)]
mu_sel = df_sel["ExpectedReturn"].values

# Show selected table
st.write(f"**{len(df_sel)} candidate projects selected** (including enforced dependencies).")
st.dataframe(df_sel[["Project", "Group", "ExpectedReturn", "Risk", "Dependencies"]])

#
# ---------- Compute frontiers per group and combined
#
st.subheader("Efficient Frontier & Portfolios")
plot_fig, ax = plt.subplots(figsize=(9, 6))

# compute and plot frontiers for selected groups
colors = {"Foundational": "C0", "Tactical": "C1", "Infrastructure": "C2", "Combined": "k"}
legend_items = []
combined_targets, combined_risks, _ = compute_analytical_frontier(mu_sel, cov_sel, points=100)
if len(combined_targets) > 0:
    ax.plot(combined_risks, combined_targets, color=colors["Combined"], linestyle="--", label="Combined Frontier")

# Overlay group frontiers if requested and group has at least 2 projects
for g in ["Foundational", "Tactical", "Infrastructure"]:
    if g not in show_groups:
        continue
    df_g = df_sel[df_sel["Group"].str.strip().str.lower() == g.lower()]
    if len(df_g) < 2:
        continue
    idxs = [df_sel.index[df_sel["Project"] == p][0] for p in df_g["Project"]]
    mu_g = df_g["ExpectedReturn"].values
    cov_g = cov_sel[np.ix_(idxs, idxs)]
    t, r, _ = compute_analytical_frontier(mu_g, cov_g, points=80)
    if len(t) > 0:
        ax.plot(r, t, color=colors[g], label=f"{g} Frontier")

# plot all candidate projects as scatter (risk vs return)
proj_risks = df_sel["Risk"].values
proj_returns = df_sel["ExpectedReturn"].values
ax.scatter(proj_risks, proj_returns, s=80, zorder=5)
for i, txt in enumerate(df_sel["Project"]):
    ax.annotate(txt, (proj_risks[i], proj_returns[i]), textcoords="offset points", xytext=(6,4), fontsize=9)

ax.set_xlabel("Risk (Std Dev)")
ax.set_ylabel("Expected Return")
ax.set_title("Efficient Frontier(s) and Candidate Projects")
ax.grid(True)
ax.legend()
st.pyplot(plot_fig)

#
# ---------- Unconstrained & Constrained Max Sharpe
#
st.markdown("### Max-Sharpe portfolios")
w_uncon, ret_u, vol_u, sr_u = unconstrained_max_sharpe(mu_sel, cov_sel, rf)
st.markdown(f"**Unconstrained max-Sharpe** — Sharpe: **{sr_u:.4f}**, Expected Return: **{ret_u:.4f}**, Risk: **{vol_u:.4f}**")
w_uncon_df = pd.DataFrame({"Project": df_sel["Project"], "Weight": w_uncon})
st.dataframe(w_uncon_df.style.format({"Weight":"{:.4f}"}))

w_constrained_result = None
if use_constrained:
    w_constrained_result = constrained_max_sharpe(mu_sel, cov_sel, rf)
    if w_constrained_result is None:
        st.warning("Constrained optimization not available (SciPy not installed) or failed.")
    else:
        w_c, ret_c, vol_c, sr_c = w_constrained_result
        st.markdown(f"**Constrained (no short selling) max-Sharpe** — Sharpe: **{sr_c:.4f}**, Expected Return: **{ret_c:.4f}**, Risk: **{vol_c:.4f}**")
        st.dataframe(pd.DataFrame({"Project": df_sel["Project"], "Weight": w_c}).style.format({"Weight":"{:.4f}"}))

#
# ---------- Prioritization & scoring
#
st.subheader("Prioritization & Ranking")
# simple standalone Sharpe
df_sel = df_sel.copy()
df_sel["Sharpe"] = (df_sel["ExpectedReturn"] - rf) / df_sel["Risk"]
df_sel["StandaloneRank"] = df_sel["Sharpe"].rank(ascending=False, method="min")
st.markdown("**Projects ranked by standalone Sharpe ratio (higher = better)**")
st.dataframe(df_sel.sort_values("StandaloneRank")[["Project", "Group", "ExpectedReturn", "Risk", "Sharpe", "StandaloneRank"]].style.format({"ExpectedReturn":"{:.4f}", "Risk":"{:.4f}", "Sharpe":"{:.4f}"}))

# Contribution to unconstrained portfolio
st.markdown("**Contribution to unconstrained max-Sharpe portfolio**")
contrib_df = pd.DataFrame({
    "Project": df_sel["Project"],
    "Weight_unconstrained": w_uncon,
    "Contribution": w_uncon * mu_sel
})
contrib_df = contrib_df.sort_values("Contribution", ascending=False).reset_index(drop=True)
st.dataframe(contrib_df.style.format({"Weight_unconstrained":"{:.4f}", "Contribution":"{:.4f}"}))

#
# ---------- Auto-include enforcement view and filtering
#
st.subheader("Dependency view & enforced inclusions")
st.markdown("If a project is selected and `auto-include` is enabled, dependent infrastructure/foundational projects are added automatically.")
dep_rows = []
for p in selected_projects:
    dep_rows.append({"Project": p, "Dependencies": ", ".join(dep_map.get(p, []))})
st.dataframe(pd.DataFrame(dep_rows))

#
# ---------- Download results
#
st.subheader("Export / Download")
# export edited projects CSV
csv_buf = df.to_csv(index=False).encode("utf-8")
st.download_button("Download edited projects CSV", data=csv_buf, file_name="projects_edited.csv", mime="text/csv")

# export selected portfolio weights (use constrained if available else unconstrained)
if w_constrained_result is not None:
    chosen_weights = w_constrained_result[0]
    chosen_label = "constrained_max_sharpe"
else:
    chosen_weights = w_uncon
    chosen_label = "unconstrained_max_sharpe"
port_df = pd.DataFrame({"Project": df_sel["Project"], "Weight": chosen_weights, "ExpectedReturn": df_sel["ExpectedReturn"], "Risk": df_sel["Risk"]})
st.download_button("Download chosen-portfolio CSV", data=port_df.to_csv(index=False).encode("utf-8"), file_name=f"portfolio_{chosen_label}.csv", mime="text/csv")

#
# ---------- Optional: group-level reports and notes
#
st.subheader("Notes & Recommendations")
st.markdown("""
- **Infrastructure / Foundational projects** often enable higher returns for Tactical projects. Use `Dependencies` to indicate these relationships so they are always included when a dependent project is chosen.
- If you want to treat Infrastructure as *mandatory* (i.e., always in any portfolio), simply select them in the project selection list and they will be included.
- **Covariance matrix matters**: if many projects are correlated (e.g., content migrations + CMS), provide a covariance matrix for accurate frontier results. Otherwise the app uses a constant correlation assumption.
- Use the constrained optimization result for realistic 'project portfolio' recommendations (no negative weights).
""")

st.caption("Built with ❤️ — tweak ExpectedReturn and Risk in the table to experiment with 'what-if' scenarios.")
