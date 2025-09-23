import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Project Portfolio Efficient Frontier", layout="wide")

st.title("📊 Project Portfolio - Efficient Frontier with Dependencies")

# --- Sample CSV (clean) ---
SAMPLE_CSV = """Project,ExpectedReturn,Risk,Group,Dependencies
CMS Upgrade (XM Cloud),6,5,Infrastructure,
CMS Modules + Branding,6,5,Foundational,CMS Upgrade (XM Cloud)
Data Infrastructure,7,6,Infrastructure,
Provider/Location Cleanup,6,6,Foundational,Data Infrastructure
Medical Services Pages,8,6,Tactical,CMS Upgrade (XM Cloud)
AI Keywords,9,7,Tactical,Medical Services Pages
Insurance Info,7,4,Tactical,
Content Hub,8,5,Foundational,CMS Upgrade (XM Cloud)
Clinical Trials Module,9,6,Tactical,Content Hub
"""

# --- File upload or use sample ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    st.info("No file uploaded, using built-in sample dataset.")
    df = pd.read_csv(io.StringIO(SAMPLE_CSV), skipinitialspace=True)

# Ensure correct columns
required_cols = ["Project", "ExpectedReturn", "Risk", "Group", "Dependencies"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

# --- Editable DataFrame ---
st.subheader("Edit Project Data")
df = st.data_editor(df, num_rows="dynamic")

# --- Handle dependencies ---
def enforce_dependencies(selected_projects, df):
    """Ensure that if a project is selected, all its dependencies are also selected."""
    added = True
    while added:
        added = False
        for _, row in df.iterrows():
            proj = row["Project"]
            if proj in selected_projects and pd.notna(row["Dependencies"]):
                deps = [d.strip() for d in row["Dependencies"].split(",") if d.strip()]
                for dep in deps:
                    if dep not in selected_projects:
                        selected_projects.add(dep)
                        added = True
    return selected_projects

# --- Efficient Frontier Calculation ---
def efficient_frontier(df):
    projects = df.to_dict("records")
    n = len(projects)

    portfolios = []

    for mask in range(1, 1 << n):
        chosen = [projects[i] for i in range(n) if (mask & (1 << i))]
        names = [p["Project"] for p in chosen]

        # enforce dependencies
        names = enforce_dependencies(set(names), df)
        chosen = [p for p in projects if p["Project"] in names]

        if not chosen:
            continue

        total_return = np.mean([p["ExpectedReturn"] for p in chosen])
        total_risk = np.mean([p["Risk"] for p in chosen])

        portfolios.append((total_risk, total_return, names))

    # Sort by risk then maximize return
    portfolios = sorted(portfolios, key=lambda x: (x[0], -x[1]))
    frontier = []
    max_return = -np.inf
    for risk, ret, names in portfolios:
        if ret > max_return:
            frontier.append((risk, ret, names))
            max_return = ret
    return portfolios, frontier

# Convert numeric columns
df["ExpectedReturn"] = pd.to_numeric(df["ExpectedReturn"], errors="coerce")
df["Risk"] = pd.to_numeric(df["Risk"], errors="coerce")

# --- Run Analysis ---
portfolios, frontier = efficient_frontier(df)

# --- Plot ---
st.subheader("Efficient Frontier Plot")
fig, ax = plt.subplots(figsize=(10, 6))

all_risks = [p[0] for p in portfolios]
all_returns = [p[1] for p in portfolios]

ax.scatter(all_risks, all_returns, alpha=0.3, label="All Portfolios", c="gray")
ax.plot([p[0] for p in frontier], [p[1] for p in frontier], "r-o", label="Efficient Frontier")

ax.set_xlabel("Risk (Std Dev)")
ax.set_ylabel("Expected Return")
ax.set_title("Efficient Frontier for Project Portfolio")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# --- Show frontier portfolios ---
st.subheader("Frontier Portfolios")
frontier_data = []
for i, (risk, ret, names) in enumerate(frontier, 1):
    st.markdown(f"**Portfolio {i}:** Return = {ret:.2f}, Risk = {risk:.2f}")
    st.write(", ".join(names))
    frontier_data.append({"Portfolio": i, "Return": ret, "Risk": risk, "Projects": "; ".join(names)})

frontier_df = pd.DataFrame(frontier_data)

# --- Download Updated Data ---
st.subheader("Download Data")

csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="💾 Download Updated Projects CSV",
    data=csv_data,
    file_name="updated_projects.csv",
    mime="text/csv",
)

csv_frontier = frontier_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📈 Download Frontier Portfolios CSV",
    data=csv_frontier,
    file_name="frontier_portfolios.csv",
    mime="text/csv",
)
