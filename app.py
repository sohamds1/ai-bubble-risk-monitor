import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import time
import random
import yfinance as yf
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AI Bubble Systemic Risk Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, dashboard look
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-med { color: #ffa421; font-weight: bold; }
    .risk-low { color: #21c354; font-weight: bold; }
    .metric-box {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA GENERATION & "LIVE" SIMULATION
# ==========================================

# NOTE: In a production app, these functions would call external APIs 
# (e.g., Bloomberg, Crunchbase, Cloud provider status pages).

@st.cache_data
def fetch_live_ai_risk_data():
    """Generates synthetic data for AI Labs."""
    labs = [
        "OpenAI-Analog", "Anthropic-Analog", "xAI-Clone", "Cohere-Like", 
        "FrontierLabX", "Mistral-Model", "DeepMind-Sim"
    ]
    data = []
    for lab in labs:
        # Randomized baselines
        data.append({
            "Lab Name": lab,
            "Cloud Commitments ($B)": round(random.uniform(2, 15), 1),
            "Burn Rate ($B/yr)": round(random.uniform(0.5, 5.0), 1),
            "Revenue ($B/yr)": round(random.uniform(0.1, 3.0), 1),
            "Cross-Dependency": round(random.uniform(0.3, 0.9), 2), # 0-1
            "Prob. of Distress": round(random.uniform(0.05, 0.4), 2),
            "Systemic Contribution": round(random.uniform(0.1, 0.9), 2)
        })
    df = pd.DataFrame(data)
    # Calculated field: Runway Imbalance
    df["Runway Risk"] = df["Burn Rate ($B/yr)"] / (df["Revenue ($B/yr)"] + 0.1)
    df["Source"] = "🤖 Synthetic (Est.)"
    return df

@st.cache_data(ttl=300) # Cache for 5 minutes to avoid hitting API limits
def fetch_live_hyperscaler_data():
    """Generates Real-Time data for Cloud Providers using Yahoo Finance."""
    # Ticker mapping
    tickers = {
        "NVIDIA": "NVDA", 
        "Microsoft": "MSFT", 
        "Google": "GOOGL", 
        "Amazon": "AMZN", 
        "Meta": "META"
    }
    
    data = []
    for name, ticker_symbol in tickers.items():
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            
            # Real Data Points
            price = info.get('currentPrice', 0)
            market_cap_t = info.get('marketCap', 0) / 1e12 # Trillions
            beta = info.get('beta', 1.0) # Volatility measure
            pe_ratio = info.get('trailingPE', 0)
            
            # Derived Risk Metrics based on Real Data
            # High P/E + High Beta = Higher Sensitivity
            sensitivity = min(0.95, (beta * 0.4) + (pe_ratio / 200))
            
            data.append({
                "Company": name,
                "Stock Price": f"${price:.2f}",
                "Market Cap ($T)": round(market_cap_t, 2),
                "P/E Ratio": round(pe_ratio, 1),
                "Sensitivity": round(sensitivity, 2),
                "Source": "✅ Real-Time (Yahoo)"
            })
        except Exception as e:
            # Fallback if API fails
            data.append({
                "Company": name,
                "Stock Price": "N/A",
                "Market Cap ($T)": 0,
                "P/E Ratio": 0,
                "Sensitivity": 0.5,
                "Source": "⚠️ API Error"
            })
            
    return pd.DataFrame(data)

@st.cache_data
def fetch_spv_data():
    """Generates synthetic data for GPU financing structures (SPVs)."""
    spvs = ["CoreWeave-Like SPV A", "Lambda-Like SPV B", "Generic GPU Lease Trust I", "Sovereign AI Fund Alpha"]
    data = []
    for spv in spvs:
        data.append({
            "SPV Name": spv,
            "Debt ($B)": round(random.uniform(1, 10), 1),
            "GPU Assets ($B)": round(random.uniform(1.2, 11), 1), # Usually slightly higher than debt
            "AI Demand Dependency": round(random.uniform(0.8, 1.0), 2), # High dependency
            "Default Prob": round(random.uniform(0.02, 0.25), 2),
            "Source": "🤖 Synthetic (Model)"
        })
    return pd.DataFrame(data)

def fetch_2008_comparison_data():
    """Static mapping for the analogy engine."""
    return pd.DataFrame([
        {"2008 Component": "Subprime Mortgages", "AI Equivalent": "Over-optimistic AI Revenue Projections", "Similarity": 85, "Explanation": "Both rely on cash flows from end-users that may not materialize quickly enough."},
        {"2008 Component": "CDOs (Collateralized Debt Obligations)", "AI Equivalent": "GPU-Backed SPVs & Cloud Commitments", "Similarity": 75, "Explanation": "Bundling risky assets (GPU leases) into investment vehicles for debt financing."},
        {"2008 Component": "Synthetic CDOs", "AI Equivalent": "Circular Revenue (Cloud $ -> AI Lab -> Cloud Revenue)", "Similarity": 90, "Explanation": "Money moving in a circle creates the illusion of growth without external value injection."},
        {"2008 Component": "Rating Agencies", "AI Equivalent": "VC Valuations & Hype Cycle", "Similarity": 80, "Explanation": "Gatekeepers assigning AAA ratings (or $100B valuations) based on theoretical future models."},
        {"2008 Component": "Lehman Brothers", "AI Equivalent": "A Major Frontier Lab Failure", "Similarity": 60, "Explanation": "The 'Too Big To Fail' node that triggers a liquidity freeze if it collapses."}
    ])

# ==========================================
# 3. METRIC CALCULATIONS
# ==========================================

def calculate_metrics(ai_df, hyper_df, spv_df, similarity_factor=0.0):
    """
    Computes the master Systemic Risk Score (0-100).
    similarity_factor is a manual override from the simulator.
    """
    # 1. AI Fragility (Weighted by Burn/Revenue imbalance)
    avg_distress = ai_df["Prob. of Distress"].mean() * 100
    
    # 2. Hyperscaler Exposure (Capex intensity replaced by Sensitivity from Real Data)
    # We use 'Sensitivity' as a proxy for downside risk
    capex_intensity = hyper_df["Sensitivity"].mean() * 100
    
    # 3. SPV Leverage Risk
    # Leverage ratio = Debt / Assets. If Debt > Assets, risk is high.
    total_debt = spv_df["Debt ($B)"].sum()
    total_assets = spv_df["GPU Assets ($B)"].sum()
    leverage_risk = (total_debt / total_assets) * 50 if total_assets > 0 else 100
    
    # Composite Score
    # Weights: AI Fragility (40%), SPV Leverage (30%), Hyperscaler Overbuild (30%)
    raw_score = (avg_distress * 0.4) + (leverage_risk * 0.3) + (capex_intensity * 0.3)
    
    # Apply simulation modifiers
    final_score = min(100, max(0, raw_score + similarity_factor))
    
    return round(final_score, 1)

def get_risk_level_text(score):
    if score < 30: return "LOW / STABLE", "risk-low"
    if score < 60: return "ELEVATED / CAUTION", "risk-med"
    return "CRITICAL / OVERHEATED", "risk-high"

# ==========================================
# 4. SESSION STATE & AUTO-REFRESH
# ==========================================

if 'last_updated' not in st.session_state:
    st.session_state.last_updated = datetime.now()
    st.session_state.ai_data = fetch_live_ai_risk_data()
    st.session_state.hyper_data = fetch_live_hyperscaler_data()
    st.session_state.spv_data = fetch_spv_data()
    st.session_state.simulation_offset = 0.0
    st.session_state.simulation_text = "Standard Monitoring Mode"

# Sidebar Controls
st.sidebar.title("Monitor Controls")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh (Live Mode)", value=False)
refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)

# Simulator in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("Scenario Simulator")
scenario = st.sidebar.selectbox("Trigger Event", 
    ["None (Live Data)", "Frontier Lab Default", "GPU Demand Crash", "2008 Liquidity Freeze"])
severity = st.sidebar.slider("Event Severity", 0, 100, 50)

# Logic to update simulation state based on sidebar
if scenario == "None (Live Data)":
    st.session_state.simulation_offset = 0.0
    st.session_state.simulation_text = "Live Market Conditions"
elif scenario == "Frontier Lab Default":
    st.session_state.simulation_offset = severity * 0.4 # High impact
    st.session_state.simulation_text = f"Simulating: Major Lab Failure (Severity {severity}%)"
elif scenario == "GPU Demand Crash":
    st.session_state.simulation_offset = severity * 0.3
    st.session_state.simulation_text = f"Simulating: GPU Resale Value Crash (Severity {severity}%)"
elif scenario == "2008 Liquidity Freeze":
    st.session_state.simulation_offset = severity * 0.5
    st.session_state.simulation_text = f"Simulating: Credit Markets Freeze (Severity {severity}%)"

# Auto-refresh logic
if auto_refresh:
    time.sleep(1) # Small delay to prevent tight loop resource hogging
    # Introduce small random drift to simulate live data updates
    st.session_state.ai_data["Prob. of Distress"] += np.random.uniform(-0.02, 0.02, len(st.session_state.ai_data))
    st.session_state.ai_data["Prob. of Distress"] = st.session_state.ai_data["Prob. of Distress"].clip(0, 1)
    st.session_state.last_updated = datetime.now()
    time.sleep(refresh_rate)
    st.rerun()

# Calculate Master Score
systemic_risk_score = calculate_metrics(
    st.session_state.ai_data, 
    st.session_state.hyper_data, 
    st.session_state.spv_data,
    st.session_state.simulation_offset
)

risk_text, risk_css = get_risk_level_text(systemic_risk_score)

# ==========================================
# 5. UI LAYOUT & VISUALIZATIONS
# ==========================================

st.title("AI Bubble Systemic Risk Monitor")

with st.expander("ℹ️ **How to Read This Dashboard (Click to Expand)**", expanded=True):
    st.markdown("""
    Welcome to the **Systemic Risk Monitor**. This tool tracks the financial stability of the AI industry using synthetic data modeled after the 2008 financial crisis.
    
    **Three Things to Watch:**
    1.  **The Gauge (Below):** If it's **Red (>60)**, the market is overheated and fragile.
    2.  **The Heatmaps (Tab 2):** Look for **Red Cells**. These are companies burning cash too fast or heavily in debt.
    3.  **The Simulator (Sidebar):** Use the sidebar to test "What If" scenarios (e.g., "What if a major AI lab goes bankrupt?").
    """)

st.markdown(f"**Status:** {st.session_state.simulation_text} | **Last Updated:** {st.session_state.last_updated.strftime('%H:%M:%S')}")

# Top Level Narrative
st.info(f"""
**Executive Summary:** The Systemic Risk Score is currently **{systemic_risk_score}**. 
Conditions represent a **{risk_text}** environment. 
**Executive Summary:** The Systemic Risk Score is currently **{systemic_risk_score}**. 
Conditions represent a **{risk_text}** environment. 
This dashboard uses **Hybrid Data**: Real-time stock metrics for public companies (via Yahoo Finance) and synthetic models for private labs/SPVs.
""")

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview & Risk Gauge", 
    "AI Ecosystem Heatmap", 
    "Interdependency Network", 
    "2008 Comparison", 
    "User Guide & Glossary"
])

# --- TAB 1: OVERVIEW ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = systemic_risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Systemic Risk Index (0-100)"},
            delta = {'reference': 50, 'increasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "rgba(255, 255, 255, 0.3)"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "#21c354"},
                    {'range': [30, 60], 'color': "#ffa421"},
                    {'range': [60, 100], 'color': "#ff4b4b"}],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': systemic_risk_score}}))
        fig_gauge.update_layout(height=400, margin=dict(l=20,r=20,t=50,b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("Key Risk Drivers")
        # Top 3 most distressed entities
        top_risk_labs = st.session_state.ai_data.sort_values("Prob. of Distress", ascending=False).head(3)
        
        st.write("**Top 3 Most Fragile Labs (Simulated):**")
        for index, row in top_risk_labs.iterrows():
            st.markdown(f"""
            <div class="metric-box" title="Distress Probability: Likelihood of default in next 12 months. Burn Rate: Cash spent per year.">
                <b>{row['Lab Name']}</b> <br>
                Distress Probability: <span style='color:{"red" if row['Prob. of Distress']>0.25 else "orange"}'>{row['Prob. of Distress']:.0%}</span> | 
                Burn Rate: ${row['Burn Rate ($B/yr)']}B/yr
            </div>
            """, unsafe_allow_html=True)
            
        st.write(f"**Current Simulation Offset:** +{st.session_state.simulation_offset:.1f} points (Due to scenario: {scenario})")

# --- TAB 2: DEEP DIVE ---
with tab2:
    st.subheader("AI Lab & Hyperscaler Fragility")
    st.info("💡 **How to read:** This heatmap shows where the risk is concentrated. **Dark Red** means high danger (e.g., a company with high debt AND high burn rate).")
    st.caption("This heatmap visualizes the 'Hot Zones' of the ecosystem. Red indicates high burn rates relative to revenue or high debt exposure.")
    
    # Prepare Heatmap Data
    hm_data = st.session_state.ai_data.copy()
    hm_data = hm_data.set_index("Lab Name")
    # Normalize for color scaling
    hm_view = hm_data[["Cloud Commitments ($B)", "Burn Rate ($B/yr)", "Prob. of Distress", "Systemic Contribution"]]
    
    fig_hm = px.imshow(
        hm_view, 
        text_auto=True, 
        aspect="auto", 
        color_continuous_scale="RdYlGn_r", # Red is high value (bad for risk)
        title="AI Lab Risk Metrics Heatmap"
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Hyperscaler Exposure (Real-Time)")
        st.markdown(st.session_state.hyper_data.style.background_gradient(cmap="Oranges", subset=["Sensitivity"]).to_html(), unsafe_allow_html=True)
    with col_b:
        st.markdown("### SPV / Debt Structures")
        st.markdown(st.session_state.spv_data.style.format({"Debt ($B)": "${:.1f}", "Default Prob": "{:.0%}"}).to_html(), unsafe_allow_html=True)

# --- TAB 3: NETWORK GRAPH ---
with tab3:
    st.subheader("Ecosystem Contagion Map")
    st.info("💡 **How to read:** This graph shows who owes money to whom. If a central node (like a Hyperscaler) fails, the shockwave travels along these lines to everyone connected.")
    st.caption("Visualizing dependencies. Nodes are Labs, Hyperscalers, or SPVs. Lines represent financial flow/dependency. Hover for details.")
    
    # Construct NetworkX Graph
    G = nx.Graph()
    
    # Add Nodes
    for i, row in st.session_state.ai_data.iterrows():
        G.add_node(row['Lab Name'], type='Lab', size=20, risk=row['Prob. of Distress'])
        
    for i, row in st.session_state.hyper_data.iterrows():
        G.add_node(row['Company'], type='Hyperscaler', size=35, risk=row['Sensitivity'])
        
    for i, row in st.session_state.spv_data.iterrows():
        G.add_node(row['SPV Name'], type='SPV', size=15, risk=row['Default Prob'])

    # Add Random Edges (Simulating relationships)
    # Labs connect to Hyperscalers (Cloud compute)
    labs = st.session_state.ai_data['Lab Name'].tolist()
    hypers = st.session_state.hyper_data['Company'].tolist()
    spvs = st.session_state.spv_data['SPV Name'].tolist()
    
    for lab in labs:
        # Connect lab to 1-2 hyperscalers
        target = random.choice(hypers)
        G.add_edge(lab, target, weight=0.8)
    
    for spv in spvs:
        # SPV connects to Hyperscaler (buying chips) and Lab (leasing chips)
        G.add_edge(spv, random.choice(hypers), weight=0.9)
        G.add_edge(spv, random.choice(labs), weight=0.7)

    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create Plotly Graph
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_info = f"Entity: {node}<br>Type: {G.nodes[node]['type']}<br>Risk Score: {G.nodes[node]['risk']:.2f}"
        node_text.append(node_info)
        # Color by risk
        risk_val = G.nodes[node]['risk']
        node_color.append(risk_val)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='RdYlGn_r', # Red is high risk
            size=20,
            color=node_color,
            colorbar=dict(title='Risk Exposure'),
            line_width=2))

    fig_net = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                    ))
    
    st.plotly_chart(fig_net, use_container_width=True)

# --- TAB 4: 2008 COMPARISON ---
with tab4:
    st.subheader("Historical Analog: 2008 Financial Crisis vs. AI Boom")
    
    col_l, col_r = st.columns([1, 1])
    
    analogy_df = fetch_2008_comparison_data()
    
    with col_l:
        st.markdown("#### Structural Similarities")
        fig_sim = px.bar(analogy_df, x="Similarity", y="2008 Component", orientation='h',
                         color="Similarity", color_continuous_scale="Reds",
                         hover_data=["Explanation", "AI Equivalent"])
        st.plotly_chart(fig_sim, use_container_width=True)
        
    with col_r:
        st.markdown("#### Analogy Mapping Table")
        st.markdown(analogy_df[["2008 Component", "AI Equivalent", "Explanation"]].to_html(index=False), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Why compare them?")
    st.markdown("""
    In 2008, the system broke because risk was hidden in complex structures (CDOs) and rating agencies assumed house prices would never fall. 
    In the AI boom, we see **'GPU SPVs'** (debt vehicles to buy chips) and **'Circular Revenue'** (Cloud Co invests in AI Lab -> AI Lab pays Cloud Co for hosting). 
    If AI revenue doesn't materialize to pay for the chips, the debt defaults—just like subprime mortgages.
    """)

# --- TAB 5: USER GUIDE ---
with tab5:
    st.markdown("# How to Use This Dashboard")
    
    st.markdown("""
    ### 1. The Risk Gauge (Overview)
    The big dial on the first page shows the aggregate risk (0-100). 
    * **0-30:** Safe. Healthy growth.
    * **30-60:** Caution. Leverage is building up.
    * **60-100:** Danger. The bubble is extremely fragile.
    
    ### 2. The Simulator (Sidebar)
    Use the sidebar to trigger "What If" scenarios.
    * **Frontier Lab Default:** What happens if a major player (like an OpenAI equivalent) runs out of cash?
    * **GPU Crash:** What happens if the chips used as collateral for loans drop 50% in value?
    
    ### 3. Glossary for Non-Experts
    * **Hyperscaler:** Massive tech companies (Google, Amazon, Microsoft) that provide the cloud computing power.
    * **SPV (Special Purpose Vehicle):** A separate company created just to hold assets (like GPUs) and take on debt, keeping that debt off the main company's books.
    * **Circular Revenue:** When Company A invests in Company B, and Company B uses that money to buy products from Company A. It looks like revenue, but it's just moving money in a circle.
    * **Burn Rate:** How much money an AI company spends per year to train models (usually much more than they make).
    """)
    
    st.markdown("---")
    st.subheader("Detailed Scenario Guide")
    st.markdown("""
    **1. Frontier Lab Default**
    *   **What it simulates:** A major AI company (like OpenAI or Anthropic) runs out of cash and cannot pay its cloud bills.
    *   **Impact:** Cloud providers lose revenue, investors panic, and the "Systemic Risk Score" spikes.
    
    **2. GPU Demand Crash**
    *   **What it simulates:** The resale value of H100 GPUs drops by 50%.
    *   **Impact:** SPVs (Special Purpose Vehicles) that used these GPUs as collateral for loans suddenly become insolvent. Banks stop lending.
    
    **3. 2008 Liquidity Freeze**
    *   **What it simulates:** A total loss of trust in the market. Banks stop lending to *everyone*.
    *   **Impact:** Maximum contagion. Even healthy companies fail because they can't get short-term loans.
    """)

# ==========================================
# 6. README / DEV INSTRUCTIONS (COMMENT)
# ==========================================
# """
# /*
# ---------------------------------------------------------
# DEVELOPER README
# ---------------------------------------------------------

# 1. DEPLOYMENT:
#    This app is ready to deploy on Streamlit Community Cloud.
#    Simply push this file to GitHub and connect it to Streamlit.

# 2. PLUGGING IN REAL DATA:
#    Locate the functions:
#    - fetch_live_ai_risk_data()
#    - fetch_live_hyperscaler_data()
   
#    Replace the random/synthetic logic with API calls to:
#    - Financial APIs (Yahoo Finance, Bloomberg) for stock data.
#    - Web scraping or News APIs for 'Distress' sentiment analysis.
   
# 3. ARCHITECTURE:
#    - The 'st.session_state' is used to persist data between refreshes.
#    - The 'auto_refresh' block handles the "Live" simulation loop.
#    - NetworkX is used for the graph logic, Plotly for rendering.

# 4. INSTALLATION:
#    pip install streamlit pandas numpy plotly networkx

# 5. RUNNING:
#    streamlit run app.py
# */
# """
