# 📉 AI Bubble Systemic Risk Monitor

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

> **A real-time, interactive dashboard that visualizes systemic risks in the global AI ecosystem and compares them to the 2008 financial crisis.**

---

## 📖 Overview

The **AI Bubble Systemic Risk Monitor** is a specialized tool designed to track, visualize, and simulate financial fragility within the booming Artificial Intelligence sector. 

By drawing structural analogies to the **2008 Financial Crisis** (e.g., comparing GPU-backed SPVs to CDOs), this dashboard provides a unique lens for investors, researchers, and enthusiasts to understand potential contagion risks.

### 🎯 Key Objectives
- **Monitor** real-time indicators of distress among Frontier AI Labs and Hyperscalers.
- **Visualize** the complex web of financial interdependencies (the "Contagion Map").
- **Simulate** "Black Swan" events like a major lab default or a GPU market crash.
- **Educate** users on complex financial engineering concepts using plain language.

---

## ✨ Features

### 1. 🌍 Systemic Risk Gauge
A master "Doomsday Clock" for the AI industry. It aggregates data from multiple sources to provide a single 0-100 risk score.
- **Green (0-30):** Stable / Healthy Growth.
- **Yellow (30-60):** Elevated Risk / Leverage Buildup.
- **Red (60-100):** Critical / Bubble Burst Imminent.

### 2. 🔥 Risk Heatmaps
Identify the "Hot Zones" of the market instantly.
- **AI Lab Fragility:** Tracks Burn Rate vs. Revenue and Probability of Distress.
- **Hyperscaler Exposure:** Monitors over-investment in data centers and sensitivity to AI revenue shocks.

### 3. 🕸️ Interdependency Network
An interactive graph visualization powered by **NetworkX** and **Plotly**.
- See how **Labs**, **Cloud Providers**, and **SPVs** are connected.
- Visualize how a default in one node can cascade to others.

### 4. 🏚️ 2008 Crisis Comparison
A direct side-by-side analysis of 2008 vs. Today.
- **Subprime Mortgages** $\rightarrow$ **Over-optimistic AI Revenue**
- **CDOs** $\rightarrow$ **GPU-Backed SPVs**
- **Lehman Brothers** $\rightarrow$ **Frontier Lab Failure**

### 5. 💥 Scenario Simulator
Don't just watch—test the system.
- Run **"What If"** simulations (e.g., *Frontier Lab Default*, *Liquidity Freeze*).
- Adjust severity sliders and watch the Systemic Risk Score update in real-time.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python Package Manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-bubble-risk-monitor.git
   cd ai-bubble-risk-monitor
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy plotly networkx matplotlib
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the Dashboard**
   Open your browser and navigate to `http://localhost:8501`.

---

## 🛠️ Technology Stack

- **Frontend:** [Streamlit](https://streamlit.io/) - For rapid, interactive web app development.
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/).
- **Visualization:** [Plotly](https://plotly.com/) - For beautiful, interactive charts and gauges.
- **Graph Theory:** [NetworkX](https://networkx.org/) - For modeling financial interdependencies.

---

## 🤝 Contributing

Contributions are welcome! Whether it's fixing a bug, adding a new data source, or improving the 2008 analogies.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## ⚠️ Disclaimer

**This tool is for educational and illustrative purposes only.** 
The data generated in the default configuration is **synthetic** and intended to demonstrate the *model's logic*. It should not be used as financial advice or for making investment decisions.

---

<p align="center">
  Built with ❤️ by [Your Name/Organization]
</p>
