# ⚽ Premier League xG Analysis (2024–25)

This project explores how **Expected Goals (xG)** shaped the 2024–25 Premier League season.  
It compares actual results with two xG-based adjustment methods to see which teams were clinical finishers, which lacked a killer instinct, and how their seasons might have looked different.

The project is accompanied by an interactive **Streamlit dashboard** where you can test scenarios (e.g. *What if Chelsea had taken their chances?*).

---

## 🔍 Features
- **xG Adjustment A (Floor Method):** Replaces goals with floored xG values.  
- **xG Adjustment B (Round Method):** Replaces goals with rounded xG values.  
- Recalculated league tables based on adjusted results.  
- Visualisations of team-level xG performance across the season.  
- Match-level breakdowns showing how extra goals could have changed outcomes.  
- Interactive dashboard built with **Streamlit**.  

---

## 📊 Example Insights
- **Nottingham Forest** maximised their finishing and could have qualified for Europe.  
- **Chelsea** created enough chances to be title contenders, but underperformed badly.  
- **Crystal Palace** scored 10 fewer goals than xG suggested, costing them a top-half finish.  

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/kalungianalytics/0008_xG.git
cd premier-league-xg-analysis
