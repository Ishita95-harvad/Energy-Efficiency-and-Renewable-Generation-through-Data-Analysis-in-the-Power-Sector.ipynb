# Energy-Efficiency-and-Renewable-Generation-through-Data-Analysis-in-the-Power-Sector.ipynb
Energy Efficiency and Renewable Generation through Data Analysis in the Power Sector.ipynb
energy-efficiency-renewables-ai/
│
├── data/
│   ├── raw/                     # Raw sensor, grid, weather data
│   ├── processed/               # Cleaned and merged data
│
├── agents/
│   ├── load_forecast_agent.py
│   ├── generation_optimizer.py
│   ├── efficiency_auditor.py
│   ├── renewable_predictor.py
│   └── visualization_agent.py
│
├── models/
│   ├── forecasting_model.pkl
│   ├── optimization_model.py
│   └── anomaly_detector.py
│
├── dashboards/
│   ├── streamlit_app.py
│   └── dashboard_components/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Forecasting_Models.ipynb
│   └── Optimization_Simulation.ipynb
│
├── api/
│   └── agent_api.py             # Flask/FastAPI to serve agent outputs
│
├── utils/
│   ├── data_loader.py
│   └── kpi_calculator.py
│
├── tests/
│   └── unit_tests.py
│
├── .env                         # API keys, DB connection
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE
