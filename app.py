"""
Dashboard de suivi des chaînes de valeur agro-industrielles
--------------------------------------------------------------
Source de données : API ouverte de la Banque mondiale (World Bank Open Data)
Technologies : Python, Streamlit, Plotly, Requests, Pandas, Statsmodels

Contexte : suit les indicateurs clés des filières agro-industrielles
(céréales, élevage, production alimentaire, terres agricoles...) pour
le Sénégal et ses voisins régionaux, en lien avec le modèle des Agropoles.

Auteur : Cheikh SOW
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from statsmodels.tsa.holtwinters import Holt

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Chaînes de Valeur Agro-Industrielles",
    page_icon="🌾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Personnalisation des Couleurs & Design (CSS)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #F8F9FA;
    }
    h1, h2, h3 {
        color: #1E5631 !important;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #4A5568;
        font-weight: 600;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #1E5631;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] {
        background-color: #1E2923;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #F7FAFC !important;
    }
</style>
""", unsafe_allow_html=True)

# Palette de couleurs dédiée pour la comparaison des pays
COLOR_DISCRETE_MAP = {
    "Sénégal": "#008751",
    "Mali": "#E2A03F",
    "Côte d'Ivoire": "#E67E22",
    "Ghana": "#27AE60",
    "Burkina Faso": "#8E44AD",
    "Nigeria": "#2C3E50",
    "Mauritanie": "#D35400"
}

WB_API_URL = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"

INDICATORS = {
    "Valeur ajoutée agriculture (% du PIB)": "NV.AGR.TOTL.ZS",
    "Croissance valeur ajoutée agricole (% annuel)": "NV.AGR.TOTL.KD.ZG",
    "Indice de production alimentaire": "AG.PRD.FOOD.XD",
    "Indice de production de cultures": "AG.PRD.CROP.XD",
    "Indice de production animale (élevage)": "AG.PRD.LVSK.XD",
    "Rendement céréalier (kg/ha)": "AG.YLD.CREL.KG",
    "Terres agricoles (% surface totale)": "AG.LND.AGRI.ZS",
    "Exportations agricoles (% exportations totales)": "TX.VAL.AGRI.ZS.UN",
}

COUNTRIES = {
    "Sénégal": "SEN",
    "Mali": "MLI",
    "Côte d'Ivoire": "CIV",
    "Ghana": "GHA",
    "Burkina Faso": "BFA",
    "Nigeria": "NGA",
    "Mauritanie": "MRT",
}

FOCUS_COUNTRY = "Sénégal"

# ---------------------------------------------------------------------------
# Configuration de la session HTTP avec retry
# ---------------------------------------------------------------------------
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
session.mount("https://", HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------------------------
# Appel API (avec cache)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_indicator(indicator_code: str, country_codes: list[str], year_start: int, year_end: int) -> pd.DataFrame:
    """Récupère les données d'un indicateur pour une liste de pays depuis l'API Banque mondiale."""
    codes = ";".join(country_codes)
    url = WB_API_URL.format(countries=codes, indicator=indicator_code)
    params = {
        "format": "json",
        "per_page": 1000,
        "date": f"{year_start}:{year_end}",
    }

    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException:
        return pd.DataFrame()

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame()

    rows = []
    for item in payload[1]:
        rows.append({
            "country": item.get("country", {}).get("value"),
            "iso3": item.get("countryiso3code"),
            "year": int(item.get("date")),
            "value": item.get("value"),
        })

    df = pd.DataFrame(rows).dropna(subset=["value"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_indicators(country_codes: list[str], year_start: int, year_end: int) -> dict:
    """Récupère tous les indicateurs définis, pour usage dans la vue comparative multi-indicateurs."""
    data = {}
    for label, code in INDICATORS.items():
        data[label] = fetch_indicator(code, country_codes, year_start, year_end)
    return data


# ---------------------------------------------------------------------------
# 🔮 Prévision (Holt - lissage exponentiel avec tendance)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def forecast_series(years: list[int], values: list[float], horizon: int) -> pd.DataFrame:
    """
    Prévoit les `horizon` prochaines années à partir d'une série annuelle,
    avec la méthode de Holt (lissage exponentiel double, tendance additive).
    Retourne un DataFrame avec les bornes d'un intervalle de confiance approximatif (~80%).
    """
    series = pd.Series(values, index=years).sort_index()

    if len(series) < 4:
        return pd.DataFrame()

    model = Holt(series.values, initialization_method="estimated")
    fit = model.fit(optimized=True)

    forecast_values = fit.forecast(horizon)

    residuals = series.values - fit.fittedvalues
    resid_std = np.std(residuals) if len(residuals) > 1 else 0.0
    z_80 = 1.28  # ~80% d'intervalle de confiance

    future_years = list(range(series.index[-1] + 1, series.index[-1] + 1 + horizon))
    steps = np.arange(1, horizon + 1)
    margin = z_80 * resid_std * np.sqrt(steps)

    forecast_df = pd.DataFrame({
        "year": future_years,
        "value": forecast_values,
        "lower": forecast_values - margin,
        "upper": forecast_values + margin,
    })
    return forecast_df


# ---------------------------------------------------------------------------
# Sidebar - Filtres
# ---------------------------------------------------------------------------
st.sidebar.title("Filtres")

selected_indicator_label = st.sidebar.selectbox("Indicateur principal", list(INDICATORS.keys()))
selected_countries = st.sidebar.multiselect(
    "Pays à comparer",
    list(COUNTRIES.keys()),
    default=["Sénégal", "Mali", "Côte d'Ivoire", "Ghana"],
)
year_range = st.sidebar.slider("Période", 2000, 2023, (2010, 2023))

st.sidebar.markdown("---")
st.sidebar.caption("Source : [World Bank Open Data](https://data.worldbank.org)")

if not selected_countries:
    st.warning("Sélectionne au moins un pays dans la barre latérale.")
    st.stop()

country_codes = [COUNTRIES[c] for c in selected_countries]
indicator_code = INDICATORS[selected_indicator_label]

# ---------------------------------------------------------------------------
# Titre principal
# ---------------------------------------------------------------------------
st.title("Dashboard de suivi des chaînes de valeur agro-industrielles")
st.caption(
    "Suivi comparatif des indicateurs agro-industriels du Sénégal et de la région — "
    "en lien avec le modèle des Agropoles (céréales, élevage, oléagineux, fruits et légumes, pêche/aquaculture)"
)

with st.spinner("Chargement des données depuis la Banque mondiale..."):
    df = fetch_indicator(indicator_code, country_codes, year_range[0], year_range[1])

if df.empty:
    st.error("Impossible de récupérer les données depuis la Banque mondiale pour le moment. L'API est peut-être ralentie ou indisponible. Veuillez réessayer dans un instant ou réduire la période.")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
latest_year = df["year"].max()
latest_df = df[df["year"] == latest_year].sort_values("value", ascending=False).reset_index(drop=True)

senegal_row = latest_df[latest_df["country"] == FOCUS_COUNTRY]
senegal_value = senegal_row["value"].iloc[0] if not senegal_row.empty else None
senegal_rank = (latest_df.index[latest_df["country"] == FOCUS_COUNTRY][0] + 1) if not senegal_row.empty else None
regional_avg = latest_df["value"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Sénégal — {latest_year}", f"{senegal_value:,.1f}" if senegal_value is not None else "N/A")
col2.metric("Rang régional", f"{senegal_rank}/{len(latest_df)}" if senegal_rank else "N/A")
col3.metric("Moyenne régionale", f"{regional_avg:,.1f}")
col4.metric("Pays en tête", latest_df.iloc[0]["country"] if not latest_df.empty else "N/A")

st.markdown("---")

# ---------------------------------------------------------------------------
# Évolution temporelle + Classement
# ---------------------------------------------------------------------------
col_ts, col_rank = st.columns([1.3, 1])

with col_ts:
    st.subheader(f"Évolution — {selected_indicator_label}")
    fig_ts = px.line(
        df.sort_values("year"),
        x="year",
        y="value",
        color="country",
        markers=True,
        color_discrete_map=COLOR_DISCRETE_MAP
    )
    fig_ts.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title="Pays",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified"
    )
    fig_ts.update_xaxes(showgrid=False)
    fig_ts.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    st.plotly_chart(fig_ts, use_container_width=True)

with col_rank:
    st.subheader(f"Classement {latest_year}")

    latest_df["bar_color"] = latest_df["country"].apply(
        lambda x: "#008751" if x == FOCUS_COUNTRY else "#CBD5E0"
    )

    fig_rank = px.bar(
        latest_df,
        x="value",
        y="country",
        orientation="h",
        color="bar_color",
        color_discrete_map="identity"
    )
    fig_rank.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"categoryorder": "total ascending"}
    )
    fig_rank.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    st.plotly_chart(fig_rank, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# 🔮 Module de prévision (ML)
# ---------------------------------------------------------------------------
st.subheader(" Prévision — projection à moyen terme")
st.caption(
    "Projection basée sur la méthode de Holt (lissage exponentiel avec tendance), "
    "adaptée aux séries annuelles courtes. Zone ombrée = intervalle de confiance ~80%."
)

col_fc1, col_fc2 = st.columns([1, 3])
with col_fc1:
    forecast_country = st.selectbox(
        "Pays à projeter", selected_countries,
        index=selected_countries.index(FOCUS_COUNTRY) if FOCUS_COUNTRY in selected_countries else 0,
    )
    forecast_horizon = st.slider("Horizon (années)", 1, 10, 5)

country_series = df[df["country"] == forecast_country].sort_values("year")

if len(country_series) < 4:
    st.info(f"Pas assez de points de données pour {forecast_country} sur cette période afin de générer une prévision fiable (minimum 4 années).")
else:
    forecast_df = forecast_series(
        country_series["year"].tolist(),
        country_series["value"].tolist(),
        forecast_horizon,
    )

    if forecast_df.empty:
        st.info("Données insuffisantes pour générer une prévision.")
    else:
        fig_forecast = go.Figure()

        fig_forecast.add_trace(go.Scatter(
            x=country_series["year"], y=country_series["value"],
            mode="lines+markers", name="Historique",
            line=dict(color="#008751", width=2),
        ))

        fig_forecast.add_trace(go.Scatter(
            x=pd.concat([forecast_df["year"], forecast_df["year"][::-1]]),
            y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
            fill="toself", fillcolor="rgba(0, 135, 81, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Intervalle ~80%", showlegend=True,
        ))

        fig_forecast.add_trace(go.Scatter(
            x=forecast_df["year"], y=forecast_df["value"],
            mode="lines+markers", name="Prévision",
            line=dict(color="#E67E22", width=2, dash="dash"),
        ))

        fig_forecast.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            legend_title="",
        )
        fig_forecast.update_yaxes(showgrid=True, gridcolor="#E2E8F0", title=selected_indicator_label)
        fig_forecast.update_xaxes(showgrid=False, title="Année")
        st.plotly_chart(fig_forecast, use_container_width=True)

        last_actual = country_series["value"].iloc[-1]
        last_forecast = forecast_df["value"].iloc[-1]
        trend = "en hausse 📈" if last_forecast > last_actual else "en baisse 📉" if last_forecast < last_actual else "stable ➖"
        st.caption(
            f"Projection {forecast_country} en {int(forecast_df['year'].iloc[-1])} : "
            f"**{last_forecast:,.1f}** (tendance {trend} par rapport à {int(country_series['year'].iloc[-1])} : {last_actual:,.1f})"
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Vue multi-indicateurs pour le Sénégal (radar) - Optionnelle à la demande
# ---------------------------------------------------------------------------
st.subheader("Vue multi-indicateurs — Sénégal vs moyenne régionale")

show_radar = st.checkbox("Afficher la comparaison globale multi-indicateurs (Radar)", value=False)

if show_radar:
    st.caption("Comparaison normalisée (0 à 1) des principaux indicateurs agro-industriels, dernière année disponible par indicateur")

    with st.spinner("Chargement des 8 indicateurs... Cela peut prendre quelques secondes."):
        all_data = fetch_all_indicators(country_codes, year_range[0], year_range[1])

    radar_labels, senegal_scores, regional_scores = [], [], []

    for label, ind_df in all_data.items():
        if ind_df.empty:
            continue
        last_yr = ind_df["year"].max()
        snapshot = ind_df[ind_df["year"] == last_yr]
        if snapshot.empty or snapshot["value"].max() == snapshot["value"].min():
            continue

        vmin, vmax = snapshot["value"].min(), snapshot["value"].max()
        snapshot = snapshot.copy()
        snapshot["norm"] = (snapshot["value"] - vmin) / (vmax - vmin)

        sen_row = snapshot[snapshot["country"] == FOCUS_COUNTRY]
        if sen_row.empty:
            continue

        radar_labels.append(label)
        senegal_scores.append(sen_row["norm"].iloc[0])
        regional_scores.append(snapshot["norm"].mean())

    if radar_labels:
        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=senegal_scores,
            theta=radar_labels,
            fill="toself",
            name="Sénégal",
            fillcolor="rgba(0, 135, 81, 0.35)",
            line=dict(color="#008751", width=2)
        ))

        fig_radar.add_trace(go.Scatterpolar(
            r=regional_scores,
            theta=radar_labels,
            fill="toself",
            name="Moyenne régionale",
            fillcolor="rgba(160, 174, 192, 0.25)",
            line=dict(color="#718096", width=2, dash="dash")
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E2E8F0"),
                angularaxis=dict(gridcolor="#E2E8F0")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            margin=dict(l=30, r=30, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Pas assez de données pour construire la vue multi-indicateurs sur cette sélection.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Table de données brutes
# ---------------------------------------------------------------------------
st.subheader("Données détaillées")
st.dataframe(
    df.sort_values(["country", "year"], ascending=[True, False]),
    use_container_width=True,
    hide_index=True,
)

st.caption("Données fournies par la Banque mondiale (World Bank Open Data) | Dashboard réalisé par Cheikh SOW")