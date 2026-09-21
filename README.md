# Dashboard de suivi des chaînes de valeur agro-industrielles

Dashboard interactif suivant les indicateurs clés des filières agro-industrielles (céréales, élevage, oléagineux, fruits et légumes, pêche/aquaculture) du Sénégal, comparés à la région ouest-africaine — en lien avec le modèle de développement des **Agropoles**.

**Auteur :** Cheikh SOW — Data Scientist & Développeur Full-Stack, Informaticien au PNDAS/AGROPOLES

## Stack technique
- Python
- Streamlit (interface interactive)
- Plotly (graphiques, radar, classements)
- Requests (appel API Banque mondiale)
- Pandas (traitement des données)

## Source de données

[World Bank Open Data](https://data.worldbank.org) — API publique, gratuite, sans clé requise. Indicateurs utilisés :

- Valeur ajoutée agriculture (% du PIB)
- Croissance de la valeur ajoutée agricole
- Indice de production alimentaire
- Indice de production de cultures
- Indice de production animale (élevage)
- Rendement céréalier (kg/ha)
- Terres agricoles (% surface totale)
- Exportations agricoles (% exportations totales)

## Fonctionnalités
- Comparaison Sénégal vs pays voisins (Mali, Côte d'Ivoire, Ghana, Burkina Faso, Nigeria, Mauritanie)
- KPIs : valeur actuelle, rang régional, moyenne régionale, pays en tête
- Évolution temporelle par indicateur (multi-pays)
- Classement régional pour la dernière année disponible
- Vue radar multi-indicateurs : Sénégal vs moyenne régionale
- Table de données brutes exportable

##Installation et lancement en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre à `http://localhost:8501`.


