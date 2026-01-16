# Reflectie – Kom Veilig Thuis Data Project

## Nieuwe skills en methodes
Tijdens dit project heb ik nieuwe data science skills ontwikkeld door het bouwen van een volledige Python pipeline. Ik heb geleerd hoe ik ruwe incidentdata kan opschonen, normaliseren en verrijken met contextuele factoren zoals tijd (nachtfactor) en recency. Daarnaast heb ik voor het eerst een eenvoudige maar interpreteerbare machine learning techniek (logistische regressie met scikit-learn) toegepast om aannames in mijn data te valideren.

## Experimenten en iteraties
In de eerste iteratie werkte ik alleen met incidentfrequentie per locatie. Dit bleek een vertekend beeld te geven, omdat oudere incidenten even zwaar mee wogen als recente. Daarom heb ik een recency-weighting toegepast, waardoor recente meldingen zwaarder meetellen.  
Daarna heb ik geëxperimenteerd met een nachtfactor, omdat incidenten ’s nachts een ander risicoprofiel hebben dan overdag. Tot slot heb ik een eenvoudige ML-validatie toegevoegd om te controleren of mijn handmatig bedachte risicoscore logisch correleert met de data.

## Wat ging goed
- Het opzetten van een duidelijke en reproduceerbare data pipeline.
- Het maken van een interpreerbare risicoscore in plaats van een ‘black box’ model.
- Het combineren van data-analyse met een maatschappelijk relevante context.

## Wat zou ik verbeteren
Als ik meer tijd had, zou ik meerdere databronnen combineren (bijvoorbeeld demografische of omgevingsdata) en uitgebreidere validatie toepassen, zoals cross-validatie. Daarnaast zou ik echte visuele output (zoals heatmaps) toevoegen om de inzichten nog beter te communiceren.

## Persoonlijke groei
Dit project heeft mij geholpen om van losse scripts naar een gestructureerde data science workflow te gaan. Ik heb geleerd om bewuster keuzes te maken in data preprocessing en modelselectie, en deze ook te onderbouwen. Hierdoor voel ik me zelfverzekerder in het toepassen van data science methodes binnen toekomstige projecten.
