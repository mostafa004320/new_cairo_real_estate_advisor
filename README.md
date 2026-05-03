
# New Cairo Apartment Advisor

A practical real estate pricing project for **New Cairo apartments** using recent Property Finder Egypt listings.

The project supports two business cases:

1. **Sale Price Prediction**: estimate fair selling price for an apartment.
2. **Rent Price Prediction**: estimate fair monthly rent for an apartment.

## Who can use it?

- Real estate companies
- Brokers / agents
- Apartment owners who want to sell
- Apartment owners who want to rent
- Buyers or tenants checking whether a price is reasonable

## Data

Raw scraped file: Property Finder Egypt export.

Cleaned dataset summary:

- Clean sale apartments: **201** listings
- Clean rent apartments: **326** listings
- Median sale price per m²: **50,388 EGP**
- Median rent price per m²/month: **339 EGP**

## Model

The project trains two independent models:

- `sale_price_model.joblib`
- `rent_price_model.joblib`

Features used:

- Area
- Bedrooms
- Bathrooms
- Latitude / longitude
- Furnished flag
- District
- Payment plan

The target is log-transformed price to reduce the effect of extreme listings.

## Run the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Run the API backend

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Then open:

```text
frontend/index.html
```

The frontend will call:

```text
http://127.0.0.1:8000/predict
```

## Important note

This is a decision-support tool, not a certified real estate appraisal. Final pricing should consider exact building condition, floor, view, finishing, maintenance, urgency, negotiation margin, and fresh comparable listings.
