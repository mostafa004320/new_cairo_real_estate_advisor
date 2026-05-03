
import pandas as pd
import joblib
from pathlib import Path
import streamlit as st
import plotly.express as px

BASE = Path(__file__).resolve().parent
sale_bundle = joblib.load(BASE / "models" / "sale_price_model.joblib")
rent_bundle = joblib.load(BASE / "models" / "rent_price_model.joblib")
sale = pd.read_csv(BASE / "data" / "sale_apartments_clean.csv")
rent = pd.read_csv(BASE / "data" / "rent_apartments_clean.csv")
all_df = pd.concat([sale.assign(mode="Sale"), rent.assign(mode="Rent")], ignore_index=True)

st.set_page_config(page_title="New Cairo Apartment Advisor", page_icon="🏠", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #07111f 0%, #0f2438 45%, #101827 100%); color: #edf6ff;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);} 
.hero {padding: 2rem; border-radius: 28px; background: radial-gradient(circle at top left, rgba(38,198,218,.35), transparent 32%), linear-gradient(135deg, rgba(255,255,255,.10), rgba(255,255,255,.04)); border: 1px solid rgba(255,255,255,.15); box-shadow: 0 20px 70px rgba(0,0,0,.35);}
.card {padding: 1.2rem; border-radius: 22px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.15); transition: .25s;}
.card:hover {transform: translateY(-4px); box-shadow: 0 15px 40px rgba(38,198,218,.18);} 
.metric {font-size: 2rem; font-weight: 800;}
.small {color: #a8bfd4; font-size: .95rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>New Cairo Apartment Advisor</h1><p>Sale and rent pricing assistant powered by recent Property Finder Egypt listings.</p></div>', unsafe_allow_html=True)
st.write("")

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f'<div class="card"><div class="small">Sale listings</div><div class="metric">{len(sale):,}</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="card"><div class="small">Rent listings</div><div class="metric">{len(rent):,}</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="card"><div class="small">Median sale / m²</div><div class="metric">{sale.price_per_m2.median():,.0f}</div><div class="small">EGP</div></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="card"><div class="small">Median rent / m²</div><div class="metric">{rent.price_per_m2.median():,.0f}</div><div class="small">EGP / month</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([.38,.62])
with left:
    st.subheader("Estimate a fair price")
    offer = st.radio("Mode", ["sale", "rent"], horizontal=True)
    area = st.number_input("Area (m²)", 40, 450, 160)
    bedrooms = st.slider("Bedrooms", 0, 8, 3)
    bathrooms = st.slider("Bathrooms", 0, 8, 2)
    district = st.selectbox("District", sorted(all_df.district.dropna().unique().tolist()), index=sorted(all_df.district.dropna().unique().tolist()).index("The 5th Settlement") if "The 5th Settlement" in all_df.district.unique() else 0)
    furnished = st.toggle("Furnished", value=False)
    payment_plan = st.selectbox("Payment plan", sorted(all_df.payment_plan.dropna().unique().tolist()))
    lat = st.number_input("Latitude", value=30.04, format="%.6f")
    lng = st.number_input("Longitude", value=31.47, format="%.6f")
    if st.button("Predict", type="primary", use_container_width=True):
        bundle = sale_bundle if offer == "sale" else rent_bundle
        X = pd.DataFrame([{"area_m2": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "latitude": lat, "longitude": lng, "furnished": int(furnished), "district": district, "payment_plan": payment_plan}])
        predicted_value = float(bundle["model"].predict(X)[0])
        pred = predicted_value * area if bundle.get("target") == "price_per_m2" else predicted_value
        st.session_state["prediction"] = (offer, pred, area)
with right:
    st.subheader("Result")
    if "prediction" not in st.session_state:
        st.info("Enter apartment details, then click Predict.")
    else:
        offer, pred, area = st.session_state["prediction"]
        unit = "EGP sale price" if offer == "sale" else "EGP monthly rent"
        st.markdown(f'<div class="card"><div class="small">Estimated {unit}</div><div class="metric">{pred:,.0f}</div><div class="small">Fair range: {pred*0.90:,.0f} - {pred*1.10:,.0f} EGP</div><div class="small">Estimated price / m²: {pred/area:,.0f} EGP</div></div>', unsafe_allow_html=True)
        st.caption("This is a decision-support estimate. Always validate with property condition, floor, exact compound, and live comparable listings.")

st.write("")
st.subheader("Market overview")
c1, c2 = st.columns(2)
fig1 = px.histogram(sale, x="price_per_m2", nbins=30, title="Sale price per m² distribution")
fig2 = px.histogram(rent, x="price_per_m2", nbins=30, title="Rent price per m² distribution")
c1.plotly_chart(fig1, use_container_width=True)
c2.plotly_chart(fig2, use_container_width=True)

fig3 = px.box(all_df, x="mode", y="price_per_m2", color="mode", points="outliers", title="Sale vs Rent price per m²")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Latest cleaned data sample")
st.dataframe(all_df[["offer_type","price","area_m2","bedrooms","bathrooms","district","furnished","price_per_m2","listed_at"]].head(50), use_container_width=True)
