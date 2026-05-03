
from pathlib import Path
from typing import Literal, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parents[1]
SALE = joblib.load(BASE / "models" / "sale_price_model.joblib")
RENT = joblib.load(BASE / "models" / "rent_price_model.joblib")
SALE_DATA = pd.read_csv(BASE / "data" / "sale_apartments_clean.csv")
RENT_DATA = pd.read_csv(BASE / "data" / "rent_apartments_clean.csv")

app = FastAPI(title="New Cairo Apartment Advisor API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PredictionRequest(BaseModel):
    offer_type: Literal["sale", "rent"] = Field(..., description="sale or rent")
    area_m2: float = Field(..., gt=20, lt=600)
    bedrooms: int = Field(..., ge=0, le=10)
    bathrooms: int = Field(..., ge=0, le=10)
    district: str = "The 5th Settlement"
    payment_plan: str = "Unknown"
    furnished: bool = False
    latitude: Optional[float] = 30.04
    longitude: Optional[float] = 31.47

@app.get("/")
def root():
    return {"status": "ok", "service": "New Cairo Apartment Advisor API"}

@app.get("/market-summary")
def market_summary():
    def pack(df):
        return {
            "rows": int(len(df)),
            "median_price": float(df["price"].median()),
            "median_price_per_m2": float(df["price_per_m2"].median()),
            "min_price": float(df["price"].min()),
            "max_price": float(df["price"].max()),
        }
    return {"sale": pack(SALE_DATA), "rent": pack(RENT_DATA)}

@app.post("/predict")
def predict(payload: PredictionRequest):
    bundle = SALE if payload.offer_type == "sale" else RENT
    df = pd.DataFrame([{
        "area_m2": payload.area_m2,
        "bedrooms": payload.bedrooms,
        "bathrooms": payload.bathrooms,
        "latitude": payload.latitude or 30.04,
        "longitude": payload.longitude or 31.47,
        "furnished": int(payload.furnished),
        "district": payload.district,
        "payment_plan": payload.payment_plan,
    }])
    predicted_value = float(bundle["model"].predict(df)[0])
    # Current models predict price_per_m2, then we multiply by area for the final estimate.
    if bundle.get("target") == "price_per_m2":
        per_m2 = predicted_value
        price = predicted_value * payload.area_m2
    else:
        price = predicted_value
        per_m2 = price / payload.area_m2
    low, high = price * 0.90, price * 1.10
    label = "estimated_sale_price_egp" if payload.offer_type == "sale" else "estimated_monthly_rent_egp"
    return {
        "offer_type": payload.offer_type,
        label: round(price, 2),
        "fair_range_egp": {"low": round(low, 2), "high": round(high, 2)},
        "estimated_price_per_m2": round(per_m2, 2),
        "confidence_note": "Use as a decision-support estimate, not as a final appraisal. Verify with recent comparable listings and property condition."
    }
