# ============================================================
# app.py — Flask Backend for Vehicle Price Predictor Pro
# ============================================================

import datetime
from flask import Flask, render_template, request
import sys
import os

# Add the project root to Python path so we can import from /src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predictor import VehiclePredictor

# --- Create the Flask app ---
app = Flask(__name__)

# --- Load the trained ML model ---
# Initializing here to avoid any race conditions during first request
try:
    predictor = VehiclePredictor()
    print("Predictor loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load predictor: {e}")
    predictor = None

@app.route("/", methods=["GET"])
def home():
    """Renders the blank input form on the home page."""
    return render_template(
        "index.html",
        car_name="",
        present_price="",
        year="",
        km_driven=0,
        fuel="",
        transmission="",
        owner="",
        prediction=None,
        category=None,
        depreciation=None,
        error=None,
        errors=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    if not predictor:
        return render_template("index.html", error="Model not loaded. Please run main.py first.")

    try:
        # STEP 1: Read inputs
        car_name = request.form.get("car_name", "").strip()
        present_price_lakhs_str = request.form.get("present_price", "").strip()
        year_str = request.form.get("year", "").strip()
        km_driven_str = request.form.get("km_driven", "").strip()
        fuel = request.form.get("fuel", "").strip()
        transmission = request.form.get("transmission", "").strip()
        owner = request.form.get("owner", "").strip()

        # STEP 2: Validation
        errors = []
        if not car_name: errors.append("Car Model / Name is required.")
        
        try:
            present_price_lakhs = float(present_price_lakhs_str)
            if present_price_lakhs <= 0: errors.append("Price must be > 0.")
        except:
            errors.append("Valid Present Price is required.")
            present_price_lakhs = 0

        try:
            year = int(year_str)
            current_year = datetime.datetime.now().year
            if year < 1990 or year > current_year: errors.append(f"Year must be between 1990 and {current_year}.")
        except:
            errors.append("Valid Manufacture Year is required.")
            year = 2020

        try:
            km_driven = float(km_driven_str)
            if km_driven < 0: errors.append("Kilometers cannot be negative.")
        except:
            errors.append("Valid Kilometers Driven is required.")
            km_driven = 0

        if not fuel or not transmission or not owner:
            errors.append("Please select Fuel, Transmission, and Owner status.")

        if errors:
            return render_template(
                "index.html",
                errors=errors,
                car_name=car_name,
                present_price=present_price_lakhs_str,
                year=year_str,
                km_driven=km_driven_str,
                fuel=fuel,
                transmission=transmission,
                owner=owner,
                prediction=None,
                category=None,
                depreciation=None,
                error=None
            )

        # STEP 3: Process and Predict
        brand = car_name.split()[0].lower()
        present_price_rupees = present_price_lakhs * 100_000

        predicted_val, category = predictor.predict(
            present_price=present_price_rupees,
            brand=brand,
            year=year,
            km_driven=km_driven,
            fuel=fuel,
            transmission=transmission,
            owner=owner
        )

        depreciation = predictor.calculate_depreciation(present_price_rupees, predicted_val)

        # STEP 4: Render Success
        return render_template(
            "index.html",
            prediction=f"{predicted_val:,.2f}",
            category=category,
            depreciation=depreciation,
            present_price_display=f"{present_price_rupees:,.0f}",
            car_name=car_name,
            present_price=present_price_lakhs,
            year=year,
            km_driven=km_driven,
            fuel=fuel,
            transmission=transmission,
            owner=owner,
            errors=None,
            error=None
        )

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    # Disable debug mode for production-like test
    app.run(host="127.0.0.1", port=5000, debug=False)