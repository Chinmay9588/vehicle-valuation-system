# ============================================================
# predictor.py — ML Model Wrapper for Vehicle Price Prediction
# ============================================================
# This module loads the trained RandomForest model and encoders,
# then uses them to predict the resale price of a used car.
# ============================================================

import pickle
import os
import datetime


class VehiclePredictor:
    """
    Wraps the trained ML model and provides easy prediction methods.
    
    Usage:
        predictor = VehiclePredictor()
        price, category = predictor.predict(present_price, brand, year, ...)
        depreciation = predictor.calculate_depreciation(present_price, price)
    """

    def __init__(self):
        """
        Loads the trained model and label encoders from disk.
        These files are created by running main.py (the training pipeline).
        """
        # Build absolute paths relative to this file's location
        base_dir      = os.path.dirname(__file__)
        model_path    = os.path.join(base_dir, '../models/vehicle_price_model.pkl')
        encoders_path = os.path.join(base_dir, '../models/encoders.pkl')

        # Load the trained RandomForestRegressor model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load the LabelEncoders for categorical columns
        # (fuel, transmission, owner, Brand)
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)

    # ----------------------------------------------------------
    def predict(self, present_price, brand, year, km_driven, fuel, transmission, owner):
        """
        Predicts the resale price of a used car.

        How it works:
          1. Calculates Car Age from the manufacture year
          2. Encodes categorical inputs (fuel, transmission, owner, brand)
             using the same LabelEncoders used during training
          3. Builds the feature array in the exact same order as training
          4. Passes the features to the ML model for prediction
          5. Assigns a price category based on the predicted value

        Args:
            present_price  (float): Ex-showroom price of the car in rupees
                                    e.g. 7.5 lakhs → pass 750000
            brand          (str):   Car brand, e.g. "maruti", "honda"
            year           (int):   Manufacture year, e.g. 2019
            km_driven      (float): Total kilometers driven, e.g. 35000
            fuel           (str):   Fuel type — "petrol", "diesel", "cng", etc.
            transmission   (str):   "manual" or "automatic"
            owner          (str):   "first", "second", "third", etc.

        Returns:
            tuple: (predicted_price_rupees, category_string)
                   e.g. (300364.30, "Economy")
        """

        # --- Calculate Car Age ---
        # Car Age = Current Year − Manufacture Year
        # e.g. if manufactured in 2019 and current year is 2026 → age = 7
        current_year = datetime.datetime.now().year
        car_age = current_year - year

        # --- Encode Categorical Variables ---
        # The model was trained on integer-encoded categories, not raw strings.
        # We use the saved LabelEncoders to convert strings → integers.
        def encode_safe(column_name, value):
            """
            Safely encodes a categorical value using the saved LabelEncoder.
            Returns 0 if the value was unseen during training (avoids crashes).
            """
            encoder = self.encoders.get(column_name)
            if not encoder:
                return 0  # Column encoder not found — use default
            try:
                # Convert to lowercase string to match training format
                return encoder.transform([str(value).lower()])[0]
            except ValueError:
                # Value was not seen during training (e.g. a rare fuel type)
                return 0

        brand_enc = encode_safe('Brand', brand)
        fuel_enc  = encode_safe('fuel', fuel)
        trans_enc = encode_safe('transmission', transmission)
        owner_enc = encode_safe('owner', owner)

        # --- Build Feature Array ---
        # IMPORTANT: The order here must exactly match the order used during
        # training (defined in main.py → final_feature_order).
        # Order: present_price, Brand, km_driven, fuel, transmission, owner, Car_Age
        features = [[
            present_price,   # Ex-showroom price in rupees
            brand_enc,       # Encoded brand (e.g. "maruti" → 12)
            km_driven,       # Total kilometers driven
            fuel_enc,        # Encoded fuel type (e.g. "petrol" → 3)
            trans_enc,       # Encoded transmission (e.g. "manual" → 1)
            owner_enc,       # Encoded owner history (e.g. "first" → 0)
            car_age          # Age of the car in years
        ]]

        # --- Get Prediction from Model ---
        predicted_price = self.model.predict(features)[0]

        # --- Assign Price Category ---
        # Based on the predicted resale value:
        #   < 5 Lakh  → Economy   (budget segment)
        #   5–10 Lakh → Mid-range (mid segment)
        #   > 10 Lakh → Luxury    (premium segment)
        if predicted_price < 500_000:
            category = "Economy"
        elif 500_000 <= predicted_price <= 1_000_000:
            category = "Mid-range"
        else:
            category = "Luxury"

        return predicted_price, category

    # ----------------------------------------------------------
    def calculate_depreciation(self, present_price, predicted_price):
        """
        Calculates how much value the car has lost since purchase.

        Formula:
            depreciation % = ((showroom_price - resale_price) / showroom_price) × 100

        Example:
            Showroom: ₹7,50,000  |  Resale: ₹3,00,000
            → Depreciation = ((7,50,000 - 3,00,000) / 7,50,000) × 100 = 60%

        Args:
            present_price  (float): Original ex-showroom price in rupees
            predicted_price (float): AI-predicted resale price in rupees

        Returns:
            float: Depreciation as a percentage (rounded to 1 decimal)
                   Returns None if present_price is zero or invalid.
        """
        if present_price and present_price > 0:
            depreciation = ((present_price - predicted_price) / present_price) * 100
            return round(depreciation, 1)
        return None

    # ----------------------------------------------------------
    def evaluate_price(self, predicted_price, asking_price):
        """
        Compares a seller's asking price against the predicted fair value.

        Logic (10% margin):
            asking < predicted × 0.9  → Underpriced (good deal for buyer)
            within ±10%               → Fair Price
            asking > predicted × 1.1  → Overpriced (seller wants too much)

        Args:
            predicted_price (float): Model's predicted resale price
            asking_price    (float): Seller's listed price (optional)

        Returns:
            str or None: "Underpriced", "Fair Price", "Overpriced", or None
        """
        if asking_price is None:
            return None

        lower_bound = predicted_price * 0.9
        upper_bound = predicted_price * 1.1

        if asking_price < lower_bound:
            return "Underpriced"
        elif lower_bound <= asking_price <= upper_bound:
            return "Fair Price"
        else:
            return "Overpriced"
