# Project Requirements: Booking Completion Prediction

## Objective
Build a predictive model that estimates whether a customer will complete a booking (`booking_complete`) using behavioral and trip context data. The model should support decision-making for conversion optimization (e.g., targeting promotions, follow-up campaigns, and perks).

## Deliverables
1. Cleaned modeling dataset with engineered features
2. EDA identifying class imbalance and key patterns
3. Statistical testing to validate feature relationships with the target
4. Trained models (baseline + tuned) with documented evaluation metrics
5. Model interpretability using SHAP to identify key drivers
6. Saved model artifact (`xgb_booking_model.pkl`) for reuse/deployment

## Key Questions to Answer
- What is the baseline conversion rate and how severe is class imbalance?
- Which features are statistically associated with booking completion?
- What encoding strategy handles high-cardinality categoricals without exploding dimensionality?
- Which model best balances precision and recall for conversion targeting?
- What are the top drivers of conversion (global + local explanations)?

## Success Criteria
- Reproducible training process
- Clear evaluation using precision, recall, F1
- Interpretable drivers (SHAP) that can inform business actions
