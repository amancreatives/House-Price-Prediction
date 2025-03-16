# House-Price-Prediction
# 🏠 House Price Prediction using XGBoost

This project predicts the house prices based on various features using XGBoost Regression. It's a supervised ML regression problem with features like crime rate, rooms, property tax, etc.

---

## 📁 Dataset

- 15 columns in total
- First column: Serial number (removed during preprocessing)
- Last column: PRICE (target)
- Remaining 13: Feature columns

---

## 🧪 Technologies Used

- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

---

## ⚙️ Workflow

1. Load dataset with Pandas
2. Preprocess:
   - Drop serial column
   - Handle missing values
   - Standardize using StandardScaler
3. Split into train-test (80-20)
4. Train model using XGBRegressor
5. Evaluate with:
   - R² Score
   - Mean Absolute Error
6. Predict new input using:

```python
inputData = (0.02, 0, 7, 0, 0.47, 6.0, 35, 4.0, 2, 250, 17, 400, 10)
