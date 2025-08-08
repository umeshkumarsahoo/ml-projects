
# 🏠 Housing Price Prediction Project

## 📌 Overview
This project builds a **machine learning model** to predict **median house values** using California housing data.
It follows a complete **data science pipeline** — from loading and exploring the data, to preprocessing, training multiple models, evaluating performance, and saving the final pipeline.

The project uses:
- **Scikit-learn** for modeling and preprocessing
- **Pandas** & **NumPy** for data handling
- **Matplotlib** / **Seaborn** for visualization

---

## 📂 Dataset
- **File:** `housing.csv`
- **Source:** [California Housing Prices Dataset](https://github.com/ageron/handson-ml2)
- **Target Variable (Label):** `median_house_value`
- **Features:** e.g., `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `median_income`, etc.

---

## 🔄 Workflow
1. **Data Loading & Exploration**
   - Load dataset using Pandas
   - Display summary statistics
   - Visualize distributions and correlations

2. **Feature & Label Separation**
   - Features = All columns except `median_house_value`
   - Label = `median_house_value` (the value we want to predict)

3. **Data Preprocessing Pipeline**
   - Handle missing values with `SimpleImputer(strategy='mean')`
   - Scale numerical features with `StandardScaler()`
   - (Optional) Handle categorical features with `OneHotEncoder()`

4. **Model Training**
   - Train **Linear Regression** model
   - Train **Random Forest Regressor**
   - Use **cross-validation** to evaluate models

5. **Model Evaluation**
   - Evaluate using RMSE (Root Mean Squared Error)
   - Compare performance between models

6. **Model Saving**
   - Save final model & preprocessing pipeline using `joblib`

---

## 📊 Results (Example Output)
| Model                | RMSE (Cross-Validation) |
|----------------------|------------------------|
| Linear Regression    | ~68,000                |
| Random Forest        | ~49,000                |

*(Results may vary based on dataset split and parameters)*

---

## 📦 Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```
Install them with:
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
```

2. **Run the main script**
```bash
python main.py
```

3. **Make Predictions with Saved Model**
```python
import joblib
import pandas as pd

# Load model and pipeline
model = joblib.load("housing_model.pkl")
pipeline = joblib.load("housing_pipeline.pkl")

# Load new data
new_data = pd.read_csv("input.csv")

# Transform & Predict
prepared_data = pipeline.transform(new_data)
predictions = model.predict(prepared_data)
print(predictions)
```

---

## 🚀 Future Improvements
- Tune hyperparameters using **GridSearchCV** or **RandomizedSearchCV**
- Add more feature engineering (ratios, interactions)
- Try more advanced models like **Gradient Boosting** or **XGBoost**
- Deploy as a **web app** using **Streamlit** or **Flask**

---

## ✨ Author
**Your Name**
📧 Email: your.email@example.com
🔗 [LinkedIn](https://www.linkedin.com/in/umesh-kumar-sahoo-94626727a) | [GitHub](https://github.com/umeshkumarsahoo)

---

**📢 Note:** This project is for learning purposes and demonstrates a standard supervised regression workflow.
