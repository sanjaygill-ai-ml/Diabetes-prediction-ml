# 🩺 Diabetes Prediction using Machine Learning

This project builds a **Machine Learning classification model** that predicts whether a person is **Diabetic (1)** or **Non-Diabetic (0)** based on medical health measurements.

It uses healthcare data and ML algorithms to support **early disease risk detection**.

---

## 📌 Problem Statement

Diabetes is a serious health condition that requires early diagnosis.  
The goal of this project is to train a model that can automatically classify patients as:

- 🔵 **0 → Non-Diabetic**
- 🔴 **1 → Diabetic**

This is a **Binary Classification** problem in supervised learning.

---

## 📊 Dataset Features

| Feature | Description |
|--------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic diabetes influence |
| Age | Age of the patient |

---

## 🧠 Machine Learning Approach

| Step | Description |
|------|-------------|
| Data Loading | Dataset loaded using Pandas |
| Data Cleaning | Checked missing & zero values |
| Feature Scaling | Standardization applied |
| Data Split | Training & Testing using `train_test_split` |
| Model Training | ML models trained on dataset |
| Evaluation | Accuracy & performance metrics |
| Prediction | Custom diabetes prediction system |

---

## 🤖 Algorithms Used

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  

---

## 📈 Model Performance

The trained models achieve strong accuracy and can effectively predict diabetes risk using medical input features.

---

## 🛠 Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib / Seaborn**
- **Scikit-learn**
- **Jupyter Notebook / Google Colab**

---

## 💾 Saved Model

The trained model can be saved for reuse:

```
diabetes_model.pkl
```

---

## ▶ How to Run This Project

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the notebook

Open and run:

```
Diabetes_Prediction_using_ML.ipynb
```

---

## 🎯 Skills Demonstrated

✔ Supervised Machine Learning  
✔ Healthcare Data Analysis  
✔ Data Preprocessing  
✔ Feature Scaling  
✔ Classification Modeling  
✔ Model Evaluation  
✔ End-to-End ML Workflow  

---

## 🚀 Future Improvements

- Hyperparameter tuning  
- Deep Learning comparison  
- Build Web App (Flask / Streamlit)  
- Deploy model online  

---

## 👨‍💻 Author

**Sanjay Gill**  
Machine Learning & Data Science Enthusiast
