# SmartBiz AI 🤖

**"From Customer Data to Intelligent Business Decisions"**

> IEEE Project #35 — Machine Learning Domain  
> Topic: "From Data to Decisions: The Power of Machine Learning in Business Recommendations"

---

## Overview

SmartBiz AI is a full-stack Django web application powered by machine learning that analyzes customer characteristics and generates intelligent, data-driven business recommendations.

---

## Features

| Feature | Description |
|---|---|
| 📊 Dashboard | Live stats & 5 interactive Chart.js charts |
| 🤖 AI Prediction | ML-powered recommendation form |
| 👥 Customer Management | Add, view, search, filter, delete customers |
| 📜 Recommendation History | Full history with search & pagination |
| 📈 Analytics | 6 analytics charts from real database data |
| ⚙️ ML Model Info | Metrics, features, dataset, training history |
| ℹ️ About | Full project documentation |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 3. Train the ML model
```bash
python manage.py train_model
```

### 4. Seed demo data (optional)
```bash
python manage.py seed_demo_data
```

### 5. Start the server
```bash
python manage.py runserver
```

Open: http://127.0.0.1:8000

---

## ML Pipeline

```
Dataset (CSV)
    ↓
Data Loading & Validation
    ↓
Data Cleaning (null removal, range checks)
    ↓
Feature Engineering
    ↓
Label Encoding (purchase_frequency, preferred_category)
    ↓
Train/Test Split (80/20)
    ↓
Algorithm Evaluation:
  - Logistic Regression
  - Decision Tree
  - Random Forest ← usually best
  - Gradient Boosting
    ↓
Best Algorithm Auto-Selected (highest test accuracy)
    ↓
Model Saved (models/model.pkl)
    ↓
Prediction API
```

---

## Dataset

A synthetic **demo dataset** of 800 records is included in `dataset/customers.csv`.

**Columns:**
- `age` — Customer age (18–75)
- `income` — Annual income ($)
- `previous_purchases` — Number of past purchases
- `purchase_frequency` — Daily / Weekly / Monthly / Rarely
- `preferred_category` — Customer's preferred shopping category
- `recommended_category` — **Target variable** for ML training

> **To use a real dataset:** Replace `dataset/customers.csv` with your file (keep the same column names), then re-run `python manage.py train_model`.

---

## Project Structure

```
business recommendation/
├── SmartBizAI/             Django project
├── recommendation/         Main app (models, views, forms, admin)
│   └── management/commands/
│       ├── train_model.py
│       └── seed_demo_data.py
├── ml/
│   ├── pipeline.py         Training pipeline
│   └── predictor.py        Inference
├── templates/              HTML templates
├── static/css|js           Design system & charts
├── dataset/                CSV dataset
├── models/                 Saved .pkl files
└── requirements.txt
```

---

## Technology Stack

- **Backend:** Python 3.10, Django 4.2
- **ML:** scikit-learn, pandas, NumPy, joblib
- **Database:** SQLite
- **Frontend:** HTML5, CSS3 (dark SaaS), JavaScript, Chart.js
- **Icons:** Font Awesome 6

---

## Management Commands

```bash
# Train the ML model
python manage.py train_model

# Train with a custom dataset
python manage.py train_model --dataset path/to/your/data.csv

# Seed 20 demo customers + ML recommendations
python manage.py seed_demo_data

# Seed custom count
python manage.py seed_demo_data --count 50

# Clear existing demo data and re-seed
python manage.py seed_demo_data --clear
```
