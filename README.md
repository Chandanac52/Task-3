# Task 3: Full Data Science Project with Flask API

This project demonstrates a complete ML workflow using the Iris dataset:
- Data collection and preprocessing
- Model training (Logistic Regression)
- Model deployment via Flask API

##  Folder Structure

```
project-task3/
├── train_model.py
├── requirements.txt
├── model/
│   └── iris_model.pkl
└── app/
    └── app.py
```

##  How to Run

1. **Train the Model**:
```bash
python train_model.py
```

2. **Run the API**:
```bash
cd app
python app.py
```

3. **Make POST request** to `/predict` with:
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

✅ Returns class: `"setosa"`, `"versicolor"`, or `"virginica"`
