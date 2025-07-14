## Hands-on-ml-ch3-classification


This repository contains 4 hands-on machine learning projects completed as part of my internship at **ARCH Technologies**. Each project was implemented using **Python, Scikit-learn, and Google Colab** with step-by-step explanations, visualizations, and real-world datasets.

---

##  Projects Included

###  MNIST Digit Classification (KNN + GridSearchCV)
- Achieved over 97% test accuracy.
- Tuned hyperparameters using GridSearchCV.
- Visualized predictions.

###  Data Augmentation on MNIST
- Expanded training data by shifting images in 4 directions.
- Improved generalization.
- Accuracy increased to ~97.6%.

###  Titanic Survival Prediction (Tabular Data)
- Used pipelines to preprocess numerical and categorical features.
- Trained a `RandomForestClassifier` with cross-validation.
- Achieved >80% average accuracy.

###  Spam Email Classifier (Text/NLP)
- Parsed raw email files (ham & spam).
- Cleaned text, extracted features with CountVectorizer.
- Trained a logistic regression classifier.
- Achieved 98% precision and 95%+ recall.

---

##  Tech Stack

- **Python 3.10+**
- **Google Colab / Jupyter**
- **Scikit-learn**
- **NumPy, Pandas, Matplotlib, Seaborn**
- **SciPy, Regex, Email Parser**

---

##  Folder Structure

```bash
.
├── mnist_knn_classifier/
│   └── mnist_knn.ipynb
├── mnist_data_augmentation/
│   └── mnist_augment.ipynb
├── titanic_survival_pipeline/
│   └── titanic_pipeline.ipynb
├── spam_email_classifier/
│   └── spam_classifier.ipynb
├── visualizations/
│   └── *.png (saved plots if any)
└── README.md
