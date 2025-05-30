# 🛳 Titanic - Machine Learning from Disaster

This project predicts survival on the Titanic using machine learning. It is based on the famous Kaggle competition and includes data preprocessing, exploratory data analysis (EDA), model building, evaluation, and prediction.


### 📁 Dataset

The project uses the Titanic dataset provided by [Kaggle](https://www.kaggle.com/competitions/titanic/data):

* `train.csv` — training data used to build the model
* `test.csv` — test data used to make final predictions
* `submission.csv` — sample submission format

---

### 🔍 Project Steps

1. **Data Loading**

   * Load training and test datasets using pandas.

2. **Exploratory Data Analysis (EDA)**

   * Visualize survival rates by sex, class, age, embarked port, and fare.
   * Analyze missing values and data distribution.

3. **Data Cleaning**

   * Handle missing values (e.g., Age, Fare, Embarked).
   * Drop unnecessary columns (`Cabin`, `Ticket`, `Name`).

4. **Encoding Categorical Variables**

   * Use `LabelEncoder` or `pd.get_dummies()` to convert `Sex` and `Embarked`.

5. **Model Building**

   * Use `RandomForestClassifier` to train the model.
   * Evaluate using accuracy, confusion matrix, and classification report.

6. **Prediction**

   * Predict survival for the test dataset.
   * Save results as `submission.csv`.

7. **Feature Importance**

   * Visualize important features affecting survival prediction.

---

### 🛠️ Tech Stack

* Python
* Jupyter Notebook / VS Code
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

---

### 📊 Visuals

* Count plots for survival analysis
* Age & Fare distributions
* Correlation heatmaps
* Feature importance bar chart

---

### 📁 Project Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
├── titanic-analysis.ipynb
├── submission.csv
└── README.md
```

---

### 💡 Insights

* Females had higher survival rates.
* First-class passengers had better chances of survival.
* Children and younger people survived more often.

---

### 🚀 How to Run

1. Clone the repo or download the `.ipynb` file.
2. Open in Jupyter Notebook or VS Code.
3. Make sure required libraries are installed.
4. Run the notebook cells step by step.

---

### 📌 Future Improvements

* Try advanced models like XGBoost or SVM.
* Perform hyperparameter tuning.
* Use cross-validation for better generalization.

---

### 📚 License

This project is for educational purposes and follows [Kaggle's competition rules](https://www.kaggle.com/competitions/titanic/rules).

