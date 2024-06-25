# Heart Disease Prediction Project

## Abstract

Heart disease remains one of the leading causes of death worldwide. Early detection and accurate diagnosis are crucial in preventing severe outcomes. This project aims to develop a machine learning model to predict the presence of heart disease in patients based on various medical attributes and test results. By leveraging data analysis and machine learning techniques, the project seeks to improve predictive accuracy and provide insights into significant risk factors associated with heart disease.

## Overview

This project aims to predict the presence of heart disease in patients using machine learning algorithms. The dataset used contains various patient characteristics and test results which are utilized to build and evaluate predictive models.

## Project Structure

- `data/`: Directory containing the dataset files.
- `notebooks/`: Jupyter notebooks for data exploration, cleaning, preprocessing, and model building.
- `src/`: Source code for the project, including scripts for data preprocessing, feature engineering, model training, and evaluation.
- `results/`: Directory to store model evaluation results, visualizations, and reports.
- `README.md`: Project overview and instructions.
- `requirements.txt`: File listing the Python dependencies for the project.

## Dataset

The dataset used in this project includes the following features:

- `age`: Age of the patient.
- `sex`: Gender of the patient (Male/Female).
- `chest_pain_type`: Type of chest pain experienced by the patient.
- `resting_blood_pressure`: Resting blood pressure (in mm Hg).
- `cholestoral`: Serum cholesterol (in mg/dl).
- `fasting_blood_sugar`: Fasting blood sugar level (binary: greater than 120 mg/ml or lower than 120 mg/ml).
- `rest_ecg`: Resting electrocardiographic results.
- `Max_heart_rate`: Maximum heart rate achieved.
- `exercise_induced_angina`: Exercise-induced angina (Yes/No).
- `oldpeak`: ST depression induced by exercise relative to rest.
- `slope`: The slope of the peak exercise ST segment.
- `vessels_colored_by_flourosopy`: Number of major vessels colored by fluoroscopy.
- `thalassemia`: Type of thalassemia (Normal, Fixed Defect, Reversable Defect).
- `target`: Presence of heart disease (binary: 0 for no, 1 for yes).

## Getting Started

### Prerequisites

- Python 3.8 or above
- Google Colab
- Git

### Libraries

The following Python libraries are used in this project:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- shap

These libraries are listed in the `requirements.txt` file.

### Running the Project in Google Colab

1. **Open Google Colab**:
   - Navigate to [Google Colab](https://colab.research.google.com/).

2. **Open the Notebook**:
   - Click on **File** -> **Open notebook**.
   - Select the **GitHub** tab.
   - Enter the repository URL `https://github.com/chinnuba/prediction-of-heart-disease` and press Enter.
   - Choose the desired notebook from the list (`notebooks/Data_Exploration_and_Cleaning.ipynb`, `notebooks/Feature_Engineering_and_Preprocessing.ipynb`, `notebooks/Model_Building_and_Evaluation.ipynb`, or `notebooks/Model_Interpretation.ipynb`).

3. **Run the Notebook**:
   - Follow the instructions within each notebook to run the cells. The notebooks will contain code snippets to clone the repository and upload the dataset as needed.
   - For example, the first cells in each notebook might include the following code to set up the environment:

     ```python
     # Clone the repository
     !git clone https://github.com/chinnuba/prediction-of-heart-disease.git
     %cd prediction-of-heart-disease

     # Install dependencies
     !pip install -r requirements.txt
     ```

### Directory Structure

```
prediction-of-heart-disease/
├── data/
│   └── heart_disease_dataset.csv
├── notebooks/
│   ├── main.ipynb
│   ├── Data_Exploration_and_Cleaning.ipynb
│   ├── Feature_Engineering_and_Preprocessing.ipynb
│   ├── Model_Building_and_Evaluation.ipynb
│   └── Model_Interpretation.ipynb
├── src/
│   ├── __init__.py
│   ├── setup.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── preprocess.py
│   └── train.py
├── results/
│   ├── evaluation_metrics.txt
│   └── visualizations/
│       ├── feature_importance.png
│       └── model_performance.png
├── README.md
└── requirements.txt
```

### Results

The results of the model evaluation, including performance metrics and visualizations, will be stored in the `results/` directory.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue to discuss what you would like to change.

## License

This project is licensed under the CC0: Public Domain License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/ketangangal/heart-disease-dataset-uci).
