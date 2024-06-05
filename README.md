# Cluster Analysis and Model Training

## Project Overview

This project involves cluster analysis and model training for predicting high-income developers based on the 2021 New Coder Survey dataset. The goal is to perform clustering to understand the characteristics of new coders and build machine learning models to predict whether a developer earns a high income.


## Project Details

### Clustering Analysis
- **Objective:** Understand the characteristics of new coders by grouping them into clusters.
- **Method:** K-Means clustering algorithm.
- **Outcome:** Identification of distinct groups among new coders based on survey responses.

### Classification Model
- **Objective:** Predict whether a developer earns a high income (>= $30,000) or low income (< $30,000).
- **Features:** Selected features from the survey data, including Q22 (income).
- **Method:** Random Forest Classifier.
- **Outcome:** A trained model that can classify developers into high-income or low-income categories.
  
## Project Structure

The project is structured as follow

new-coder-survey-analysis/
├── data
│ └── 2021 New Coder Survey.csv
├── notebooks
│ ├── 1_data_analysis.ipynb
│ └── 2_model_training.ipynb
├── README.md
└── requirements.txt

### Directories

- **data**: Contains the dataset used for analysis and model training.
  - `2021 New Coder Survey.csv`: The main dataset file.
- **notebooks**: Contains Jupyter notebooks for data analysis and model training.
  - `1_data_analysis.ipynb`: Notebook for initial data exploration and preprocessing.
  - `2_model_training.ipynb`: Notebook for training and evaluating the machine learning models.
- **README.md**: Project documentation.
- **requirements.txt**: List of Python dependencies required for the project.

## Setup Instructions

1. **Clone the repository**:

    ```bash
    git clone https://github.com/username/repository.git
    cd new-coder-survey-analysis
    ```

2. **Create and activate a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Place the dataset in the `data` directory**:

    Make sure the `2021 New Coder Survey.csv` file is placed in the `data` directory.

## Usage Guide

### Data Analysis

Open and run the `notebooks/1_data_analysis.ipynb` notebook to:

- Load and explore the dataset.
- Clean and preprocess the data.
- Perform exploratory data analysis (EDA).

### Model Training

Open and run the `notebooks/2_model_training.ipynb` notebook to:

- Load the preprocessed data.
- Split the data into training and testing sets.
- Train a classification model to predict whether a developer is in the high-income bracket.
- Evaluate the model's performance.
- Save the trained model.

## Example

Here is an example of how to use the notebooks:

1. **Run Data Analysis Notebook**:

    Open `notebooks/1_data_analysis.ipynb` in google Colab and run all cells to preprocess the data.

2. **Run Model Training Notebook**:

    Open `notebooks/2_model_training.ipynb` in google Colab and run all cells to train and evaluate the model.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions or suggestions, please contact me at [sauravsamal9@gmail.com].
