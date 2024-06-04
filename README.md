# New Coder Survey Analysis and Classification

## Project Overview

This project aims to analyze data from the 2021 New Coder Survey and build machine learning models to predict whether a developer is in a high-income bracket based on their information. For new coders in this dataset, low income is defined as less than $30,000, while high income is $30,000 or more.

## Project Structure

The project is structured as follows:
new-coder-survey-analysis/
├── data
│ └── 2021 New Coder Survey.csv
├── notebooks
│ ├── 1_data_analysis.ipynb
│ └── 2_model_training.ipynb
├── README.md
└── requirements.txt

- `data/`: Contains the dataset used for analysis.
- `notebooks/`: Jupyter notebooks for data analysis, visualization, and model training.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies required to run the project.

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

    Open `notebooks/1_data_analysis.ipynb` in Jupyter Notebook or Jupyter Lab and run all cells to preprocess the data.

2. **Run Model Training Notebook**:

    Open `notebooks/2_model_training.ipynb` in Jupyter Notebook or Jupyter Lab and run all cells to train and evaluate the model.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions or suggestions, please contact me at [email@example.com].
