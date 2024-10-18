# PV-Power-Generation-Forecast

## Table of Contents
- [Overview](#Overview)
- [To Do List](#To-Do-List)
- [Installation](#Installation)
- [Dataset](#Dataset)
- [Usage](#Usage)
- [Results](#Results)

## 🌞 Overview
This project focuses on predicting solar photovoltaic (PV) power generation based on regional microclimate data. The objective is to forecast the power output of PV devices installed in various terrains using environmental data such as temperature, humidity, wind speed, solar radiation, and rainfall.

## 📝 To Do List
- [ ] Discuss data partitioning schemes for testing our method
    > Daniel
- [ ] Develop a function to integrate external data into our dataset
    > Benson
- [ ] Rewrite the entire pipeline to comply with the competition format
    > Ken
- [ ] Reproduce the【LSTM+迴歸分析】from the SampleCode for each dataset and document the results
- [ ] Run sample code and write down the 【trained MSE】 for each data (1-17)
    > Raymin

## 💻 Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/Shengwei-Peng/PV-Power-Generation-Forecast.git
    ```
2. Navigate to the project directory:
    ```sh
    cd PV-Power-Generation-Forecast
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## 📊 Dataset
### 1. Folder Structure
The dataset consists of multiple CSV files stored in a folder, with each file representing data from a specific location. Each location is identified by a unique **LocationCode**.

```
TrainingData/
├── L1_Train.csv
├── L2_Train.csv
├── L3_Train.csv
├── L4_Train.csv
├── ...
└── L17_Train.csv
```

### 2. Raw Data
Each CSV file contains the following columns:

| **Column Name**   | **Description**                             |
| ----------------- | ------------------------------------------- |
| `WindSpeed(m/s)`  | Wind speed measured in meters per second    |
| `Pressure(hpa)`   | Atmospheric pressure in hectopascals        |
| `Temperature(°C)` | Temperature measured in degrees Celsius     |
| `Humidity(%)`     | Humidity level as a percentage              |
| `Sunlight(Lux)`   | Sunlight intensity measured in Lux          |
| `Power(mW)`       | **Target:** Power output in milliwatts (mW) |

### 3. Processed Data
Once the raw data is processed, it is returned as a list of dictionaries. Each dictionary corresponds to either a specific CSV file or the combined data from all files, depending on the `combine` setting:

- **True:** The list contains only one element.
- **False:** The length of the list equals the number of CSV files in the folder (i.e., one dictionary per file).

```python
[
    {
        "file_name": "<csv_file_name_or_combined>",
        "time_series": {
            "train": {
                "x": np.array((n_samples, seq_length, n_features), dtype=np.float32),
                "y": np.array((n_samples, n_features), dtype=np.float32)
            },
            "valid": {
                "x": np.array((n_samples, seq_length, n_features), dtype=np.float32),
                "y": np.array((n_samples, n_features), dtype=np.float32)
            },
            // Optional
            "test": {
                "x": np.array((n_samples, seq_length, n_features), dtype=np.float32),
                "y": np.array((n_samples, n_features), dtype=np.float32)
            } 
        },
        "regression": {
            "train": {
                "x": np.array((n_samples, n_features), dtype=np.float32),
                "y": np.array((n_samples, 1), dtype=np.float32)
            },
            "valid": {
                "x": np.array((n_samples, n_features), dtype=np.float32),
                "y": np.array((n_samples, 1), dtype=np.float32)
            },
            // Optional
            "test": {
                "x": np.array((n_samples, n_features), dtype=np.float32),
                "y": np.array((n_samples, 1), dtype=np.float32)
            } 
        }
    }
]
```

## 🚀 Usage
To run the script for the PV power generation forecast, follow the usage command below. This assumes you have organized your datasets correctly in the specified folder.

```bash
python main.py \
    --train_folder ./data/TrainData \
    --look_back_steps 12 \
    --n_valid_months 2 \
    --random_state 42 \
    --combine
```

| Argument          | Type   | Default    | Description                                                                                    |
| ----------------- | ------ | ---------- | ---------------------------------------------------------------------------------------------- |
| `train_folder`    | `Path` | `Required` | Path to the folder containing CSV files for training data.                                     |
| `test_folder`     | `Path` | `None`     | Path to the folder containing CSV files for testing data (optional).                           |
| `look_back_steps` | `int`  | `12`       | Number of time steps (lags) to look back for preparing the time series data for model input.   |
| `n_valid_months`  | `int`  | `2`        | Number of most recent months to use as the validation set.                                     |
| `random_state`    | `int`  | `42`       | Integer used to ensure reproducibility of results by fixing the random seed.                   |
| `combine`         | `bool` | `False`    | If set, combines all CSV files in the folder and trains a single model with the combined data. |

## 📈 Results
