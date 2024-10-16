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
- [ ] Understand the【微氣候數據處理】program in the SampleCode.
    > Daniel
- [ ] Search for available external data (e.g.,氣象局資料)
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
- Observe the outlier of Data 1-17
- Verify the data completeness.
- Divide the data into training and validation datasets.

## 🚀 Usage
To run the script for the PV power generation forecast, follow the usage command below. This assumes you have organized your datasets correctly in the specified folder.

```bash
python main.py \
    --data_folder ./data/TrainData \
    --look_back_steps 12 \
    --n_valid_months 2 \
    --random_state 42 \
    --combine_data
```

| Argument          | Type   | Default    | Description                                                                                    |
| ----------------- | ------ | ---------- | ---------------------------------------------------------------------------------------------- |
| `train_folder`    | `Path` | `Required` | Path to the folder containing CSV files for training data.                                     |
| `test_folder`     | `Path` | `None`     | Path to the folder containing CSV files for testing data (optional).                           |
| `combine_data`    | `bool` | `False`    | If set, combines all CSV files in the folder and trains a single model with the combined data. |
| `random_state`    | `int`  | `42`       | Integer used to ensure reproducibility of results by fixing the random seed.                   |
| `n_valid_months`  | `int`  | `2`        | Number of most recent months to use as the validation set.                                     |
| `look_back_steps` | `int`  | `12`       | Number of time steps (lags) to look back for preparing the time series data for model input.   |

## 📈 Results
