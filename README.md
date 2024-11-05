# PV-Power-Generation-Forecast

## 📚 Table of Contents
- [Overview](#Overview)
- [Installation](#Installation)
- [Dataset](#Dataset)
- [Results](#Results)

## 🌞 Overview
This project focuses on predicting solar photovoltaic (PV) power generation based on regional microclimate data. The objective is to forecast the power output of PV devices installed in various terrains using environmental data such as temperature, humidity, wind speed, solar radiation, and rainfall.

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
### Raw Data
The raw data CSV file contains the following columns:

| **Column Name**       | **Description**                                  | **Data Type** |
| --------------------- | ------------------------------------------------ | ------------- |
| `LocationCode`        | Location identifier                              | Integer       |
| `DateTime`            | Timestamp of measurement                         | DateTime      |
| `WindSpeed(m/s)`      | Wind speed in m/s                                | Float         |
| `Pressure(hpa)`       | Atmospheric pressure in hPa                      | Float         |
| `Temperature(°C)`     | Temperature in °C                                | Float         |
| `Humidity(%)`         | Humidity percentage                              | Float         |
| `Sunlight(Lux)`       | Sunlight intensity in Lux                        | Float         |
| `Power(mW)`           | **Target:** Power output in mW                   | Float         |

## 📈 Results

| Time  | Submitter | MAE     | MSE     | RMSE    | R² Score   |
| ----- | --------- | ------- | ------- | ------- | ---------- |
| 10/30 | Ken       | 403.523 | 302237  | 549.761 |    0.23937 |
| 11/04 | Daniel    | 759.586 | 845315  | 919.41  |   -1.06507 |
| 11/04 | Daniel    | 126.671 | 70367.2 | 265.268 |   0.828095 |
| 11/04 | Benson    | 597.676 | 466027 | 682.662 |  -0.138488 |
| 11/05 | Benson    | 414.572 | 360026 | 600.021 |    0.12047 |
