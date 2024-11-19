# PV-Power-Generation-Forecasting

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
    git clone https://github.com/Shengwei-Peng/PV-Power-Generation-Forecasting.git
    ```
2. Navigate to the project directory:
    ```sh
    cd PV-Power-Generation-Forecasting
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

| ID  | Submitter | Upload Time | Public Score | Note                |
| --- | --------- | ----------- | ------------:| ------------------- |
| 01  | Ken       | 2024-11-18  |   2080700.95 |                     |
| 02  | Ken       | 2024-11-18  |   1789119.30 | All zero            |
| 03  | Ken       | 2024-11-18  |   1936269.36 | Average (Overall)   |
| 04  | Ken       | 2024-11-18  |   1811221.68 | Average (10-Minute) |
| 05  | Benson    | 2024-11-18  |   1837177.48 | Average total error |
| 06  | Ken       | 2024-11-19  |   1987431.54 |                     |
| 07  | Daniel    | 2024-11-19  |   1720059.50 | All 76              |
| 08  | Ken       | 2024-11-19  |    985780.43 |                     |
| 09  | Ken       | 2024-11-19  |    854752.05 |                     |
| 10  | Ken       | 2024-11-19  |   1072685.57 | Previous day        |