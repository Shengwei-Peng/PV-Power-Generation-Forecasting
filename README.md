# PV-Power-Generation-Forecast

## 📚 Table of Contents
- [Overview](#Overview)
- [Installation](#Installation)
- [Dataset](#Dataset)
- [Usage](#Usage)
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

### Processed Data
Once the raw data is processed, it is returned as a dictionaries:

```python
{
    "time_series": {
        "train": {
            "x": np.array((n_samples, seq_length, n_features), dtype=np.float32),
            "y": np.array((n_samples, n_features), dtype=np.float32)
        },
        # Optional
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
        # Optional
        "test": {
            "x": np.array((n_samples, n_features), dtype=np.float32),
            "y": np.array((n_samples, 1), dtype=np.float32)
        } 
    }
}
```

## 🚀 Usage

## 📈 Results
