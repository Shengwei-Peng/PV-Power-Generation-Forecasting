# PV-Power-Generation-Forecasting

## 📚 Table of Contents
- [Overview](#Overview)
- [Usage](#usage)
- [Installation](#Installation)
- [Dataset](#Dataset)
- [Results](#Results)
- [Acknowledgements](#acknowledgements)
- [Contributing](#Contributing)
- [License](#license)
- [Contact](#contact)

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

### External Data

## 🛠️ Usage

## 📈 Results

<details>
    <summary>Show/Hide Results Table</summary>

| ID  | Submitter | Upload Time |  Public Score | Note                |
| --- | --------- | ----------- | -------------:| ------------------- |
| 01  | Ken       | 2024-11-18  |    2080700.95 |                     |
| 02  | Ken       | 2024-11-18  |    1789119.30 | All zero            |
| 03  | Ken       | 2024-11-18  |    1936269.36 | Average (Overall)   |
| 04  | Ken       | 2024-11-18  |    1811221.68 | Average (10-Minute) |
| 05  | Benson    | 2024-11-18  |    1837177.48 | Average total error |
| 06  | Ken       | 2024-11-19  |    1987431.54 |                     |
| 07  | Daniel    | 2024-11-19  |    1720059.50 | All 76              |
| 08  | Ken       | 2024-11-19  |     985780.43 |                     |
| 09  | Ken       | 2024-11-19  |     854752.05 |                     |
| 10  | Ken       | 2024-11-19  |    1072685.57 | Previous day        |
| 11  | Ken       | 2024-11-20  |     824205.41 |                     |
| 12  | Ken       | 2024-11-20  |     628281.62 |                     |
| 13  | Ken       | 2024-11-20  |     581359.93 |                     |
| 14  | Ken       | 2024-11-20  |     560151.59 |                     |
| 15  | Ken       | 2024-11-20  |     570915.15 |                     |
| 16  | Ken       | 2024-11-21  |     503108.75 |                     |
| 17  | Ken       | 2024-11-21  |     575301.23 |                     |
| 18  | Ken       | 2024-11-21  |     566783.82 |                     |
| 19  | Ken       | 2024-11-21  |     503497.19 |                     |
| 20  | Ken       | 2024-11-21  |     651794.75 |                     |
| 21  | Ken       | 2024-11-22  |     500977.86 |                     |
| 22  | Ken       | 2024-11-22  |     500588.64 |                     |
| 23  | Ken       | 2024-11-22  |     502596.13 |                     |
| 24  | Ken       | 2024-11-22  |    2067988.84 |                     |
| 25  | Ken       | 2024-11-22  |     483064.06 |                     |
| 26  | Ken       | 2024-11-23  |     472851.80 |                     |
| 27  | Ken       | 2024-11-23  |     461282.36 |                     |
| 28  | Ken       | 2024-11-23  |     418670.68 |                     |
| 29  | Ken       | 2024-11-23  |     407476.71 |                     |
| 30  | Ken       | 2024-11-23  |     401027.85 |                     |
| 31  | Ken       | 2024-11-24  |     407476.71 |                     |
| 32  | Ken       | 2024-11-24  |     400222.27 |                     |
| 33  | Ken       | 2024-11-24  |     442772.71 |                     |
| 34  | Ken       | 2024-11-24  |     398627.30 |                     |
| 35  | Ken       | 2024-11-24  |     393346.07	|                     |
| 36  | Ken       | 2024-11-25  |     417173.37 |                     |
| 37  | Ken       | 2024-11-25  |     404582.89 |                     |
| 38  | Ken       | 2024-11-25  |     407552.42	|                     |
| 39  | Ken       | 2024-11-25  |     377853.09	|                     |
| 40  | Ken       | 2024-11-25  |     397801.42 |                     |
| 41  | Ken       | 2024-11-26  |     388330.38 |                     |
| 42  | Ken       | 2024-11-26  |     390831.75 |                     |
| 43  | Ken       | 2024-11-26  |     378537.11 |                     |
| 44  | Ken       | 2024-11-26  |     368758.14 |                     |
| 45  | Ken       | 2024-11-26  |     381491.54	|                     |
| 46  | Ken       | 2024-11-27  |     388500.90	|                     |
| 47  | Ken       | 2024-11-27  |     377664.51	|                     |
| 48  | Ken       | 2024-11-27  |     402065.87	|                     |
| 49  | Ken       | 2024-11-27  |     397547.12	|                     |
| 50  | Ken       | 2024-11-27  |     431869.78	|                     |
| 51  | Ken       | 2024-11-28  |     388686.62 |                     |
| 52  | Ken       | 2024-11-28  |     366082.67	|                     |
| 53  | Ken       | 2024-11-28  |     394886.78 |                     |
| 54  | Ken       | 2024-11-28  |     386407.25 |                     |
| 55  | Ken       | 2024-11-28  | **356359.07** |                     |

</details>

## 🙏 Acknowledgements

We would like to express our sincere gratitude to the following organizations for their support and contributions to this project:

- **Competition Guidance Unit**: Information and Technology Education Department of the Ministry of Education.
- **Competition Planning Unit**: Artificial Intelligence Competition and Annotation Data Collection Project Office of the Ministry of Education.
- **Topic Provider**: Department of Information Engineering, National Dong Hwa University.
- **Platform Sponsor**: Trend Micro.

## 🤝 Contributing

We welcome contributions to the project! Please follow the guidelines below:

1. Fork the repository.
2. Create a new branch (`feature/your-feature-name`).
3. Commit your changes.
4. Submit a pull request.

## ⚖️ License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.

## 📧 Contact

For any questions or inquiries, please contact m11207330@mail.ntust.edu.tw