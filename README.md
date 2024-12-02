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

| ID  | Submitter | Upload Time |  Public Score | Private Score | Note                |
| --- | --------- | ----------- | -------------:| -------------:| ------------------- |
| 01  | Ken       | 2024-11-18  |    2080700.95 |    2279572.22 |                     |
| 02  | Ken       | 2024-11-18  |    1789119.30 |    2174549.41 | All zero            |
| 03  | Ken       | 2024-11-18  |    1936269.36 |    2148889.63 | Average (Overall)   |
| 04  | Ken       | 2024-11-18  |    1811221.68 |    1842050.13 | Average (10-Minute) |
| 05  | Benson    | 2024-11-18  |    1837177.48 |    2096727.91 | Average total error |
| 06  | Ken       | 2024-11-19  |    1987431.54 |    2148135.81 |                     |
| 07  | Daniel    | 2024-11-19  |    1720059.50 |    2060600.83 | All 76              |
| 08  | Ken       | 2024-11-19  |     985780.43 |     960078.27 |                     |
| 09  | Ken       | 2024-11-19  |     854752.05 |     879881.16 |                     |
| 10  | Ken       | 2024-11-19  |    1072685.57 |    1101202.66 | Previous day        |
| 11  | Ken       | 2024-11-20  |     824205.41 |     870917.05 |                     |
| 12  | Ken       | 2024-11-20  |     628281.62 |     695738.97 |                     |
| 13  | Ken       | 2024-11-20  |     581359.93 |     667819.00 |                     |
| 14  | Ken       | 2024-11-20  |     560151.59 |     647792.75 |                     |
| 15  | Ken       | 2024-11-20  |     570915.15 |     598089.80 |                     |
| 16  | Ken       | 2024-11-21  |     503108.75 |     590526.12 |                     |
| 17  | Ken       | 2024-11-21  |     575301.23 |     617709.15 |                     |
| 18  | Ken       | 2024-11-21  |     566783.82 |     601752.06 |                     |
| 19  | Ken       | 2024-11-21  |     503497.19 |     556083.36 |                     |
| 20  | Ken       | 2024-11-21  |     651794.75 |     721346.92 |                     |
| 21  | Ken       | 2024-11-22  |     500977.86 |     558400.65 |                     |
| 22  | Ken       | 2024-11-22  |     500588.64 |     551565.93 |                     |
| 23  | Ken       | 2024-11-22  |     502596.13 |     541711.74 |                     |
| 24  | Ken       | 2024-11-22  |    2067988.84 |    2212960.51 |                     |
| 25  | Ken       | 2024-11-22  |     483064.06 |     593766.05 |                     |
| 26  | Ken       | 2024-11-23  |     472851.80 |     568710.49 |                     |
| 27  | Ken       | 2024-11-23  |     461282.36 |     564863.34 |                     |
| 28  | Ken       | 2024-11-23  |     418670.68 |     532466.45 |                     |
| 29  | Ken       | 2024-11-23  |     407476.71 |     508553.97 |                     |
| 30  | Ken       | 2024-11-23  |     401027.85 |     511159.04 |                     |
| 31  | Ken       | 2024-11-24  |     407476.71 |     507011.25 |                     |
| 32  | Ken       | 2024-11-24  |     400222.27 |     460386.16 |                     |
| 33  | Ken       | 2024-11-24  |     442772.71 |     531965.98 |                     |
| 34  | Ken       | 2024-11-24  |     398627.30 |     506857.27 |                     |
| 35  | Ken       | 2024-11-24  |     393346.07 |     451063.93 |                     |
| 36  | Ken       | 2024-11-25  |     417173.37 |     495012.90 |                     |
| 37  | Ken       | 2024-11-25  |     404582.89 |     523922.83 |                     |
| 38  | Ken       | 2024-11-25  |     407552.42 |     514735.62 |                     |
| 39  | Ken       | 2024-11-25  |     377853.09 |     485979.75 |                     |
| 40  | Ken       | 2024-11-25  |     397801.42 |     491359.28 |                     |
| 41  | Ken       | 2024-11-26  |     388330.38 |     473736.73 |                     |
| 42  | Ken       | 2024-11-26  |     390831.75 |     491397.55 |                     |
| 43  | Ken       | 2024-11-26  |     378537.11 |     487647.68 |                     |
| 44  | Ken       | 2024-11-26  |     368758.14 |     424508.42 |                     |
| 45  | Ken       | 2024-11-26  |     381491.54 |     495907.34 |                     |
| 46  | Ken       | 2024-11-27  |     388500.90 |     487503.77 |                     |
| 47  | Ken       | 2024-11-27  |     377664.51 |     452648.44 |                     |
| 48  | Ken       | 2024-11-27  |     402065.87 |     488541.13 |                     |
| 49  | Ken       | 2024-11-27  |     397547.12 |     482793.11 |                     |
| 50  | Ken       | 2024-11-27  |     431869.78 |     512407.47 |                     |
| 51  | Ken       | 2024-11-28  |     388686.62 |     482839.81 |                     |
| 52  | Ken       | 2024-11-28  |     366082.67 |     423951.51 |                     |
| 53  | Ken       | 2024-11-28  |     394886.78 |     499104.56 |                     |
| 54  | Ken       | 2024-11-28  |     386407.25 |     490622.69 |                     |
| 55  | Ken       | 2024-11-28  | **356359.07** | **419747.93** |                     |

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
