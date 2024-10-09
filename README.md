# PV-Power-Generation-Forecast

## Table of Contents
- [Overview](#Overview)
- [To Do List](#To-Do-List)
- [Dataset](#Dataset)
- [Results](#Results)


## Overview
This project focuses on predicting solar photovoltaic (PV) power generation based on regional microclimate data. The objective is to forecast the power output of PV devices installed in various terrains using environmental data such as temperature, humidity, wind speed, solar radiation, and rainfall.

## To Do List
- [ ] Perform data analysis and create visualizations.
    > Daniel
- [ ] Find and integrate external data sources.
    > Benson
- [ ] Apply appropriate preprocessing techniques (e.g., normalization).
    > Ken
- [ ] Use deep learning methods (e.g., CNNs, RNNs, or Transformers).
    > Raymin

## Dataset

## Results
| Exp | Model             | MAE      | MSE      | RMSE     | R² Score | Note       |
| --- | ----------------- | -------- | -------- | -------- | -------- | ---------- |
| 001 | Linear Regression | 138.6666 | 53683.0  | 231.6959 | 0.8370   |            |
| 002 | XGBoost           | 89.3658  | 48590.8  | 220.4331 | 0.8524   |            |
| 003 | LightGBM          | 81.7840  | 41748.5  | 204.3245 | 0.8732   |            |
| 004 | CatBoost          | 93.5267  | 50719.8  | 225.2106 | 0.8460   |            |
| 005 | NGBoost           | 89.7819  | 45448.0  | 213.1853 | 0.8620   |            |
| 006 | TabNet            | 95.0234  | 49660.3  | 222.8460 | 0.8492   |            |
| 007 | MLP               | 90.5893  | 44563.4  | 211.1003 | 0.8647   |            |
| 008 | LightGBM          | 134.2215 | 104164.4 | 281.8344 | 0.6721   | Individual |
