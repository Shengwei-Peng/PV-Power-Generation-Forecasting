# PV-Power-Generation-Forecast

## Table of Contents
- [Overview](#Overview)
- [To Do List](#To-Do-List)
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

## Results
| Exp | Method                     | MAE      | MSE        | RMSE     | R² Score |
| --- | -------------------------- | -------- | ---------- | -------- | -------- |
| 001 | Random Forest (individual) | 126.4456 | 97277.6438 | 267.8911 | 0.6877   |
| 002 | Random Forest (combined)   |  95.5347 | 52882.9839 | 229.9630 | 0.8394   |