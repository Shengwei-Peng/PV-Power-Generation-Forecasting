# PV-Power-Generation-Forecast

## Table of Contents
- [Overview](#Overview)
- [To Do List](#To-Do-List)
- [Results](#Results)

## Overview
This project focuses on predicting solar photovoltaic (PV) power generation based on regional microclimate data. The objective is to forecast the power output of PV devices installed in various terrains using environmental data such as temperature, humidity, wind speed, solar radiation, and rainfall.

## To Do List
- [ ] Merge all data for unified model training.
- [ ] Perform data analysis and create visualizations.
- [ ] Identify and integrate external data sources.
- [ ] Apply appropriate preprocessing techniques (e.g., normalization).
- [ ] Use deep learning methods (e.g., CNNs, RNNs, or Transformers).

## Results
| Exp | Method                     | MAE      | MSE        | RMSE     | R² Score |
| --- | -------------------------- | -------- | ---------- | -------- | -------- |
| 001 | Random Forest (individual) | 126.4456 | 97277.6438 | 267.8911 | 0.6877   |