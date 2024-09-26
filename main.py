"""main"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def main() -> None:
    """main"""
    data = pd.read_csv("./TrainingData/L1_Train.csv")

    x = data[["WindSpeed(m/s)", "Pressure(hpa)", "Temperature(°C)", "Humidity(%)", "Sunlight(Lux)"]]
    y = data["Power(mW)"] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
    }

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        print(f"{name} Mean Absolute Error: {mae}")

if __name__ == "__main__":
    main()
