"""main"""
from src.utils import load_data, find_best_model

def main() -> None:
    """main"""
    data = load_data("./TrainingData/L1_Train.csv")
    model = find_best_model(data)

if __name__ == "__main__":
    main()
