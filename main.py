"""main"""
from src.utils import parse_arguments, train_individual_models


def main() -> None:
    """main"""
    args = parse_arguments()
    train_individual_models(args.data_folder)

if __name__ == "__main__":
    main()
