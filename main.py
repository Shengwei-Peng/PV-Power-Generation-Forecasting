"""main"""
from src.utils import parse_arguments, train


def main() -> None:
    """main"""
    args = parse_arguments()
    train(args.data_folder, args.combine_data)

if __name__ == "__main__":
    main()
