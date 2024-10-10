"""main"""
from src import parse_arguments, Trainer


def main() -> None:
    """main"""
    args = parse_arguments()
    trainer = Trainer(**args)
    trainer.train()

if __name__ == "__main__":
    main()
