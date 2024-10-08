"""main"""
from src import parse_arguments, set_seed, Trainer


def main() -> None:
    """main"""
    args = parse_arguments()
    set_seed(args["random_state"])
    trainer = Trainer(**args)
    trainer.pre_process()
    trainer.train()

if __name__ == "__main__":
    main()
