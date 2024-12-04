"""reproduc"""
import pickle
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.utils import train_and_predict, create_ensemble_submission, post_process

def main() -> None:
    """main"""
    parser = ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--upload_template", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="Reproduction")
    args = parser.parse_args()

    folder = Path(args.folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    upload_template = Path(args.upload_template)

    dataset = {
        "train": {
            "X": pd.read_csv(f"{folder}/train_x.csv"),
            "y": pd.read_csv(f"{folder}/train_y.csv").squeeze()
        },
        "test": {
            "X": pd.read_csv(f"{folder}/test_x.csv")
        }
    }

    print("Shapes of the data:")
    print(f"Train X: {dataset['train']['X'].shape}")
    print(f"Train y: {dataset['train']['y'].shape}")
    print(f"Test X: {dataset['test']['X'].shape}")

    if args.retrain:
        train_and_predict(
            model=CatBoostRegressor(iterations=int(1e7), task_type="GPU", verbose=int(1e5)),
            model_name="catboost",
            dataset=dataset,
            upload_template=upload_template,
            output_folder=output_folder,
        )
        train_and_predict(
            model=LGBMRegressor(num_leaves=int(2**15 - 1)),
            model_name="lightgbm",
            dataset=dataset,
            upload_template=upload_template,
            output_folder=output_folder,
        )
        train_and_predict(
            model=XGBRegressor(
                n_estimators=int(1e6), learning_rate=0.001, tree_method="hist", device="cuda"
            ),
            model_name="xgboost",
            dataset=dataset,
            upload_template=upload_template,
            output_folder=output_folder,
        )
    else:
        with open(folder / "lightgbm_model.pkl", "rb") as file:
            model = pickle.load(file)
        predictions = model.predict(dataset["test"]["X"])
        upload_data = pd.read_csv(upload_template)
        upload_data["答案"] = post_process(predictions)
        upload_data.to_csv(output_folder / "lightgbm_pred.csv", index=False)

        with open(folder / "xgboost_model.pkl", "rb") as file:
            model = pickle.load(file)
        predictions = model.predict(dataset["test"]["X"])
        upload_data = pd.read_csv(upload_template)
        upload_data["答案"] = post_process(predictions)
        upload_data.to_csv(output_folder / "xgboost_pred.csv", index=False)

        predictions = pd.read_csv(folder / "catboost_pred.csv")
        predictions.to_csv(output_folder / "catboost_pred.csv", index=False)

    create_ensemble_submission(
        model_preds=[
            output_folder / "catboost_pred.csv",
            output_folder / "lightgbm_pred.csv",
            output_folder / "xgboost_pred.csv"
        ],
        upload_template=upload_template,
        output_file= output_folder / "submission.csv"
    )

if __name__ == "__main__":
    main()
