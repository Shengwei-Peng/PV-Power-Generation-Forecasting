"""models"""
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch import optim


class MLPRegressor:
    """MLPRegressor"""
    def __init__(
        self,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 1024
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """fit"""
        input_dim = train_x.shape[1]

        if self.model is None:
            self._build_model(input_dim)

        train_x_tensor = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        train_y_tensor = torch.tensor(train_y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        with tqdm(range(self.epochs), unit="epoch", desc="Training") as epoch_progress:
            for _ in epoch_progress:
                running_loss = 0.0
                for batch_x, batch_y in dataloader:
                    self.optimizer.zero_grad()

                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                avg_loss = running_loss / len(dataloader)
                epoch_progress.set_postfix(epoch_loss=avg_loss)

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        """predict"""
        self.model.eval()
        test_x_tensor = torch.tensor(test_x, dtype=torch.float32).to(self.device)

        test_dataset = torch.utils.data.TensorDataset(test_x_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)

        all_predictions = []

        with torch.no_grad():
            for batch_x in test_loader:
                predictions = self.model(batch_x[0])
                all_predictions.append(predictions.cpu().numpy())

        return np.vstack(all_predictions)

    def _build_model(self, input_dim: int) -> None:
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, 1)
        )
        self.model = self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
