"""models"""
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch import optim


class Model:
    """Model"""
    def __init__(
        self,
        model_type: str,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 128,
    ) -> None:
        self.model_type = model_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """fit"""
        self._build_model(train_x)

        train_x_tensor = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        train_y_tensor = torch.tensor(train_y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        with tqdm(
            range(self.epochs), unit="epoch", desc=f"Training {self.model_type}"
            ) as epoch_progress:
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

    def save(self, save_path: str) -> None:
        """save"""
        torch.save(self.model, save_path)

    def _build_model(self, x: np.ndarray) -> None:
        if self.model_type == "MLP":
            self.model = MLP(x.shape[1], 1024, 1)
        elif self.model_type == "LSTM":
            self.model = LSTM(x.shape[2], 1024, 5)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()


class MLP(nn.Module):
    """MLP"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """forward"""
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class LSTM(nn.Module):
    """LSTM"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward"""
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x
