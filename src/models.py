"""models"""
import math

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
        epochs: int = 1,
        batch_size: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
    ) -> None:
        self.model_type = model_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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

        for _ in tqdm(range(self.epochs), unit="epoch", desc=f"Training {self.model_type}"):
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

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
            self.model = MLP(
                x.shape[1], self.hidden_size, 1, num_layers=self.num_layers
            )
        elif self.model_type == "LSTM":
            self.model = LSTM(
                x.shape[2], self.hidden_size, x.shape[2], num_layers=self.num_layers
            )
        elif self.model_type == "Transformer":
            self.model = Transformer(
                x.shape[2], self.hidden_size, x.shape[2], num_layers=self.num_layers
            )
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()


class MLP(nn.Module):
    """MLP"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward"""
        for layer in self.layers:
            x = layer(x)
        return x


class LSTM(nn.Module):
    """LSTM"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward"""
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x


class Transformer(nn.Module):
    """Transformer"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward"""
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x


class PositionalEncoding(nn.Module):
    """PositionalEncoding"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
