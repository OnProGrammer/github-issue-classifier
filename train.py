import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import youtokentome as yttm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


CLASS_NAMES = ["BUG", "FEATURE", "DOCS", "SUPPORT"]


class IssueDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int],
                 tokenizer: yttm.BPE, max_length: int = 256):
        self.labels = labels
        self.max_length = max_length
        self.encoded = [self._encode(t, tokenizer) for t in texts]

    def _encode(self, text: str, tokenizer: yttm.BPE) -> np.ndarray:
        ids = tokenizer.encode(text, output_type=yttm.OutputType.ID)
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        else:
            ids = ids + [0] * (self.max_length - len(ids))
        return np.array(ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.encoded[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


class IssueClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, fc_hidden: int = 128,
                 num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.head(h_cat)


class Trainer:
    def __init__(self, csv_path: str, bpe_model_path: str, model_save_path: str,
                 bpe_vocab_size: int = 16000, max_seq_len: int = 256,
                 batch_size: int = 64, num_epochs: int = 6,
                 learning_rate: float = 5e-4, device: str | None = None):
        self.csv_path = csv_path
        self.bpe_model_path = bpe_model_path
        self.model_save_path = model_save_path
        self.bpe_vocab_size = bpe_vocab_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer: yttm.BPE | None = None
        self.model: IssueClassifier | None = None

        self.train_texts: list[str] = []
        self.test_texts: list[str] = []
        self.train_labels: list[int] = []
        self.test_labels: list[int] = []

        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
        }

    def load_data(self) -> None:
        df = pd.read_csv(self.csv_path).dropna(subset=["text", "label", "split"])
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)

        train_df = df[df["split"] == "train"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)

        self.train_texts = train_df["text"].tolist()
        self.train_labels = train_df["label"].tolist()
        self.test_texts = test_df["text"].tolist()
        self.test_labels = test_df["label"].tolist()

    def train_bpe(self) -> None:
        if os.path.exists(self.bpe_model_path):
            self.tokenizer = yttm.BPE(model=self.bpe_model_path)
            return

        corpus_path = os.path.join(os.path.dirname(self.bpe_model_path), "bpe_corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for t in self.train_texts + self.test_texts:
                f.write(t.replace("\r\n", " ").replace("\n", " ").strip() + "\n")

        yttm.BPE.train(
            data=corpus_path,
            vocab_size=self.bpe_vocab_size,
            model=self.bpe_model_path,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )
        self.tokenizer = yttm.BPE(model=self.bpe_model_path)

    def create_dataloaders(self) -> None:
        train_ds = IssueDataset(self.train_texts, self.train_labels,
                                self.tokenizer, self.max_seq_len)
        test_ds = IssueDataset(self.test_texts, self.test_labels,
                               self.tokenizer, self.max_seq_len)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    def setup_model(self) -> None:
        vocab_size = self.tokenizer.vocab_size()
        self.model = IssueClassifier(vocab_size=vocab_size).to(self.device)

    def fit(self) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_epoch(criterion, optimizer)
            val_loss, val_acc, _ = self._evaluate(criterion)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch}/{self.num_epochs} | "
                  f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"Val loss={val_loss:.4f} acc={val_acc:.4f}")

        _, _, report = self._evaluate(criterion, compute_report=True)
        print(f"\nFinal test report:\n{report}")

    def _train_epoch(self, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        y_true, y_pred = [], []

        for x_batch, y_batch in self.train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss, accuracy_score(y_true, y_pred)

    @torch.no_grad()
    def _evaluate(self, criterion: nn.Module,
                  compute_report: bool = False) -> tuple[float, float, str | None]:
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []

        for x_batch, y_batch in self.test_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            logits = self.model(x_batch)
            total_loss += criterion(logits, y_batch).item() * x_batch.size(0)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

        avg_loss = total_loss / len(self.test_loader.dataset)
        acc = accuracy_score(y_true, y_pred)

        report = None
        if compute_report:
            report = classification_report(y_true, y_pred,
                                           target_names=CLASS_NAMES, digits=4)
        return avg_loss, acc, report

    def save_model(self) -> None:
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_save_path)

    def plot_training_curves(self, out_path: str = "graphs/training_stats.png") -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        epochs = list(range(1, len(self.history["train_loss"]) + 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(epochs, self.history["train_loss"], label="Train Loss")
        ax1.plot(epochs, self.history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_xticks(epochs)
        ax1.legend()

        ax2.plot(epochs, self.history["train_acc"], label="Train Accuracy")
        ax2.plot(epochs, self.history["val_acc"], label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_xticks(epochs)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def plot_confusion_matrix(self, out_path: str = "graphs/confusion_matrix.png") -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.model.eval()
        y_true, y_pred = [], []

        for x_batch, y_batch in self.test_loader:
            x_batch = x_batch.to(self.device)
            logits = self.model(x_batch)
            y_true.extend(y_batch.numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45)
        ax.set_yticklabels(CLASS_NAMES)

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Confusion Matrix")
        fig.colorbar(ax.images[0], ax=ax)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train issue classifier (BiLSTM + BPE)")
    parser.add_argument("--csv", type=str, default="out/issues.csv")
    parser.add_argument("--bpe-model", type=str, default="out/bpe_issues.model")
    parser.add_argument("--model-out", type=str, default="out/issues_ml_4_model.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / mps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(
        csv_path=args.csv,
        bpe_model_path=args.bpe_model,
        model_save_path=args.model_out,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
    )

    trainer.load_data()
    trainer.train_bpe()
    trainer.create_dataloaders()
    trainer.setup_model()
    trainer.fit()
    trainer.save_model()
    trainer.plot_training_curves()
    trainer.plot_confusion_matrix()
