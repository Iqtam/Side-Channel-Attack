import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import argparse
# Configuration

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8
INPUT_SIZE = 1000
HIDDEN_SIZE = 128
PATIENCE = 5

# Ensure models directory exists
# os.makedirs(MODELS_DIR, exist_ok=True)


class FingerprintClassifier(nn.Module):
    """Basic neural network model for website fingerprinting classification."""

    def __init__(self, input_size, hidden_size, num_classes):
        super(FingerprintClassifier, self).__init__()

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After two 2x pooling operations
        self.fc_input_size = conv_output_size * 64

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)

        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ComplexFingerprintClassifier(nn.Module):
    """A more complex neural network model for website fingerprinting classification."""

    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFingerprintClassifier, self).__init__()

        # 1D Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After three 2x pooling operations
        self.fc_input_size = conv_output_size * 128

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_size, num_classes)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)

        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)

        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, input_size, num_classes, cnn_channels=64, lstm_hidden=128, lstm_layers=1):
        super(CNN_LSTM_Classifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(cnn_channels, cnn_channels * 2,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm_input_size = cnn_channels * 2
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.cnn(x)     # (B, C, T')

        x = x.permute(0, 2, 1)  # (B, T', C) for LSTM

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)  # (B, T', 2H)
        x = lstm_out[:, -1, :]      # Take the last output (B, 2H)

        x = self.fc(x)
        return x


class WebsiteTraceDataset(Dataset):
    def __init__(self, json_path, input_size=1000, scaler=None, label_encoder=None, normalized=False):
        with open(json_path, 'r') as f:
            data = json.load(f)
        traces = []
        labels = []
        websites = []
        for item in data:
            trace = item['trace_data']['trace'] if isinstance(item['trace_data'], dict) else item['trace_data']
            if len(trace) < input_size:
                trace = trace + [0] * (input_size - len(trace))
            else:
                trace = trace[:input_size]
            traces.append(trace)
            labels.append(item['website'])
            websites.append(item['website'])
        self.websites = sorted(list(set(websites)))
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.websites)
        else:
            self.label_encoder = label_encoder
        self.labels = self.label_encoder.transform(labels)
        traces = np.array(traces, dtype=np.float32)
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(traces)
        else:
            self.scaler = scaler
        if not normalized:
            traces = self.scaler.transform(traces)
        self.traces = torch.tensor(traces, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        return self.traces[idx], self.labels[idx]

def plot_training_curves(train_acc, test_acc, train_loss, test_loss, save_dir):
    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

def evaluate(model, test_loader, website_names, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for traces, labels in test_loader:
            traces, labels = traces.to(device), labels.to(device)
            outputs = model(traces)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=website_names,
        zero_division=1
    )

    print("\nClassification Report:")
    print(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)

    return all_preds, all_labels

def train(model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    best_loss = float('inf')
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for traces, labels in test_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_test_loss = running_loss / len(test_loader.dataset)
        epoch_test_acc = correct / total
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_acc)

        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f}')

        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            best_accuracy = epoch_test_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with val loss: {best_loss:.4f}, acc: {best_accuracy:.4f}')
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    plot_training_curves(train_accuracies, test_accuracies, train_losses, test_losses, os.path.dirname(model_save_path))
    return best_accuracy



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalized', type=str, default="False", help='Whether the dataset is already normalized (True/False)')
    parser.add_argument('--model', type=str, default="Simple", choices=['Simple', 'Complex', 'CNNLSTM'], help='Model architecture to use')
    parser.add_argument('--dataset_path', type=str, default="./data/dataset_2005028_3k.json", help='Path to the dataset JSON file')
    args = parser.parse_args()
    args.normalized = args.normalized.lower() == "true"

    model_dir = os.path.join("./saved_models", args.model)
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "fingerprint_classifier.pt")

    print("Loading and preprocessing dataset...")
    full_dataset = WebsiteTraceDataset(args.dataset_path, input_size=INPUT_SIZE, normalized=args.normalized)
    label_encoder = full_dataset.label_encoder
    scaler = full_dataset.scaler
    website_names = list(label_encoder.classes_)
    print(f"Websites: {website_names}")
    print(f"Total samples: {len(full_dataset)}")

    labels_np = full_dataset.labels.numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-TRAIN_SPLIT, random_state=42)
    train_idx, test_idx = next(sss.split(np.zeros(len(labels_np)), labels_np))
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(website_names)
    if args.model == "Simple":
        model = FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    elif args.model == "Complex":
        model = ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    elif args.model == "CNNLSTM":
        model = CNN_LSTM_Classifier(INPUT_SIZE, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining model...")
    best_acc = train(model, train_loader, test_loader, criterion, optimizer, EPOCHS, model_save_path)
    print(f"Best test accuracy: {best_acc:.4f}")

    print("\nEvaluating best model...")
    model.load_state_dict(torch.load(model_save_path))
    eval_path = os.path.join(model_dir, "evaluation.txt")
    evaluate(model, test_loader, website_names, save_path=eval_path)

    import joblib
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print("Saved label encoder and scaler.")

if __name__ == "__main__":
    main()
