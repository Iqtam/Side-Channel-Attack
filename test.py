import json
import torch.nn as nn
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import joblib  # Add this import

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

def load_single_example_from_dataset(json_path, target_website, example_index=0):
    """Load a specific example from the dataset"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Find examples for the target website
    examples = []
    for item in data:
        if item['website'] == target_website:
            examples.append(item)
    
    if not examples:
        print(f"No examples found for {target_website}")
        return None
    
    if example_index >= len(examples):
        print(f"Index {example_index} out of range. Found {len(examples)} examples for {target_website}")
        return None
    
    return examples[example_index]

def load_model_and_preprocessors():
    models_dir = "./saved_models/CNNLSTM_70k"
    
    try:
        # Load model
        model_path = f"{models_dir}/fingerprint_classifier.pt"
        model_state = torch.load(model_path, map_location='cpu')
        
        # Initialize model
        model = CNN_LSTM_Classifier(input_size=1000, num_classes=3)
        model.load_state_dict(model_state)
        model.eval()
        print("Model loaded successfully!")
        
        # Try different ways to load scaler and label encoder
        try:
            # Method 1: joblib (most common)
            scaler = joblib.load(f"{models_dir}/scaler.pkl")
            label_encoder = joblib.load(f"{models_dir}/label_encoder.pkl")
            print("Loaded with joblib")
        except:
            try:
                # Method 2: pickle
                with open(f"{models_dir}/scaler.pkl", 'rb') as f:
                    scaler = pickle.load(f)
                with open(f"{models_dir}/label_encoder.pkl", 'rb') as f:
                    label_encoder = pickle.load(f)
                print("Loaded with pickle")
            except:
                # Method 3: Check if they're saved with different extensions
                try:
                    scaler = joblib.load(f"{models_dir}/scaler.joblib")
                    label_encoder = joblib.load(f"{models_dir}/label_encoder.joblib")
                    print("Loaded with .joblib extension")
                except:
                    print("Could not load scaler and label_encoder. Check file format.")
                    return None, None, None
        
        print(f"Classes: {label_encoder.classes_}")
        return model, scaler, label_encoder
        
    except Exception as e:
        print(f"Error loading model/preprocessors: {e}")
        return None, None, None

def pad_or_truncate_trace(trace, target_size):
    """Same preprocessing as in training"""
    if len(trace) >= target_size:
        return trace[:target_size]
    else:
        return trace + [0] * (target_size - len(trace))

def test_single_example():
    # Configuration
    dataset_path = "./data/dataset_merged_70k.json"  # Adjust path if needed
    target_website = "https://google.com"  # Change to test different sites
    example_index = 1  # Change to test different examples
    
    # Load model and preprocessors
    print("Loading model and preprocessors...")
    model, scaler, label_encoder = load_model_and_preprocessors()
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Load a single example from dataset
    print(f"\nLoading example {example_index} for {target_website}...")
    example = load_single_example_from_dataset(dataset_path, target_website, example_index)
    
    if example is None:
        return
    
    print(f"Loaded example:")
    print(f"Website: {example['website']}")
    print(f"Trace length: {len(example['trace_data'])}")
    print(f"First 10 trace values: {example['trace_data'][:10]}")
    
    # Process the trace (same as in training)
    trace = example['trace_data']
    processed_trace = pad_or_truncate_trace(trace, 1000)
    
    print(f"\nProcessed trace length: {len(processed_trace)}")
    
    # Scale the trace
    scaled_trace = scaler.transform([processed_trace])
    print(f"Scaled trace shape: {scaled_trace.shape}")
    
    # Convert to tensor and predict
    trace_tensor = torch.FloatTensor(scaled_trace)
    
    with torch.no_grad():
        outputs = model(trace_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Get prediction
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_website = label_encoder.inverse_transform([predicted_class])[0]
    confidence = probabilities[0][predicted_class].item() * 100
    
    print(f"\n=== PREDICTION RESULTS ===")
    print(f"True website: {example['website']}")
    print(f"Predicted website: {predicted_website}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Correct prediction: {example['website'] == predicted_website}")
    
    print(f"\nAll probabilities:")
    for i, prob in enumerate(probabilities[0]):
        website = label_encoder.inverse_transform([i])[0]
        print(f"  {website}: {prob.item() * 100:.1f}%")

if __name__ == "__main__":
    test_single_example()