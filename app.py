import os
import time
import json
import io
import base64
import matplotlib.pyplot as plt
from flask import Flask, send_from_directory, request, jsonify
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
# from database import save_trace, get_all_traces, clear_traces, init_db

# Set Matplotlib backend to non-interactive 'Agg' to avoid thread-related issues
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Model Configuration
INPUT_SIZE = 1000
HIDDEN_SIZE = 128
MODELS_DIR = "./saved_models/CNNLSTM_70k"
MODEL_NAME = "CNNLSTM"
DATASET_STATS_PATH = "./data/dataset_merged_70k_stats.json"


mean_vector = None
std_vector = None


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


# Global variables for model and preprocessing
model = None
scaler = None
label_encoder = None
device = None


def load_model():
    """Load the trained model and preprocessing objects"""
    global model, scaler, label_encoder, device

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load preprocessing objects
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        label_encoder = joblib.load(
            os.path.join(MODELS_DIR, "label_encoder.pkl"))
        # print(label_encoder.classes_)

        # Initialize and load model
        num_classes = len(label_encoder.classes_)
        if MODEL_NAME == "Simple":
            model = FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
        elif MODEL_NAME == "Complex":
            model = ComplexFingerprintClassifier(
                INPUT_SIZE, HIDDEN_SIZE, num_classes)
        elif MODEL_NAME == "CNNLSTM":
            model = CNN_LSTM_Classifier(INPUT_SIZE, num_classes)
        model.load_state_dict(torch.load(os.path.join(
            MODELS_DIR, "fingerprint_classifier.pt"), map_location=device))
        model.to(device)
        model.eval()

        print(f"Model loaded successfully!")
        print(f"Classes: {list(label_encoder.classes_)}")
        print(f"Device: {device}")
        # print("Label encoder classes:", label_encoder.classes_)
        # print("Label encoder mapping:")
        # for i, label in enumerate(label_encoder.classes_):
        #     print(f"{i}: {label}")

        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def load_dataset_stats():
    """Load dataset normalization statistics"""
    global mean_vector, std_vector

    try:
        with open(DATASET_STATS_PATH, 'r') as f:
            stats = json.load(f)
            mean_vector = np.array(stats['mean'], dtype=np.float32)
            std_vector = np.array(stats['std'], dtype=np.float32)
        print("Dataset statistics loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading dataset statistics: {e}")
        return False


def predict_website(trace):
    """Predict the website from a trace"""
    global model, scaler, label_encoder, device

    if model is None or scaler is None or label_encoder is None:
        return None, None

    try:
        # Preprocess the trace
        # print(f"Received trace of length {len(trace)}")
        if len(trace) < INPUT_SIZE:
            trace = trace + [0] * (INPUT_SIZE - len(trace))
        else:
            trace = trace[:INPUT_SIZE]

        trace_array = np.array([trace], dtype=np.float32)
        trace_normalized = scaler.transform(
            trace_array)  # ✅ GOOD — using loaded scaler
        trace_tensor = torch.tensor(
            trace_normalized, dtype=torch.float32).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(trace_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = label_encoder.inverse_transform([predicted.item()])[
                0]
            confidence_score = confidence.item()

            # Get top 3 predictions
            top_probs, top_indices = torch.topk(
                probabilities, k=min(3, len(label_encoder.classes_)))
            top_predictions = []
            for i in range(len(top_indices[0])):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                class_name = label_encoder.inverse_transform([idx])[0]
                top_predictions.append({
                    'website': class_name,
                    'confidence': float(prob)
                })

            return predicted_class, confidence_score, top_predictions

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None, None


# Load the model at startup
# load_dataset_stats()
load_model()


# Initialize the database
# init_db()

# In-memory storage for the current session
stored_traces = []
stored_heatmaps = []


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    """ 
    Implement the collect_trace endpoint to receive trace data from the frontend and generate a heatmap.
    1. Receive trace data from the frontend as JSON
    2. Generate a heatmap using matplotlib
    3. Store the heatmap and trace data in the backend temporarily
    4. Return the heatmap image and optionally other statistics to the frontend
    """
    try:
        data = request.json
        trace = data.get('trace')

        if not trace:
            return jsonify({'error': 'No trace data received'}), 400

        # Generate timestamp for this trace
        timestamp = int(time.time())

        # Store the trace data
        stored_traces.append({
            'timestamp': timestamp,
            'trace': trace
        })

        # Convert trace to numpy array for visualization
        trace_array = np.array(trace)

        # Generate heatmap
        plt.figure(figsize=(10, 2))
        plt.imshow([trace_array], aspect='auto', cmap='hot')
        # plt.colorbar(orientation='horizontal')
        # plt.title(f'Trace {timestamp}')
        # plt.xlabel('Sample Index')
        # plt.ylabel('Cache Access Count')
        plt.axis('off')  # Hide axes for cleaner look
        plt.tight_layout()
        # Save figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # Convert figure to base64 for embedding in HTML
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        image_data = f'data:image/png;base64,{image_base64}'

        # Store the heatmap
        heatmap_data = {
            'timestamp': timestamp,
            'image': image_data
        }
        stored_heatmaps.append(heatmap_data)

        # Real-time website prediction
        predicted_website, confidence, top_predictions = predict_website(trace)

        prediction_result = None
        if predicted_website is not None:
            prediction_result = {
                'predicted_website': predicted_website,
                'confidence': float(confidence),
                'top_predictions': top_predictions
            }
            print(
                f"Predicted website: {predicted_website} (confidence: {confidence:.3f})")

        # Save trace to database
        # save_trace(trace)

        # Calculate stats for clarity
        min_val = int(np.min(trace_array))
        max_val = int(np.max(trace_array))
        range_val = max_val - min_val

        # Print debugging information
        print(
            f"Trace stats - Min: {min_val}, Max: {max_val}, Range: {range_val}, Samples: {len(trace_array)}")

        # Return the heatmap and metadata
        response_data = {
            'heatmap': heatmap_data,
            'stats': {
                'min': min_val,
                'max': max_val,
                'range': range_val,
                'mean': float(np.mean(trace_array)),
                'std': float(np.std(trace_array)),
                'samples': len(trace_array)
            }
        }

        # Add prediction result if available
        if prediction_result:
            response_data['prediction'] = prediction_result

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in collect_trace: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """ 
    Implement a clear results endpoint to reset stored data.
    1. Clear stored traces and heatmaps
    2. Return success/error message
    """
    try:
        # Clear in-memory storage
        stored_traces.clear()
        stored_heatmaps.clear()

        # Clear database
        # clear_traces()

        return jsonify({
            'message': 'All results cleared successfully',
            'status': 'success'
        })
    except Exception as e:
        print(f"Error in clear_results: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/traces', methods=['GET'])
def get_traces():
    """
    Return all stored traces for download
    """
    try:

        return jsonify(stored_traces)
    except Exception as e:
        print(f"Error in get_traces: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/heatmaps', methods=['GET'])
def get_heatmaps():
    """
    Return all stored heatmaps
    """
    try:
        return jsonify(stored_heatmaps)
    except Exception as e:
        print(f"Error in get_heatmaps: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stored_traces', methods=['GET'])
def get_stored_traces():
    """
    Return all stored traces in memory
    """
    try:
        return jsonify(stored_traces)
    except Exception as e:
        print(f"Error in get_stored_traces: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_results', methods=['GET'])
def get_results():
    """
    Return current in-memory traces and heatmaps for the Selenium script
    """
    try:
        return jsonify({
            'traces': stored_traces,
            'heatmaps': stored_heatmaps
        })
    except Exception as e:
        print(f"Error in get_results: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_from_trace():
    """
    Real-time website prediction endpoint
    """
    try:
        data = request.json
        trace = data.get('trace')

        if not trace:
            return jsonify({'error': 'No trace data received'}), 400

        predicted_website, confidence, top_predictions = predict_website(trace)

        if predicted_website is None:
            return jsonify({'error': 'Model not available for prediction'}), 500

        return jsonify({
            'predicted_website': predicted_website,
            'confidence': float(confidence),
            'top_predictions': top_predictions,
            'status': 'success'
        })

    except Exception as e:
        print(f"Error in predict_from_trace: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """
    Get information about the loaded model
    """
    try:
        if model is None or label_encoder is None:
            return jsonify({
                'model_loaded': False,
                'error': 'Model not loaded'
            })

        return jsonify({
            'model_loaded': True,
            'classes': list(label_encoder.classes_),
            'num_classes': len(label_encoder.classes_),
            'input_size': INPUT_SIZE,
            'device': str(device)
        })
    except Exception as e:
        print(f"Error in get_model_info: {e}")
        return jsonify({'error': str(e)}), 500

# server should reload on changes


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, host='0.0.0.0', port=5000)
