# Website Fingerprinting Classification System

A comprehensive machine learning system for website fingerprinting classification using neural networks. This project implements multiple deep learning architectures to classify network traffic traces and identify websites based on their traffic patterns.

## 🎯 Project Overview

This system performs website fingerprinting classification using network traffic traces. It can classify traffic from three specific websites:
- **https://cse.buet.ac.bd/moodle/** - Educational platform
- **https://google.com** - Search engine
- **https://prothomalo.com** - News website

The system uses deep learning models including CNN, LSTM, and hybrid CNN-LSTM architectures to achieve classification accuracy of ~84%.

## 📁 Project Structure

```
├── app.py                          # Flask web application for model serving
├── collect.py                      # Data collection script using Selenium
├── train.py                        # Model training script
├── database.py                     # Database operations for trace storage
├── test.py                         # Testing utilities
├── requirements.txt                # Python dependencies
├── chromedriver.exe               # Chrome WebDriver executable
├── train.sh                       # Training automation script
├── data_and_model_link.txt        # Google Drive links to datasets and models
├── report.pdf                     # Project report
│
├── data/                          # Dataset directory
│   ├── dataset_2005028_3k.json   # Personal dataset (3k samples)
│   ├── dataset_merged_70k.json   # Merged dataset (70k samples)
│
├── saved_models/                  # Trained models directory
│   ├── CNNLSTM_70k/              # CNN-LSTM model (70k dataset)
│   ├── CNNLSTM_3k/           # CNN-LSTM model (3k dataset)
│   ├── Complex_70k/              # Complex CNN model
│   ├── Simple_70k/               # Simple CNN model
│   ├── Complex_3k/               # Simple CNN model
│   ├── Simple_70k/               # Simple CNN model
│
├── static/                        # Web application static files
│   ├── index.html                # Main web interface
│   ├── index.js                  # Frontend JavaScript
│   ├── warmup.js                 # Warmup functionality
│   └── worker.js                 # Web worker for background tasks
│
└── db/                           # Database files
    └── webfingerprint.db         # SQLite database for traces

```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Chrome Browser
- NVIDIA GPU (optional, for faster training)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone <repository-url>
cd template
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download datasets and pre-trained models (Optional):**

The complete datasets and pre-trained models are available on Google Drive. Check `data_and_model_link.txt` for the download link:

```bash
# View the Google Drive link
cat data_and_model_link.txt
```

**Available Downloads:**
- **Complete datasets**: `dataset_merged_70k.json` (70,000 samples)
- **Pre-trained models**: All model variants (CNNLSTM, Complex, Simple)
- **Model checkpoints**: Best performing model weights
- **Evaluation results**: Training curves and classification reports

**Note**: You can either:
- Download the complete datasets/models from Google Drive for immediate use
- Or collect your own data using `collect.py` and train models from scratch

4. **Initialize the database:**
```bash
python -c "from database import Database; db = Database(['test']); db.init_database()"
```

## 🚀 Usage

### 1. Data Collection

Collect network traffic traces from websites:

```bash
python collect.py
```

**Configuration:**
- Modify `WEBSITES` list in `collect.py` to add target websites
- Adjust `TRACES_PER_SITE` for number of traces per website
- Ensure Flask server is running on `http://localhost:5000`

### 2. Model Training

Train neural network models on collected data:

```bash
python train.py
```

**Available Models:**
- **Simple**: Basic fully connected neural network
- **Complex**: Multi-layer CNN with batch normalization
- **CNNLSTM**: Hybrid CNN-LSTM architecture (recommended)

**Training Configuration:**
```python
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
INPUT_SIZE = 1000
HIDDEN_SIZE = 128
```

### 3. Web Application

Start the Flask web server for real-time classification:

```bash
python app.py
```

Access the web interface at `http://localhost:5000`

**Features:**
- Real-time website classification
- Interactive trace visualization
- Model confidence scores
- Result history management

### 4. Model Evaluation

Test model performance on specific examples:

```bash
python test.py
```

## 🧠 Model Architectures

### 1. Simple Fingerprint Classifier
```python
class FingerprintClassifier(nn.Module):
    # Basic CNN architecture
    # Conv1D → ReLU → MaxPool → FC → Dropout → Output
```

### 2. Complex Fingerprint Classifier
```python
class ComplexFingerprintClassifier(nn.Module):
    # Advanced CNN with batch normalization
    # Multiple Conv1D layers → BatchNorm → Residual connections
```

### 3. CNN-LSTM Classifier
```python
class CNN_LSTM_Classifier(nn.Module):
    # Hybrid architecture combining CNN and LSTM
    # Conv1D → LSTM → Attention → FC layers
```

## 📊 Dataset Information

### Dataset Statistics
- **Total Samples**: 70,000 traces
- **Classes**: 3 websites
- **Distribution**: Balanced (~23,333 samples per class)
- **Input Size**: 1000 features per trace
- **Feature Type**: Network timing and packet size data

### Data Preprocessing
1. **Trace Padding/Truncation**: Normalize all traces to 1000 features
2. **Standardization**: Z-score normalization using sklearn StandardScaler
3. **Label Encoding**: Convert website URLs to integer labels

## 🔧 Configuration

### Model Configuration (`app.py`)
```python
INPUT_SIZE = 1000          # Input feature size
HIDDEN_SIZE = 128          # Hidden layer size
MODELS_DIR = "./saved_models/CNNLSTM_70k"  # Model directory
```

### Collection Configuration (`collect.py`)
```python
WEBSITES = [
    "https://cse.buet.ac.bd/moodle",
    "https://google.com", 
    "https://prothomalo.com"
]
TRACES_PER_SITE = 1        # Traces to collect per site
```

## 📈 Performance

### Model Accuracy
- **CNN-LSTM (70k dataset)**: ~84% accuracy
- **Complex Model**: ~82% accuracy  
- **Simple Model**: ~79% accuracy

### Classification Report
```
                                precision    recall  f1-score   support
https://cse.buet.ac.bd/moodle/       0.82      0.79      0.81      4699
            https://google.com       0.78      0.80      0.79      4700
        https://prothomalo.com       0.89      0.90      0.89      4700

                      accuracy                           0.83     14099
                     macro avg       0.83      0.83      0.83     14099
                  weighted avg       0.83      0.83      0.83     14099
```

## 🐛 Troubleshooting

### Common Issues

1. **Chrome Driver Issues**:
```bash
# Download compatible ChromeDriver
# Place chromedriver.exe in project root
```

2. **Model Loading Errors**:
```bash
# Check model file paths
# Ensure all model files (.pt, .pkl) exist
```

3. **Flask Server Not Running**:
```bash
# Start server before data collection
python app.py
```

4. **Memory Issues During Training**:
```python
# Reduce batch size in train.py
BATCH_SIZE = 32  # Instead of 64
```

## 📋 API Endpoints

### Flask Web Application
- `GET /` - Main web interface
- `POST /api/predict` - Predict website from trace
- `GET /api/get_results` - Retrieve stored results
- `POST /api/clear_results` - Clear all results
- `GET /api/model_info` - Get model information

## 🔬 Technical Details

### Feature Extraction
- Network traces contain timing and packet size information
- Features are normalized to handle different network conditions
- Traces are padded/truncated to fixed size (1000 features)

### Data Storage
- SQLite database for trace storage (`webfingerprint.db`)
- JSON export functionality for dataset sharing
- Automatic backup during collection

### Model Training
- Stratified train-test split (80/20)
- Early stopping with patience=5
- Learning rate scheduling
- Model checkpointing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📝 License

This project is part of an academic assignment for CSE 406 - Computer Security Sessional.

## 👥 Authors

- **Student ID**: 2005028
- **Course**: CSE 406 - Computer Security Sessional
- **Institution**: Bangladesh University of Engineering and Technology (BUET)

## 📚 References

- Website Fingerprinting Research Papers
- PyTorch Documentation
- Selenium WebDriver Documentation
- Flask Framework Documentation

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with website terms of service and privacy policies when collecting data.
