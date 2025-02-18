# movie-review-sentiment-analyzer

This repository implements a sentiment analysis project focused on movie reviews. It combines a production-ready web service with a custom training pipeline that leverages model distillation techniques to provide accurate sentiment predictions using state-of-the-art transformer models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup & Installation](#setup--installation)
- [Running the Web App](#running-the-web-app)
- [Training & Model Distillation](#training--model-distillation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Future Enhancements](#future-enhancements)

## Overview

This project consists of two main components:

1. **Web Service**  
   A Flask-based API (`app.py`) that serves sentiment predictions. The API is complemented by a modern, responsive front-end (`index.html`) built with Tailwind CSS, allowing users to easily submit movie reviews and view analysis results.

2. **Training Pipeline**  
   A Python script (`sentiment_analysis.py`) that demonstrates a model distillation approach. A larger teacher model guides a smaller student model using a hybrid loss function—combining distillation loss with task-specific cross-entropy loss. This pipeline uses a subset of the IMDB dataset for training and evaluation.

## Features

- **Real-time Sentiment Prediction:**  
  Analyze movie reviews on-the-fly with sentiment labels (positive/negative) and associated confidence scores.

- **Model Distillation:**  
  Efficient training of a lightweight student model using knowledge distillation from a larger teacher model.

- **User-friendly Interface:**  
  A sleek web interface enables users to input reviews and see animated progress bars representing prediction probabilities.

- **Easy Deployment:**  
  A simple Flask server setup and static HTML front-end make local deployment and testing straightforward.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/movie-review-sentiment-analyzer.git
   cd movie-review-sentiment-analyzer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
If `requirements.txt` is not available, manually install:
```bash
pip install flask flask-cors torch transformers datasets tqdm
```

## Running the Web App

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Access the web interface:**
   
   Open your browser and navigate to http://127.0.0.1:5000 to start analyzing movie reviews.

## Training & Model Distillation

The `sentiment_analysis.py` script demonstrates how to train a student model using model distillation:

### Data Loading & Preprocessing
- Utilizes a subset of the IMDB dataset, tokenized with Hugging Face’s tokenizer.

### Model Initialization
- Loads a pre-trained teacher model (e.g., `distilbert-base-uncased-finetuned-sst-2-english`) and a smaller student model (e.g., `prajjwal1/bert-mini`).
- The teacher's weights are frozen to guide the student during training.

### Distillation Training Loop
- Combines a distillation loss (using KL divergence) with standard cross-entropy loss to train the student model efficiently.

### Checkpointing & Saving
- Intermediate checkpoints are saved during training.
- The final model is stored in a Hugging Face-compatible format in the `final_sentiment_model` directory.

### Running the Training Pipeline
To run the training pipeline, uncomment the code inside the `main()` function and execute:

```bash
python sentiment_analysis.py
```

## Usage

### API Endpoint
The `/predict` endpoint accepts POST requests with a JSON body containing the field `review_text`:

```json
{
  "review_text": "The movie was fantastic and engaging!"
}
```

## Troubleshooting

### Server Not Starting
- Ensure that the Flask dependencies are installed and that your environment's Python version is compatible (Python 3.7+).
- Verify that there are no port conflicts on port 5000.

### Model Loading Issues
- If you encounter errors regarding model paths, double-check the path specified in `app.py` for loading the sentiment analysis model.
- Update the path to point to your local model directory.

### API Errors
- For API-related issues, inspect the terminal logs where the Flask server is running.
- Errors during prediction are logged, and you may need to adjust memory settings or device configurations (CPU vs CUDA).

## Dependencies

- Python 3.7+
- Flask & Flask-CORS
- PyTorch
- Hugging Face Transformers
- Datasets (for loading the IMDB dataset)
- TQDM (for progress tracking during training)

## Contributing

Contributions, issues, and feature requests are welcome! Please check the [issues page](https://github.com/yourusername/movie-review-sentiment-analyzer/issues) for more details.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- Hugging Face for their transformer models and tokenizers.
- Contributors of the IMDB dataset.
- The open-source community for their valuable tools and libraries.

## Future Enhancements

- **Model Improvements:**  
  Explore additional pre-trained models and ensemble techniques to further improve sentiment prediction accuracy.

- **Extended Functionality:**  
  Integrate more nuanced sentiment categories (e.g., neutral sentiment) and extend support to reviews from domains other than movies.

- **Deployment:**  
  Consider containerizing the application using Docker and deploying on cloud platforms for scalability.

- **User Interface Enhancements:**  
  Improve the web interface with additional features such as a history of analyzed reviews and real-time analytics.
