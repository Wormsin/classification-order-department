# Order Classification System

## Overview
This project is designed to automatically classify incoming orders and assign them to the appropriate department within a company. The classification process is powered by a fine-tuned `ruBERT` model, which has demonstrated high accuracy due to extensive text analysis conducted prior to training.

## Features
- Uses the `ruBERT` model for text classification
- Implements a web-based interface using `Dash` for user interaction
- Computes prediction entropy to indicate confidence in classification
- Displays classification probabilities for each department

## Model Details
The classification model is based on `ruBERT` and is fine-tuned for order classification. The model:
- Tokenizes input text using `AutoTokenizer`
- Processes text using `AutoModelForSequenceClassification`
- Outputs probabilities for each department
- Computes entropy to assess prediction confidence

## Demo Application
The project includes a demo application (`demo.py`) built with `Dash`. The application allows users to input order details and receive department classification predictions along with entropy scores.

### How It Works
1. The user enters order details in the input field.
2. Upon clicking the "Predict" button, the input is tokenized and processed by the `ruBERT` model.
3. The model outputs probability scores for each department.
4. The entropy of the prediction is computed to assess confidence.
5. The results are displayed with high-probability predictions highlighted in red.

## Installation & Requirements
### Dependencies
Ensure you have the following Python packages installed:
- `dash`
- `dash-bootstrap-components`
- `plotly`
- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `transformers`

Install dependencies using:
```sh
pip install dash dash-bootstrap-components plotly numpy pandas scikit-learn torch transformers
```

### Running the Demo Application
1. Place the trained model files in the appropriate directory.
2. Ensure `cleaned_data.csv` is available in the project directory.
3. Run the application with:
   ```sh
   python demo.py
   ```
4. Open `http://127.0.0.1:8051/` in a web browser to interact with the application.

## Future Improvements
- Enhance model accuracy with additional training data
- Improve UI/UX for better usability
- Expand to support multiple languages

## Contact
For questions or contributions, please reach out to the project team.


