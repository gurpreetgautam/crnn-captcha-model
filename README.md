# CRNN CAPTCHA Recognition Model

A deep learning project implementing a Convolutional Recurrent Neural Network (CRNN) with Bidirectional LSTM for CAPTCHA recognition. The model is trained on synthetically generated CAPTCHA datasets to recognize text in distorted images.

## Features

- **Synthetic CAPTCHA Generation**: Generate two types of CAPTCHAs:
  - State Portal CAPTCHAs: Blue circles background with black text, squares, and lines
  - CPP CAPTCHAs: Light background with random fonts, colors, and noise
- **CRNN Model**: Convolutional layers for feature extraction + Bidirectional LSTM for sequence recognition
- **CTC Loss**: Connectionist Temporal Classification for sequence-to-sequence learning without alignment
- **Real-time Prediction**: Load trained model and predict CAPTCHA text from images

## Project Structure

```
crnn-captcha-model/
├── generator.py              # CAPTCHA generation scripts
├── predictor.py              # Model prediction interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── captcha_dataset/          # Generated CAPTCHA images
│   ├── cpp_captchas/         # CPP-style CAPTCHAs
│   └── state_portal_captchas/ # State portal-style CAPTCHAs
├── fonts/                    # Font files for CAPTCHA generation (contains all fonts used to create the CAPTCHA dataset)
└── MODEL/                    # Trained model and training code
    ├── __init__.py
    ├── CRNN_best_model.pth   # Trained PyTorch model weights
    └── crnn.ipynb            # Jupyter notebook with training code
```

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

- Python 3.8+
- PyTorch 2.11.0
- CUDA (optional, for GPU acceleration)
- PIL (Pillow)
- NumPy
- Matplotlib
- Other dependencies listed in `requirements.txt`

## Usage

### 1. Generate CAPTCHA Dataset

The `generator.py` script creates synthetic CAPTCHAs for training the model. It generates two distinct CAPTCHA styles to improve model robustness.

#### CAPTCHA Types

**State Portal CAPTCHAs:**
- Background: White with blue circles (30-80 circles, radius 0.3-0.5)
- Text: Black Arial font, 6 random alphanumeric characters
- Noise: Black circles (10-20, radius 1-4), black squares (25-35, side 2-7), 2 random colored lines
- Image size: 200x50 pixels

**CPP CAPTCHAs:**
- Background: Random light color (230-255 RGB)
- Text: 6 random alphanumeric characters, random fonts from `fonts/` directory
- Font features: Random colors, slight rotation (-15° to 15°), optional shadow effect
- Noise: Random colored points (150 per shadow iteration, 3-7 iterations), background color points (100)
- Image size: 200x50 pixels

#### Usage

Run the generator script:

```python
python generator.py
```

This generates and saves CAPTCHA images in `captcha_dataset/cpp_captchas/` and `captcha_dataset/state_portal_captchas/`. By default, it creates 3 images of each type. Modify the `NUM_IMAGES` variable in the script to generate more.

#### Configuration

Key parameters in `generator.py`:
- `WIDTH = 200` - Image width
- `HEIGHT = 50` - Image height
- `FONT_SIZE = 40` - Base font size
- `CHAR_SET = string.ascii_letters + string.digits` - Character set for text generation
- Font paths loaded from `fonts/` directory

### 2. Train the Model

Open and run the Jupyter notebook `MODEL/crnn.ipynb` to train the CRNN model on the generated dataset. The notebook includes:

- Dataset loading and preprocessing
- Model architecture definition
- Training loop with CTC loss
- Validation and early stopping
- Model saving

### 3. Predict CAPTCHA Text

Use the predictor to recognize text in CAPTCHA images:

```python
from predictor import predict_captcha

result = predict_captcha("path/to/captcha_image.png")
print(f"Predicted text: {result}")
```

The model automatically loads the trained weights from `MODEL/CRNN_best_model.pth`.

## Model Architecture

- **CNN Backbone**: 7 convolutional layers with batch normalization, ReLU activation, max pooling, and dropout
- **RNN Layers**: 2 Bidirectional LSTM layers (256 hidden units each)
- **Output**: 63 classes (26 lowercase + 26 uppercase + 10 digits + 1 blank for CTC)
- **Input Size**: 32x128 grayscale images
- **Loss Function**: Connectionist Temporal Classification (CTC)

## Dataset

The model is trained on synthetically generated CAPTCHAs with:
- Random 6-character strings (letters + digits)
- Various fonts, colors, and distortions
- Noise elements (circles, squares, lines, points)
- Two different CAPTCHA styles for robustness

## Performance

The trained model achieves high accuracy on the synthetic dataset. Performance may vary on real-world CAPTCHAs depending on similarity to training data.

## Contributing

Feel free to contribute by:
- Improving the CAPTCHA generation algorithms
- Enhancing the model architecture
- Adding more CAPTCHA types
- Optimizing training procedures

## License

This project is open-source. Please check individual file headers for specific licensing information.

## Acknowledgments

- Based on CRNN architecture for text recognition
- Uses PyTorch for deep learning framework
- PIL for image processing
