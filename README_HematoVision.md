
# HematoVision: Advanced Blood Cell Classification Using Transfer Learning

## Team Details

- **Team ID**: LTVIP2025TMID41359  
- **Team Size**: 4  
- **Team Leader**: Golla Jahnavi  
- **Team Members**:  
  - Gujjala Pranay Kumar  
  - Irakam Siva Venkata Bhanu Prakash  
  - Johan Abhishek  

---

## Project Overview

HematoVision is an innovative project aimed at developing an accurate and efficient model for classifying blood cells. It leverages **transfer learning** techniques with pre-trained Convolutional Neural Networks (CNNs) to expedite the training process and significantly enhance classification accuracy. The project provides a reliable and scalable tool for pathologists and healthcare professionals, improving the precision and efficiency of blood cell analysis.

## Features

- **Accurate Blood Cell Classification**: Classifies four distinct types of blood cells: Eosinophil, Lymphocyte, Monocyte, and Neutrophil.  
- **Transfer Learning**: Utilizes a pre-trained MobileNetV2 model to achieve high accuracy with reduced training time and computational resources.  
- **Web Application Interface**: A user-friendly Flask-based web application for easy image upload and real-time prediction display.  
- **Production-Ready**: Designed for deployment, with a clear project structure and optimized code.

## Project Structure

```
HematoVision_App/
├── app.py                 # Main Flask application script
├── requirements.txt       # Python dependencies
├── blood_cell.h5          # Trained MobileNetV2 model file
├── templates/             # HTML templates for the web interface
│   ├── home.html
│   └── result.html
└── static/
    └── uploads/           # Directory for temporarily storing uploaded images
```

## Installation and Setup (Local)

### Prerequisites

- Python 3.8+
- pip
- Git

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gujjala-pranay/hematovision-app.git
   cd hematovision-app
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   ```

   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open in browser**: Visit `http://127.0.0.1:5000/`

## Usage

- **Step 1**: Upload a blood cell image (PNG, JPG, JPEG, or GIF)
- **Step 2**: Click the "Classify Blood Cell" button
- **Step 3**: View the classification result with the image

## Model Details

- **Architecture**: MobileNetV2 with custom classification layers
- **Dataset**: 12,500 augmented blood cell images from Kaggle
- **Training**: 5 epochs, Adam optimizer, categorical cross-entropy
- **Accuracy**: ~85.3% validation accuracy
- **Model File**: `blood_cell.h5`

## Deployment

Ready for deployment on platforms like Render, Railway, or Heroku. Refer to the `HematoVision_Deployment_Guide.md` for full steps.

## Contributing

Fork the repository, create pull requests, or submit issues to contribute.
