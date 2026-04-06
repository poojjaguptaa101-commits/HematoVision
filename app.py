import os
import numpy as np
import cv2 # OpenCV for image processing
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import base64 # For encoding image to base64
import io # For handling image bytes

# Initialize Flask application
app = Flask(__name__)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
# IMPORTANT: Ensure the path and name of your .h5 model file is correct.
# It should be in the same directory as app.py.
# If your model was saved as 'blood_cell.h5' from the training step, use that name.
MODEL_PATH = 'blood_cell.h5' # Using 'blood_cell.h5' as saved by ModelCheckpoint
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully!")
except Exception as e:
    print(f"Error loading model '{MODEL_PATH}': {e}")
    print("Please ensure the model file exists and the path is correct.")
    model = None # Set model to None if loading fails

# Define the class labels in the order your model was trained on
# This order is usually alphabetical if flow_from_dataframe was used without explicit class_mode
CLASS_NAMES = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict image class
def predict_image_class(image_bytes, model):
    # Convert image bytes to numpy array
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR) # Read image using OpenCV

    # Convert BGR to RGB (OpenCV reads images in BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to target size for the model
    img_resized = cv2.resize(img_rgb, (244, 244))

    # Preprocess the image for the model
    # Add batch dimension and apply MobileNetV2 preprocessing
    img_preprocessed = preprocess_input(img_resized.reshape(1, 244, 244, 3))

    # Make prediction
    predictions = model.predict(img_preprocessed)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_label = CLASS_NAMES[predicted_class_idx]

    # Return predicted label and the original RGB image (for display)
    return predicted_class_label, img_rgb

# Route for the home page and handling predictions
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if model is None:
            return "Error: Model not loaded. Please check server logs.", 500

        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Read image data from the uploaded file
            image_bytes = file.read()

            # Predict the class and get the image for display
            predicted_class_label, img_for_display = predict_image_class(image_bytes, model)

            # Convert the image for display back to BGR for OpenCV encoding, then to base64
            # This is to ensure consistent color channels if you want to display the processed image
            # If you want to display the *original* uploaded image, you'd use image_bytes directly
            _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_for_display, cv2.COLOR_RGB2BGR))
            img_str = base64.b64encode(img_encoded).decode('utf-8')

            # Render the result page with prediction and base64 image
            return render_template("result.html", prediction=predicted_class_label, image_data=img_str)
        else:
            return "Invalid file type. Please upload an image (png, jpg, jpeg, gif).", 400
    # If it's a GET request, render the home page
    return render_template("home.html")

# Main function to run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

