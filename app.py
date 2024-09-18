from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
import numpy as np
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)  # Corrected from _name_ to __name__
app.secret_key = 'your_secret_key_here'  # Replace with a secure key

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model_path = r'C:\Users\Subhash\OneDrive\Desktop\calories\Calories.h5'
model = load_model(model_path)

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Labels corresponding to the classes in your model
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 
    5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 
    10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 
    15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 
    20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 
    25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 
    29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 
    33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

# Map predicted labels to calories
def fetch_calories(prediction):
    """Fetch calories based on the predicted class."""
    calorie_mapping = {
        "apple": 52, "banana": 89, "beetroot": 43, "bell pepper":39,"cabbage": 25,
        "capsicum": 40, "carrot": 41, "cauliflower": 25, "chilli pepper": 40,
        "corn": 86, "cucumber": 16, "eggplant": 25, "garlic": 149, "ginger": 80,
        "grapes": 69, "jalepeno": 29, "kiwi": 61, "lemon": 29, "lettuce": 15,
        "mango": 60, "onion": 40, "orange": 47, "paprika": 282, "pear": 57,
        "peas": 81, "pineapple": 50, "pomegranate": 83, "potato": 77, 
        "raddish": 16, "soy beans": 173, "spinach": 23, "sweetcorn": 86, 
        "sweetpotato": 86, "tomato": 18, "turnip": 28, "watermelon": 30
    }
    return calorie_mapping.get(prediction.lower(), "Calories not found")

# Check if file is allowed (valid image extension)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image to match model input requirements
def preprocess_image(img_path, target_size=(150, 150)):
    """Resize and normalize the image for prediction."""
    img = image.load_img(img_path, target_size=target_size)  # Resize the image
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Predict calories based on the uploaded image
def predict_calories(img_path):
    """Predict calories using the loaded model."""
    try:
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = labels.get(predicted_class_index, "Unknown")
        calories = fetch_calories(predicted_class)
        return calories
    except Exception as e:
        return f"Error predicting calories: {e}"

# Define the routes for the Flask app
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            calories = predict_calories(filepath)

            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Initialize session records if not set
            if 'records' not in session:
                session['records'] = []

            # Add the new record to the session
            record = {
                'filename': filename,
                'calories': calories
            }
            session['records'].append(record)

            # Calculate total calories
            total_calories = sum(int(record['calories']) for record in session['records'] if isinstance(record['calories'], (int, float)))

            return render_template('result.html', filename=filename, calories=calories, 
                                   current_datetime=current_datetime, record=session['records'], 
                                   item_calories={rec['filename']: rec['calories'] for rec in session['records']}, 
                                   total_calories=total_calories, current_date_time=current_datetime)

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

@app.route('/test')
def test():
    return "Test Page - Flask is working!"

if __name__ == '__main__':  # Corrected from _name_ to __name__
    print("Starting Flask app...")
    try:
        app.run(debug=True, host='0.0.0.0', port=5003, use_reloader=False)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
    else:
        print("Flask app started successfully.")
