from flask import Flask, request, render_template, flash, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import sqlite3
import numpy as np
import cv2
import os
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for both flash messages and sessions

# Load Models
resnet_model = load_model('pneumonia_detector_model.h5')  # Your ResNet model
yolo_model = YOLO('best.pt')  # Your YOLO model

# Database
DATABASE = 'database.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Define folders
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
YOLO_FOLDER = 'static/yolo_results'
RESNET_FOLDER = 'static/resnet_results'

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, YOLO_FOLDER, RESNET_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def compute_image_hash(file):
    file.seek(0)
    hash_val = hashlib.sha256(file.read()).hexdigest()
    file.seek(0)
    return hash_val

# Helper function to process the image
def process_image(file):
    original_filename = file.filename
    filename_no_ext, ext = os.path.splitext(original_filename)

    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    uploaded_path = os.path.join(UPLOAD_FOLDER, original_filename)
    cv2.imwrite(uploaded_path, img)

    img_resized = cv2.resize(img, (220, 220))
    processed_img_path = os.path.join(PROCESSED_FOLDER, f'processed_{original_filename}')
    cv2.imwrite(processed_img_path, img_resized)

    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img, original_filename, uploaded_path, processed_img_path, img_array

# Helper function for ResNet prediction
def predict_resnet(img_array):
    resnet_prediction = resnet_model.predict(img_array)
    probability = float(resnet_prediction[0][0])
    resnet_class_label = 'Pneumonia' if probability > 0.5 else 'Normal'
    probability_percent = probability * 100 if resnet_class_label == 'Pneumonia' else (1 - probability) * 100
    return resnet_class_label, probability_percent

# Helper function for YOLO prediction
def predict_yolo(img):
    results = yolo_model.predict(source=img, imgsz=640, conf=0.25)
    annotated_img = results[0].plot()
    return annotated_img

# Home Page
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('predict'))
    return redirect(url_for('login'))

# Registration Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        role = request.form['role']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                'INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)',
                (username, hashed_password, email, role)
            )
            conn.commit()
            user_id = cursor.lastrowid
            session['user_id'] = user_id
            session['username'] = username
            flash('Registration successful!', 'success')
            return redirect(url_for('predict'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose a different one.', 'error')
        finally:
            conn.close()

    return render_template('register.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
# [Keep all your imports as you have them]
# ...

# Upload and Predict Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash('You must be logged in to access this page.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        try:
            img_hash = compute_image_hash(file)

            conn = get_db_connection()
            cursor = conn.cursor()

            # Check for existing image by hash
            cursor.execute('SELECT id, original_image_filename, preprocessed_image_filename FROM images WHERE image_hash = ?', (img_hash,))
            existing_image = cursor.fetchone()

            if existing_image:
                image_id = existing_image['id']
                original_filename = existing_image['original_image_filename']
                processed_img_path = os.path.join(PROCESSED_FOLDER, existing_image['preprocessed_image_filename'])

                # You may still need to preprocess to get the img_array (if not cached/stored)
                img, _, _, _, img_array = process_image(file)
            else:
                # Process new image
                img, original_filename, uploaded_path, processed_img_path, img_array = process_image(file)

                cursor.execute(
                    'INSERT INTO images (user_id, original_image_filename, preprocessed_image_filename, image_hash) VALUES (?, ?, ?, ?)',
                    (session['user_id'], original_filename, os.path.basename(processed_img_path), img_hash)
                )
                conn.commit()
                image_id = cursor.lastrowid

            # Predict using ResNet
            resnet_class_label, resnet_probability = predict_resnet(img_array)

            # Insert ResNet prediction
            cursor.execute(
                'INSERT INTO predictions (image_id, model_name, prediction_label, confidence_score) VALUES (?, ?, ?, ?)',
                (image_id, 'ResNet-50V2', resnet_class_label, resnet_probability)
            )
            conn.commit()

            # Predict using YOLO
            annotated_img = predict_yolo(img)
            yolo_img_filename = f'yolo_{original_filename}'
            yolo_img_path = os.path.join(YOLO_FOLDER, yolo_img_filename)
            cv2.imwrite(yolo_img_path, annotated_img)

            # Insert YOLO result
            cursor.execute(
                'INSERT INTO predictions (image_id, model_name, prediction_label, confidence_score) VALUES (?, ?, ?, ?)',
                (image_id, 'YOLOv11', 'Object Detection Completed', 1.0)
            )
            conn.commit()

            conn.close()

            resnet_img_filename = f'resnet50_{resnet_class_label}_{original_filename}'
            resnet_img_path = os.path.join(RESNET_FOLDER, resnet_img_filename)
            cv2.imwrite(resnet_img_path, img)

            return render_template(
                'predict.html',
                resnet_result=resnet_class_label,
                resnet_probability=resnet_probability,
                yolo_filename='yolo_results/' + yolo_img_filename,
                resnet_filename='resnet_results/' + resnet_img_filename
            )

        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
            return redirect(request.url)

    return render_template('predict.html', resnet_result=None, yolo_filename=None, resnet_filename=None)


if __name__ == '__main__':
    app.run(debug=True)
