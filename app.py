import time
from flask import Flask, Response, request, send_file, render_template, jsonify, send_from_directory
from matplotlib import pyplot as plt
import pandas as pd
from werkzeug.utils import secure_filename
import os
import pyttsx3
import logging
from flask_cors import CORS
import numpy as np
import cv2
import io
import seaborn as sns
import smtplib
import vonage
from datetime import datetime
from cvzone.HandTrackingModule import HandDetector
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from geopy.geocoders import ArcGIS
from googlesearch import search
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

# Directories for uploaded and processed files
UPLOAD_FOLDER = 'uploads'
FILTERED_FOLDER = 'filtered'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FILTERED_FOLDER'] = FILTERED_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FILTERED_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Index Route
@app.route('/')
def portfolio():
    return render_template("portfolio.html")

@app.route('/pythonTask')
def pythonTask():
    return render_template("index.html")

@app.route('/machineTask')
def machineTask():
    return render_template("python.html")

@app.route('/dataTask')
def dataTask():
    return render_template("data.html")
#----------------------------------------Image Creation--------------------
@app.route('/create_image', methods=['POST'])
def create_image():
    try:
        width = int(request.form['width'])
        height = int(request.form['height'])
        shape = request.form['shape'].strip().lower()
        start_x = int(request.form['start_x'])
        start_y = int(request.form['start_y'])
        end_x = int(request.form.get('end_x', 0))
        end_y = int(request.form.get('end_y', 0))
        color_b = int(request.form['color_b'])
        color_g = int(request.form['color_g'])
        color_r = int(request.form['color_r'])

        # Create image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        color = (color_b, color_g, color_r)

        if shape == 'rectangle':
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, -1)
        elif shape == 'line':
            thickness = int(request.form['thickness'])
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
        elif shape == 'circle':
            radius = int(request.form['radius'])
            cv2.circle(image, (start_x, start_y), radius, color, -1)
        else:
            return render_template('index.html', error="Shape not recognized.")

        # Convert the image to a format Flask can send
        _, img_encoded = cv2.imencode('.png', image)
        img_io = io.BytesIO(img_encoded.tobytes())
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='custom_image.png')

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")
#---------------------------------------------------Image Filtering-------------------------------------------
@app.route('/uploadfilter', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    filter_color = request.form.get('filter_color')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        output_path = os.path.join(app.config['FILTERED_FOLDER'], filename)
        
        error, status = apply_color_filter(file_path, filter_color, output_path)
        if error:
            return jsonify({"error": error}), status
        
        return send_from_directory(app.config['FILTERED_FOLDER'], filename, as_attachment=True)

def apply_color_filter(image_path, filter_color, output_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Could not load image.", 500

    filtered_image = image.copy()

    filter_color = filter_color.lower()
    if filter_color == 'red':
        filtered_image[:, :, 1] = 0
        filtered_image[:, :, 2] = 0
    elif filter_color == 'green':
        filtered_image[:, :, 0] = 0
        filtered_image[:, :, 2] = 0
    elif filter_color == 'blue':
        filtered_image[:, :, 0] = 0
        filtered_image[:, :, 1] = 0
    else:
        return "Invalid color filter.", 400

    cv2.imwrite(output_path, filtered_image)
    return None, None

#------------------------------------Face Detection and Cropping-------------------------------
@app.route('/crop-face', methods=['POST'])
def crop_face():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            output_image_path, error = process_image(file_path)
            if error:
                return jsonify({'error': error}), 500

            # Return the URL to access the image
            output_filename = os.path.basename(output_image_path)
            return jsonify({'output_image_url': f'/output/{output_filename}'})

    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

@app.route('/output/<filename>')
def send_output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, 'Error loading image'

        # Load the Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            return None, 'Error loading face cascade'

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return None, 'No face detected'

        # Crop and resize the first detected face
        (x, y, w, h) = faces[0]
        cropped_face = image[y:y+h, x:x+w]
        resized_face = cv2.resize(cropped_face, (100, 100))

        # Ensure the original image has enough space for placing the resized face
        if image.shape[0] >= 100 and image.shape[1] >= 100:
            image[0:100, 0:100] = resized_face
        else:
            return None, 'Original image size is too small to place the cropped face'

        # Generate a unique filename for the output
        output_filename = f"{int(time.time() * 1000)}.jpg"
        output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_image_path, image)

        return output_image_path, None

    except Exception as e:
        return None, str(e)

#--------------------------------sunglass filter--------------------------------------------
# Configuration for file upload
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'output/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def apply_sunglasses_filter(image_path):
    """Apply a sunglasses filter to the image."""
    try:
        image = cv2.imread(image_path)
        sunglasses_img = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)

        if sunglasses_img is None:
            return None, 'Sunglasses image not found'

        # Resize sunglasses to fit the face region (assuming full image for now)
        sunglasses_img = cv2.resize(sunglasses_img, (image.shape[1], image.shape[0]))
        
        if sunglasses_img.shape[2] < 4:
            return None, 'Sunglasses image does not have an alpha channel'

        # Separate alpha channel from the sunglasses image
        alpha_channel = sunglasses_img[:, :, 3] / 255.0
        sunglasses_img = sunglasses_img[:, :, :3]

        # Blend the sunglasses with the original image
        for c in range(0, 3):
            image[:, :, c] = (1.0 - alpha_channel) * image[:, :, c] + alpha_channel * sunglasses_img[:, :, c]

        # Save the result
        output_filename = f"sunglass_applied_{os.path.basename(image_path)}"
        output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_image_path, image)

        return output_image_path, None

    except Exception as e:
        return None, str(e)

@app.route('/apply_sunglasses_filter', methods=['POST'])
def apply_sunglasses_filter_route():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            output_image_path, error = apply_sunglasses_filter(file_path)
            if error:
                return jsonify({'error': error}), 500

            # Return the URL to access the image
            output_filename = os.path.basename(output_image_path)
            return jsonify({'output_image_url': f'/output/{output_filename}'})

        return jsonify({'error': 'File type not allowed'}), 400

    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

# Serve output images
@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

#---------------------------------location ----------------------------------------------
# Initialize the geolocator
nom = ArcGIS()

@app.route('/geocode', methods=['POST'])
def geocode_location():
    data = request.json
    location_input = data.get('location')

    if not location_input:
        return jsonify({"error": "Please provide a location."}), 400

    # Geocode the user input
    location = nom.geocode(location_input)

    # Check if the location was found
    if location:
        return jsonify({
            "address": location.address,
            "latitude": location.latitude,
            "longitude": location.longitude
        })
    else:
        return jsonify({"error": "Location not found."}), 404
    
#---------------------top5 searches---------------------------
@app.route('/search', methods=['GET'])
def google_search():
    # Get the query parameter from the URL
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    # Perform the search and fetch top 5 results
    top_5_results = search(query, num_results=5)

    # Prepare the response as a list of dictionaries
    results = [{"rank": i+1, "url": result} for i, result in enumerate(top_5_results)]

    return jsonify(results)
#----------------------------------------text to speech----------------------------------
# Initialize the text-to-speech engine
text_speech = pyttsx3.init()

# Optional: Configure the logging level to reduce verbosity
import logging
logging.getLogger('comtypes').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.INFO)

# Route for text-to-speech conversion
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Generate speech
        audio_file = "output.mp3"
        text_speech.save_to_file(text, audio_file)
        text_speech.runAndWait()
        
        # Ensure the file exists before sending it
        if os.path.exists(audio_file):
            return send_file(audio_file, as_attachment=True)
        else:
            return jsonify({"error": "Audio file was not created"}), 500
        
    except Exception as e:
        # Handle exceptions and provide an error message
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Set the environment variable to disable oneDNN custom operations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#-------------------------------audio control------------------------------
# Initialize the audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

@app.route('/get-volume', methods=['GET'])
def get_volume():
    current_volume = volume.GetMasterVolumeLevelScalar()
    return jsonify({"volume": round(current_volume * 100, 2)})

@app.route('/set-volume', methods=['POST'])
def set_volume():
    data = request.get_json()
    new_volume = data.get('volume')
    if new_volume is not None and 0.0 <= new_volume <= 1.0:
        volume.SetMasterVolumeLevelScalar(new_volume, None)
        return jsonify({"message": f"Volume set to {new_volume * 100:.2f}%"})
    else:
        return jsonify({"error": "Invalid volume level. It should be between 0.0 and 1.0."}), 400

@app.route('/mute', methods=['POST'])
def mute():
    volume.SetMute(1, None)
    return jsonify({"message": "Audio muted"})

@app.route('/unmute', methods=['POST'])
def unmute():
    volume.SetMute(0, None)
    return jsonify({"message": "Audio unmuted"})

@app.route('/get-mute-status', methods=['GET'])
def get_mute_status():
    mute_status = volume.GetMute()
    return jsonify({"mute_status": "Muted" if mute_status else "Unmuted"})
#--------------------------email-sent----------------------
def send_email(smtp_server, smtp_port, sender_email, receiver_email, password, subject, body):
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(message)
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {e}"

@app.route('/send-email', methods=['POST'])
def send_email_api():
    data = request.get_json()
    
    smtp_server = data.get('smtp_server')
    smtp_port = data.get('smtp_port')
    sender_email = data.get('sender_email')
    receiver_email = data.get('receiver_email')
    password = data.get('password')
    subject = data.get('subject')
    body = data.get('body')
    
    if not (smtp_server and smtp_port and sender_email and receiver_email and password and subject and body):
        return jsonify({"error": "Missing required fields"}), 400
    
    result = send_email(smtp_server, smtp_port, sender_email, receiver_email, password, subject, body)
    
    if "successfully" in result:
        return jsonify({"message": result})
    else:
        return jsonify({"error": result}), 500
#------------------------do a sms using voanage---------------------
# Initialize the Vonage client
client = vonage.Client(key="key", secret="your secret key")
sms = vonage.Sms(client)

@app.route('/send_sms', methods=['POST'])
def send_sms():
    # Get the JSON data from the request
    data = request.get_json()
    from_number = "RUNCODES"  # Sender number, usually a fixed value or a valid number in your account
    to_number = data.get('to')
    message_text = data.get('text')

    if not to_number or not message_text:
        return jsonify({'error': 'Missing phone number or message text'}), 400

    # Send the SMS
    responseData = sms.send_message(
        {
            "from": from_number,
            "to": to_number,
            "text": message_text,
        }
    )

    # Check the response status
    if responseData["messages"][0]["status"] == "0":
        return jsonify({'status': 'Message sent successfully.'}), 200
    else:
        return jsonify({'error': responseData['messages'][0]['error-text']}), 500
#-----------------------------------------hand detector----------------------------
# Initialize the Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect hands
        hands, img = detector.findHands(frame)

        # Process the hands detected
        if hands:
            for hand in hands:
                handType = hand["type"]
                lmList = hand["lmList"]  # List of 21 landmarks
                bbox = hand["bbox"]  # Bounding box info x,y,w,h
                centerPoint = hand["center"]  # Center of the hand cx,cy
                handLabel = handType

                # Display hand type on the frame
                cv2.putText(img, handLabel, (centerPoint[0] - 50, centerPoint[1] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display gesture information
                fingers = detector.fingersUp(hand)
                if fingers == [1, 0, 0, 0, 0]:
                    gesture = "Thumb Up"
                elif fingers == [0, 1, 0, 0, 0]:
                    gesture = "Index Finger Up"
                elif fingers == [0, 0, 1, 0, 0]:
                    gesture = "Middle Finger Up"
                elif fingers == [0, 0, 0, 1, 0]:
                    gesture = "Ring Finger Up"
                elif fingers == [0, 0, 0, 0, 1]:
                    gesture = "Little Finger Up"
                elif fingers == [1, 1, 1, 1, 1]:
                    gesture = "All Fingers Up"
                else:
                    gesture = "Unknown Gesture"

                # Display the gesture on the frame
                cv2.putText(img, gesture, (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        # Yield the frame in byte format for the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_gesture', methods=['GET'])
def detect_gesture():
    ret, frame = cap.read()
    if not ret:
        return jsonify({'error': 'Unable to capture frame'}), 500

    hands, img = detector.findHands(frame)

    if hands:
        gesture_info = []
        for hand in hands:
            fingers = detector.fingersUp(hand)
            if fingers == [1, 0, 0, 0, 0]:
                gesture = "Thumb Up"
            elif fingers == [0, 1, 0, 0, 0]:
                gesture = "Index Finger Up"
            elif fingers == [0, 0, 1, 0, 0]:
                gesture = "Middle Finger Up"
            elif fingers == [0, 0, 0, 1, 0]:
                gesture = "Ring Finger Up"
            elif fingers == [0, 0, 0, 0, 1]:
                gesture = "Little Finger Up"
            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "All Fingers Up"
            else:
                gesture = "Unknown Gesture"

            gesture_info.append({
                'handType': hand["type"],
                'gesture': gesture
            })

        return jsonify(gesture_info), 200
    else:
        return jsonify({'message': 'No hands detected'}), 200
#-----------------------------------data processing--------------------------
# Global variable to store the loaded data
data = None

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    global data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        try:
            data = pd.read_csv(file)
            return jsonify({'message': 'File loaded successfully'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/summary', methods=['GET'])
def summary_statistics():
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    summary = data.describe(include='number').transpose()
    summary = summary[['count', 'mean', 'std', 'min', 'max']]
    return summary.to_json(orient='index')

@app.route('/visualize_histogram', methods=['GET'])
def visualize_histogram():
    column = request.args.get('column')
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    if column not in data.columns:
        
        return jsonify({'error': 'Column not found'}), 400

    plt.figure()
    data[column].hist()
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return send_file(img, mimetype='image/png', as_attachment=True, download_name=f'{column}_histogram.png')

@app.route('/visualize_boxplot', methods=['GET'])
def visualize_boxplot():
    column = request.args.get('column')
    if data is None:
        return jsonify({'error': 'No data loaded'}), 400
    if column not in data.columns:
        return jsonify({'error': 'Column not found'}), 400

    plt.figure()
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png', as_attachment=True, download_name=f'{column}_boxplot.png')
if __name__ == '__main__':
    app.run()