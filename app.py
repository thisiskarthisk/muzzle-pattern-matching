import cv2
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy



app = Flask(__name__)

# PostgreSQL Configuration
DB_USER = "odoo_user"
DB_PASSWORD = "letmein1!"
DB_NAME = "muzzle_matching_animal"
DB_HOST = "localhost"

app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
MATCHED_FOLDER = 'static/matched'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MATCHED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MATCHED_FOLDER'] = MATCHED_FOLDER



# Database Model
class MuzzleMatch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original1 = db.Column(db.String(255), nullable=False)
    original2 = db.Column(db.String(255), nullable=False)
    muzzle1 = db.Column(db.String(255), nullable=False)
    muzzle2 = db.Column(db.String(255), nullable=False)
    matched = db.Column(db.String(255), nullable=False)
    match_percentage = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.TIMESTAMP, server_default=db.func.now())

# Create DB Tables
with app.app_context():
    db.create_all()

# **Function: Save Uploaded Images**
def save_uploaded_images(image1, image2):
    image1_path = os.path.join(UPLOAD_FOLDER, image1.filename)
    image2_path = os.path.join(UPLOAD_FOLDER, image2.filename)
    image1.save(image1_path)
    image2.save(image2_path)
    return image1_path, image2_path



def extract_muzzle(image_path, output_filename):
    """Extracts and enhances the muzzle pattern from an animal image."""
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Assume the muzzle is in the lower 50% of the image
    muzzle_area = img[int(height * 0.5):height, :]

    # Convert to grayscale
    gray = cv2.cvtColor(muzzle_area, cv2.COLOR_BGR2GRAY)

    # Enhance pattern using Adaptive Thresholding
    enhanced_muzzle = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    processed_path = os.path.join(PROCESSED_FOLDER, output_filename)
    cv2.imwrite(processed_path, enhanced_muzzle)

    return processed_path

def match_muzzle_patterns(image1_path, image2_path, output_filename):
    """Matches and visualizes key points between two muzzle patterns, returns matching percentage."""
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # ORB Detector
    orb = cv2.ORB_create(500)  # Increased number of keypoints

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None, "No keypoints found in one or both images"

    # BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate Matching Percentage
    total_keypoints = max(len(kp1), len(kp2))
    match_percentage = (len(matches) / total_keypoints) * 100 if total_keypoints > 0 else 0

    # Draw first 20 matches
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    matched_path = os.path.join(MATCHED_FOLDER, output_filename)
    cv2.imwrite(matched_path, matched_img)

    return matched_path, f"{match_percentage:.2f}%"

@app.route('/')
def index():
    return render_template('index.html')



# @app.route('/upload', methods=['POST'])
# def upload_images():
#     if 'image1' not in request.files or 'image2' not in request.files:
#         return jsonify({'error': 'Please upload two images'}), 400

#     image1 = request.files['image1']
#     image2 = request.files['image2']

#     image1_path = os.path.join(UPLOAD_FOLDER, image1.filename)
#     image2_path = os.path.join(UPLOAD_FOLDER, image2.filename)
#     image1.save(image1_path)
#     image2.save(image2_path)

    # muzzle1_path = extract_muzzle(image1_path, "muzzle1.png")
    # muzzle2_path = extract_muzzle(image2_path, "muzzle2.png")
    # matched_path, match_percentage = match_muzzle_patterns(muzzle1_path, muzzle2_path, "matched.png")

#     if matched_path is None:
#         return jsonify({'error': match_percentage}), 400

#     return jsonify({
#         'original1': image1_path,
#         'original2': image2_path,
#         'muzzle1': muzzle1_path,
#         'muzzle2': muzzle2_path,
#         'matched': matched_path,
#         'match_percentage': match_percentage
#     })


# **API: Upload Images and Process**
@app.route("/upload", methods=["POST"])
def upload_images():
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Please upload two images"}), 400

    image1, image2 = request.files["image1"], request.files["image2"]
    image1_path, image2_path = save_uploaded_images(image1, image2)

    # muzzle1_path = extract_muzzle(image1_path, "muzzle1_" + image1.filename)
    # muzzle2_path = extract_muzzle(image2_path, "muzzle2_" + image2.filename)
    # matched_path, match_percentage = match_muzzle_patterns(muzzle1_path, muzzle2_path, "matched_" + image1.filename)
    muzzle1_path = extract_muzzle(image1_path, "muzzle1.png")
    muzzle2_path = extract_muzzle(image2_path, "muzzle2.png")
    matched_path, match_percentage = match_muzzle_patterns(muzzle1_path, muzzle2_path, "matched.png")


    if matched_path is None:
        return jsonify({"error": match_percentage}), 400

    # Save Match Info to DB
    new_match = MuzzleMatch(
        original1=image1_path,
        original2=image2_path,
        muzzle1=muzzle1_path,
        muzzle2=muzzle2_path,
        matched=matched_path,
        match_percentage=match_percentage,
    )
    db.session.add(new_match)
    db.session.commit()

    return jsonify({
        "original1": image1_path,
        "original2": image2_path,
        "muzzle1": muzzle1_path,
        "muzzle2": muzzle2_path,
        "matched": matched_path,
        "match_percentage": match_percentage
    })

# **API: Get Match History**
@app.route("/history", methods=["GET"])
def get_history():
    matches = MuzzleMatch.query.order_by(MuzzleMatch.timestamp.desc()).all()
    return jsonify([{
        "id": match.id,
        "original1": match.original1,
        "original2": match.original2,
        "muzzle1": match.muzzle1,
        "muzzle2": match.muzzle2,
        "matched": match.matched,
        "match_percentage": match.match_percentage,
        "timestamp": match.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    } for match in matches])


if __name__ == '__main__':
    app.run(debug=True,port=5003)
