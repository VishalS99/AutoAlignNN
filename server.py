# Flask dependencies
from flask import Flask, request, redirect, flash, render_template
from werkzeug.utils import secure_filename
# Image processing dependencies
import cv2
import os
# Processing files
from docdetect import *
from test import unetProcessing
from compress import *
from harsh_thresh import *

UPLOAD_FOLDER = 'uploads/'

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# File type validation
ALLOWED_EXTENSIONS = set(['png', 'jpeg', 'jpg', 'tif'])
def allowed_file_types(filename):
    return '.' in filename and \
            filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
    

# Initial Route
@app.route('/',methods=['GET'])
def form():
    print(app.root_path)
    return render_template('home.html')


@app.route('/doc', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No document uploaded!!')
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            flash('Invalid Document!!')
            return redirect(request.url)
        # Validation
        if file and allowed_file_types(file.filename):
            filename = secure_filename(file.filename)
            filetype = filename.split('.')[1]
            filename = str(0) + '.' + filetype
            # Upload file into /uploads
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('Uploaded successfully', filename)

            # Check for condition where document is a device screenshot
            print("[INFO] Compressing image to (256,256)... ")
            if os.stat('uploads/'+filename).st_size < 500000:
                thresh_filename = harsh_thresh('uploads/'+filename)
                compress(thresh_filename)

            else: compress(filename)

            # Masking
            unetProcessing()
            # Document extraction
            docdetect(filename)
            # Path to saved processed files
            full_filename = os.path.join('static', 'FinalTransformedDoc', '0.jpg')
            print(full_filename)
            return render_template("home.html", document = full_filename)

if __name__ == '__main__':
    app.run(port=3000)