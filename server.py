import os
from flask import Flask, request, redirect, url_for, send_from_directory, flash, render_template
from werkzeug.utils import secure_filename
import cv2
from docdetect import *
from test import unetProcessing
from compress import *

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpeg', 'jpg', 'tif'])

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file_types(filename):
    return '.' in filename and \
            filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
    
    
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
        
        if file and allowed_file_types(file.filename):
            filename = secure_filename(file.filename)
            noOfFiles = len([name for name in os.listdir('uploads/') if os.path.isfile(os.path.join('uploads/', name))])

            print("No of files: ", noOfFiles)
            filetype = filename.split('.')[1]
            filename = str(0) + '.' + filetype
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('Uploaded successfully', filename)
            compress(filename)
            unetProcessing()
            main(filename)
            # filename = filename.split('.')[0]
            # filename = filename + '.jpg'
            full_filename = os.path.join('static', 'FinalTransformedDoc', '0.jpg')
            print(full_filename)
            return render_template("home.html", document = full_filename)

if __name__ == '__main__':
    app.run(port=3000)