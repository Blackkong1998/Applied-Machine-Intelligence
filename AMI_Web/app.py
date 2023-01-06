import io
import os
import json
from base64 import b64encode

from PIL import Image
from flask import Flask, flash, request, redirect, render_template, Response, jsonify
from werkzeug.utils import secure_filename
import sys

webapp_dir = os.path.dirname(__file__)  # path to AMI_Web directory
main_dir = os.path.dirname(webapp_dir)  # path to Group07 directory

# Group07 directory needs to be appended to path, otherwise other modules from other python files cannot be found
sys.path.append(main_dir)

from continious_learning.Learner import Learner
from inference import TrainedModel
from db_integration.DBManager import DBManager

# -------------------------------- #
#
#
#
# IMPORTANT. MAKE SURE YOUR CURRENT WORKING DIRECTORY IS AMI_WEB
# Otherwise the upload folder will not be the correct path
cwd = os.getcwd()
if cwd != webapp_dir:
    os.chdir(webapp_dir)
#
# -------------------------------- #

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'webp', 'png'}

app = Flask(__name__, template_folder="templates")
app.secret_key = "secret key"

# initialize the trained model
global model
model = TrainedModel()


def allowed_file(filename: str):
    """Check if the file is an image with the correct extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_json_file(label_dict: dict):
    """Create a json file for the annotation."""
    # label dict is a dictionary with the filename as key, and the annotated class as value
    formatted_labeldict = dict()
    formatted_labeldict["annotations"] = []
    for filename, annotation_class in label_dict.items():
        tmp_dict = {"file_name": filename, "label": annotation_class}
        formatted_labeldict["annotations"].append(tmp_dict)

    labeljson = json.dumps(formatted_labeldict, indent=4)

    return labeljson


@app.route('/')
def home():
    """Load Home page."""
    return render_template('Home.html')


file_names_uploadlabel = []
labels_uploadlabel = dict()
export_done = True
displayed = []
images_dict_label_annotation = dict()
images_dict_label_annotation_prediction = dict()

path_db = os.path.join(webapp_dir, "data")

if not os.path.exists(path_db):
    os.makedirs(path_db)

DB_MANAGER = DBManager(path=path_db,
                       db_name="db_labels.db",
                       table_name="labels")


@app.route('/uploadlabel', methods=['GET', 'POST'])
def uploadlabel():
    """Load Upload and Label page."""
    global file_names_uploadlabel
    global labels_uploadlabel
    global export_done
    global displayed
    global images_dict_label_annotation

    new_files = []

    wrong_extension = False
    if request.method == 'POST':
        if request.form['action'] == 'uploadfiles':
            # check if the post request has the file part
            if 'files[]' not in request.files:
                flash('No file part')
                return redirect(request.url)
            files = request.files.getlist('files[]')

            # if user does not select file, browser also
            # submit an empty part without filename
            for file in files:
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if not allowed_file(file.filename):
                    wrong_extension = True
                if file and allowed_file(file.filename):
                    export_done = False
                    filename = secure_filename(file.filename)
                    if filename not in file_names_uploadlabel:
                        file_names_uploadlabel.append(filename)

                        images_dict_label_annotation[filename] = file.read()
                        new_files.append(file)
                        flash(filename)

        elif request.form['action'] == 'resetfiles':
            file_names_uploadlabel = []
            labels_uploadlabel = dict()
            images_dict_label_annotation = dict()
            export_done = True
            displayed = []

    if file_names_uploadlabel or wrong_extension:
        images = [Image.open(file) for file in new_files]

        for img in images:
            file_object = io.BytesIO()
            img.save(file_object, 'PNG')
            displayed.append("data:image/png;base64," +
                             b64encode(file_object.getvalue()).decode('ascii'))

        return render_template('UploadLabel.html', filenames=file_names_uploadlabel, images_lst=displayed,
                               wrong_extension=wrong_extension, labels=labels_uploadlabel)

    return render_template('UploadLabel.html')


@app.route('/showLabels', methods=['POST'])
def show_labels():
    """Assign damage label to each uploaded image."""
    global export_done
    if request.method == 'POST':
        labeled_filename = request.form.get("filename")
        damage = request.form.get("damage")
        labels_uploadlabel[labeled_filename] = damage
        export_done = False
        return ('', 204)


@app.route('/checkComplete', methods=['GET', 'POST'])
def check_complete():
    """Check is all the if files are labeled."""
    if file_names_uploadlabel and \
            (all(filename in labels_uploadlabel for filename in file_names_uploadlabel)):
        return jsonify({"complete": "True"})
    else:
        return jsonify({"complete": "False"})


@app.route('/exportLabels', methods=['GET', 'POST'])
def export_labels():
    """Save labeled images to database and download new json file."""
    global export_done
    label_json_string = create_json_file(labels_uploadlabel)

    for filename, label in labels_uploadlabel.items():
        DB_MANAGER.insert_entry(image=images_dict_label_annotation[filename],
                                filename=filename,
                                label=label)

    if DB_MANAGER.check_if_suitable_for_training(threshold=100):
        model_path = os.path.join(main_dir, "model_best.pth")
        learner = Learner(model_path=model_path,
                          model_output_path=model_path,
                          batch_size=8,
                          annotation_path=os.path.join(main_dir, "new_dataset", "Annotations",
                                                       "updated_annotation.json"))

        learner.train(epochs=1,
                      model_name=model_path,
                      learning_rate=0.0001,
                      database_path=os.path.join(
                          webapp_dir, "data", "db_labels.db"),
                      table_name="labels",
                      annotation_path=os.path.join(main_dir, "new_dataset", "Annotations", "updated_annotation.json"))

    export_done = True
    return label_json_string


@app.route('/checkExportDone', methods=['GET', 'POST'])
def check_export_done():
    """Check if there has already been an export with the current images."""
    return jsonify({"done": str(export_done)})


@app.route('/currentImageUpload', methods=['GET', 'POST'])
def current_image_upload():
    """Get image to be displayed for the Upload and Label page."""
    return displayed[int(request.form.get("image_count"))]


@app.route('/clearDisplayed', methods=['GET', 'POST'])
def clear_displayed():
    """Clear all variables."""
    global file_names_uploadlabel
    global labels_uploadlabel
    global export_done
    global displayed
    global images_dict_label_annotation
    global file_names_predictcorrect
    global predictions_predictcorrect
    global corrections_predictcorrect
    global corrected_predictcorrect
    global displayed_upload
    global images_dict_label_annotation_prediction
    file_names_uploadlabel = []
    labels_uploadlabel = dict()
    images_dict_label_annotation = dict()
    export_done = True
    displayed = []
    file_names_predictcorrect = []
    predictions_predictcorrect = dict()
    corrections_predictcorrect = dict()
    corrected_predictcorrect = dict()
    images_dict_label_annotation_prediction = dict()
    displayed_upload = []

    return ('', 204)


file_names_predictcorrect = []
predictions_predictcorrect = dict()
corrections_predictcorrect = dict()
corrected_predictcorrect = dict()
displayed_upload = []


@app.route('/predictcorrect', methods=['GET', 'POST'])
def predictcorrect():
    """Load Predict and Correct page."""
    global file_names_predictcorrect
    global predictions_predictcorrect
    global corrections_predictcorrect
    global corrected_predictcorrect
    global displayed_upload
    global images_dict_label_annotation_prediction

    new_files_upload = []

    wrong_extension = False
    if request.method == 'POST':
        if request.form['action'] == 'uploadfiles':
            # check if the post request has the file part
            if 'files[]' not in request.files:
                flash('No file part')
                return redirect(request.url)
            files = request.files.getlist('files[]')

            # if user does not select file, browser also
            # submit an empty part without filename
            for file in files:
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if not allowed_file(file.filename):
                    wrong_extension = True
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    if filename not in file_names_predictcorrect:  # duplicate uploaded images only displayed once
                        file_names_predictcorrect.append(filename)
                        predictions_predictcorrect[filename] = {}
                        new_files_upload.append(file)

                        images_dict_label_annotation_prediction[filename] = file.read()

                        pred_label, pred_prob = model.predict(file)

                        predictions_predictcorrect[filename]['class'] = pred_label
                        corrected_predictcorrect[filename] = pred_label
                        predictions_predictcorrect[filename]['conf'] = str(round(pred_prob, 2))

        elif request.form['action'] == 'resetfiles':
            file_names_predictcorrect = []
            predictions_predictcorrect = dict()
            corrections_predictcorrect = dict()
            corrected_predictcorrect = dict()
            images_dict_label_annotation_prediction = dict()
            displayed_upload = []

        elif request.form['action'] == 'exportlabel':
            # creates the json file in the format asked from AMI team from initial prediction with corrections from user
            label_json_string = create_json_file(corrected_predictcorrect)

            return Response(label_json_string,
                            mimetype="application/json",
                            headers={'Content-Disposition': 'attachment;filename=label.json'})

    if file_names_predictcorrect or wrong_extension:

        images = [Image.open(file) for file in new_files_upload]

        for img in images:
            file_object = io.BytesIO()
            img.save(file_object, 'PNG')
            displayed_upload.append(
                "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii'))

        return render_template('PredictCorrect.html', images_upload=displayed_upload,
                               filenames=file_names_predictcorrect,
                               wrong_extension=wrong_extension,
                               predictions=predictions_predictcorrect,
                               corrections=corrections_predictcorrect,
                               corrected=corrected_predictcorrect)

    return render_template('PredictCorrect.html')


@app.route('/showCorrections', methods=['GET', 'POST'])
def show_corrections():
    """Display correction label."""
    if request.method == 'POST':
        corrected_filename = request.form.get('filename')
        damage = request.form.get("damage")
        # if the user does a correction that is the same as the original prediction --> it is not a correction
        # --> delete entry from corrections dict
        if predictions_predictcorrect[corrected_filename]['class'] == damage:
            corrections_predictcorrect.pop(corrected_filename, None)
        else:
            corrections_predictcorrect[corrected_filename] = damage
        corrected_predictcorrect[corrected_filename] = damage
        # Flag to see if there was a correction
        correction_changed = predictions_predictcorrect[corrected_filename]['class'] != damage
    return jsonify({"predicted_class": predictions_predictcorrect[corrected_filename]['class'],
                    "predicted_conf": predictions_predictcorrect[corrected_filename]['conf'],
                    "changed": str(correction_changed)})


@app.route('/currentImagePredict', methods=['GET', 'POST'])
def current_image_predict():
    """Get image to be displayed for the Predict and Correct page."""
    return displayed_upload[int(request.form.get("image_count"))]


@app.route('/correctionsDone', methods=['GET', 'POST'])
def corrections_done():
    """Save corrected labels to the Database and activate active learning."""
    if corrections_predictcorrect:
        for filename, label in corrections_predictcorrect.items():
            DB_MANAGER.insert_entry(image=images_dict_label_annotation_prediction[filename],
                                    label=label,
                                    filename=filename)

        if DB_MANAGER.check_if_suitable_for_training(threshold=100):
            model_path = os.path.join(main_dir, "model_best.pth")
            learner = Learner(model_path=model_path,
                              model_output_path=model_path,
                              batch_size=8,
                              annotation_path=os.path.join(main_dir, "new_dataset", "Annotations",
                                                           "updated_annotation.json"))

            learner.train(epochs=1,
                          model_name=model_path,
                          learning_rate=0.0001,
                          database_path=os.path.join(
                              webapp_dir, "data", "db_labels.db"),
                          table_name="labels",
                          annotation_path=os.path.join(main_dir, "new_dataset", "Annotations",
                                                       "updated_annotation.json"))
        return ('', 204)
    else:
        return ('', 403)


@app.route('/aboutus')
def aboutus():
    """Load About Us page."""
    return render_template('AboutUs.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
