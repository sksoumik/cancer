# way to upload an image: endpoint
from flask import Flask
from flask import render_template
from flask import request
import os

app = Flask(__name__)
image_upload_folder = "uploads/"


@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                image_upload_folder,
                image_file.filename
                )
            image_file.save(image_location)
            return render_template("index.html", prediction=1)
    return render_template("index.html", prediction=0)


if __name__ == '__main__':
    app.run(debug=True)

# way to save the image
# function to make a prediction
# show the results
