from flask import Flask, render_template, request
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)

model = YOLO("yolov8n.pt")  # Make sure the .pt file is in the same folder

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        image = request.files["image"]

        upload_folder = "static"
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, image.filename)
        image.save(image_path)

        img = cv2.imread(image_path)
        results = model(img)

        result_path = os.path.join(upload_folder, f"result_{image.filename}")
        annotated_img = results[0].plot()
        cv2.imwrite(result_path, annotated_img)

        return render_template("result.html", result_image=f"result_{image.filename}")

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
