from flask import Flask, request, Response
import io
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8n.pt")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img_bytes = file.read()

    img = Image.open(io.BytesIO(img_bytes))

    results = model(img, stream=True)

    for result in results:
        annotated_img = result.plot()

        annotated_img = Image.fromarray(annotated_img)

        img_byte_arr = io.BytesIO()
        annotated_img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        return Response(img_byte_arr, mimetype="image/png")

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    return Response(img_byte_arr, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)