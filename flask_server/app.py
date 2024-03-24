from flask import Flask, request, jsonify
import io
from PIL import Image
import numpy as np
from collections import Counter
from ultralytics import YOLO
import base64
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

    response_data = {
        "image": None,
        "objects": []
    }

    for result in results:
        annotated_img = result.plot()
        annotated_img = Image.fromarray(annotated_img)

        # Get the count of objects and their names
        tensor_values = result.boxes.cls
        arr = np.array(tensor_values)
        counts = Counter(arr)
        classes = result.names
        objects = []
        for i in counts:
            objects.append({"name": classes[i], "count": counts[i]})

        # Convert the annotated image to base64
        img_byte_arr = io.BytesIO()
        annotated_img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        base64_img = base64.b64encode(img_byte_arr).decode('utf-8')

        response_data["image"] = base64_img
        response_data["objects"] = objects

    if response_data["image"] is None:
        # If no objects were detected, return the original image
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        base64_img = base64.b64encode(img_byte_arr).decode('utf-8')
        response_data["image"] = base64_img

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True)