from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as T
from torchvision import models
import numpy as np
import cv2
from PIL import Image
import base64
import io

app = Flask(__name__)

# ---------------------------
# LOAD AI MODEL (DeepLabV3)
# ---------------------------

model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

transform = T.Compose([
    T.Resize(520),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Approximate class for wall/building
WALL_CLASS = 12

# ---------------------------
# SEGMENT WALL FUNCTION
# ---------------------------

def segment_wall(image):
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    prediction = output.argmax(0).byte().cpu().numpy()

    mask = (prediction == WALL_CLASS).astype(np.uint8) * 255

    return mask


# ---------------------------
# APPLY COLOR BLEND
# ---------------------------

def apply_color(original, mask, hex_color):
    # Convert hex to RGB
    color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

    color_layer = np.zeros_like(original)
    color_layer[:] = color

    mask_3 = cv2.merge([mask, mask, mask])

    blended = np.where(
        mask_3 == 255,
        cv2.addWeighted(original, 0.3, color_layer, 0.7, 0),
        original
    )

    return blended


# ---------------------------
# ROUTES
# ---------------------------

@app.route("/")
def home():
    return render_template("Color_Cards_Page_AI.html")


@app.route("/paint", methods=["POST"])
def paint():
    data = request.json

    image_data = base64.b64decode(data["image"])
    hex_color = data["color"]

    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    original = np.array(image)

    mask = segment_wall(image)
    result = apply_color(original, mask, hex_color)

    _, buffer = cv2.imencode(".jpg", result)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": encoded_image})


# ---------------------------
# RUN SERVER
# ---------------------------

if __name__ == "__main__":
    app.run(debug=True)