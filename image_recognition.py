import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB3 # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions # type: ignore

# Load pre-trained EfficientNet-B3 model
model = EfficientNetB3(weights="imagenet")

def recognize_image(img_path):
    """Recognize landmarks using EfficientNet-B3"""
    img = image.load_img(img_path, target_size=(300, 300))  # Adjusted for EfficientNet
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]  # Returns top 3 predictions
