import os
import gradio as gr
import numpy as np
import tensorflow as tf

IMG_SIZE = 160
MODEL_PATH = os.path.join("models", "age_regressor.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. "
        "Make sure models/age_regressor.h5 exists in your repo."
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(pil_img):
    img = tf.keras.preprocessing.image.img_to_array(pil_img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), antialias=True)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return tf.expand_dims(img, 0)

def predict(img):
    arr = preprocess_image(img)
    pred = model.predict(arr, verbose=0)[0][0]
    age = float(np.clip(pred, 0, 100))
    return f"Estimated age: {age:.1f} years"

with gr.Blocks() as demo:
    gr.Markdown("# Age Regression (UTKFace) — Gradio Demo")
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload a cropped face image")
    out = gr.Textbox(label="Prediction")
    gr.Button("Predict").click(fn=predict, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch()
