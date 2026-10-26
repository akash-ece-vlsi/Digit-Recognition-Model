
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("digit_model.keras")

# Drawing canvas settings
CANVAS_SIZE = 280
BRUSH_SIZE = 18

root = tk.Tk()
root.title("AI Handwritten Digit Recognizer")
root.resizable(False, False)

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black",
    cursor="cross"
)
canvas.pack(padx=10, pady=10)

# Keep a matching image for model input
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
draw = ImageDraw.Draw(image)


def paint(event):
    x, y = event.x, event.y
    r = BRUSH_SIZE // 2

    canvas.create_oval(
        x-r, y-r, x+r, y+r,
        fill="white", outline="white"
    )
    draw.ellipse(
        (x-r, y-r, x+r, y+r),
        fill=255
    )



def predict_digit():
    bbox = image.getbbox()

    if bbox is None:
        result.config(text="Draw a digit first!")
        return

    # Crop the digit and preserve its proportions
    digit = image.crop(bbox)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Place the digit on a 28x28 image
    processed = Image.new("L", (28, 28), 0)
    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2
    processed.paste(digit, (x, y))

    # Centre the digit using its intensity-weighted centre of mass
    pixels = np.array(processed, dtype=np.float32)
    total = pixels.sum()

    if total == 0:
        result.config(text="Draw a digit first!")
        return

    rows, cols = np.indices(pixels.shape)
    cy = (rows * pixels).sum() / total
    cx = (cols * pixels).sum() / total

    # Shift the digit's centre of mass to the image centre
    processed = processed.transform(
        (28, 28),
        Image.Transform.AFFINE,
        (1, 0, cx - 13.5, 0, 1, cy - 13.5),
        resample=Image.Resampling.BILINEAR
    )

    pixels = np.array(processed, dtype=np.float32) / 255.0
    predictions = model.predict(pixels.reshape(1, 28, 28), verbose=0)[0]

    number = int(np.argmax(predictions))
    confidence = float(predictions[number]) * 100

    result.config(
        text=f"Prediction: {number} ({confidence:.1f}% confidence)"
    )



def clear_canvas():
    global image, draw

    canvas.delete("all")
    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(image)
    result.config(text="Draw a digit from 0 to 9")


canvas.bind("<B1-Motion>", paint)
canvas.bind("<Button-1>", paint)

tk.Button(
    root, text="Recognize Digit",
    command=predict_digit
).pack(pady=5)

tk.Button(
    root, text="Clear",
    command=clear_canvas
).pack(pady=5)

result = tk.Label(
    root,
    text="Draw a digit from 0 to 9",
    font=("Arial", 13)
)
result.pack(pady=10)

root.mainloop()
