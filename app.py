import streamlit as st
import cv2
import numpy as np
from PIL import Image

def custom_dodge(image, mask):
    # The "Dodge" blend mode divides the image by the inverted mask
    # This creates the "glowing edges" look that simulates shading
    return cv2.divide(image, 255 - mask, scale=256)

def adjust_gamma(image, gamma=1.0):
    # Gamma correction controls the darkness/contrast of the strokes
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- UI Setup ---
st.set_page_config(page_title="Graphite Sketcher", layout="wide")
st.title("✏️ Realistic Graphite Sketcher")
st.markdown("Create sketches that look like they were drawn with a **dark 4B pencil**.")

# --- Sidebar Controls ---
st.sidebar.header("Pencil Settings")

# Blur: Controls smoothness. Odd numbers only.
blur_amount = st.sidebar.slider("Smoothening (Blur)", 3, 99, 21, step=2)

# Gamma: Controls the darkness of the pencil strokes
stroke_darkness = st.sidebar.slider("Pencil Pressure (Darkness)", 0.1, 5.0, 1.5)

# Sharpening: Adds grit/texture
sharpen_amount = st.sidebar.slider("Paper Texture (Sharpen)", 0, 10, 5)

# File Uploader
uploaded_file = st.file_uploader("Upload a Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Invert the grayscale image
    img_invert = cv2.bitwise_not(gray)

    # 4. Apply Blur (User controlled)
    img_smoothing = cv2.GaussianBlur(img_invert, (blur_amount, blur_amount), 0)

    # 5. Create the Sketch using Dodge Blend
    final_sketch = custom_dodge(gray, img_smoothing)

    # 6. Apply Gamma Correction (To darken faint lines)
    # This is the secret to realism - making the lines truly black
    final_sketch = adjust_gamma(final_sketch, gamma=stroke_darkness)

    # 7. Apply Sharpening (Optional, for texture)
    if sharpen_amount > 0:
        kernel = np.array([[-1,-1,-1], [-1, 9+sharpen_amount, -1], [-1,-1,-1]])
        # Normalize the kernel to avoid over-saturation
        kernel = kernel / np.sum(kernel) if np.sum(kernel) != 0 else kernel
        final_sketch = cv2.filter2D(final_sketch, -1, kernel)

    # --- Display Results ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    with col2:
        st.subheader("Graphite Sketch")
        st.image(final_sketch, use_column_width=True)

    # Download
    result_img = Image.fromarray(final_sketch)
    import io
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Sketch",
        data=byte_im,
        file_name="graphite_sketch.jpg",
        mime="image/jpeg"
    )
