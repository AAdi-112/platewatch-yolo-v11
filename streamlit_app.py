import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")                  

model = load_model()

# Streamlit UI
st.title("Platewatch (License Plate Detection) with YOLO-11")
st.write("Upload an image to detect license plates.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO prediction
    results = model.predict(image)

    # Show results
    for r in results:
        boxes = r.boxes.xyxy.tolist()
        classes = r.boxes.cls.tolist()
        confs = r.boxes.conf.tolist()
        st.write("Detections:")
        for i, box in enumerate(boxes):
            st.write(f"Class: {model.names[int(classes[i])]}, Conf: {confs[i]:.2f}, Box: {box}")

    # Render and display annotated image
    results[0].save(filename="output.jpg")
    st.image("output.jpg", caption="Detection Result", use_column_width=True)
