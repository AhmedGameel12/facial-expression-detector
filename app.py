import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model_fer2013_optimized.h5")

# Define emotion labels
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Streamlit UI
st.title("Facial Expression Recognition")
st.subheader("Upload an image to predict the emotion")

# Upload image
uploaded_file = st.file_uploader("Upload a .jpg or .png file", type=["jpg", "png", "jpeg"])

# Function to predict emotion
def predict_emotion(img):
    img = img.resize((48, 48)).convert("L")  # Resize to 48x48 and convert to grayscale
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0).reshape(1, 48, 48, 1)

    prediction = model.predict(img)[0]
    class_index = np.argmax(prediction)
    
    return label_dict[class_index], prediction

# Display and process uploaded image
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image_pil = Image.open(uploaded_file)

    # Predict emotion
    emotion, confidence = predict_emotion(image_pil)

    # Display results
    st.markdown(f"### Predicted Emotion: **{emotion}** 😃")
    st.bar_chart(confidence)
    st.markdown("### Confidence Scores")
    for i, label in label_dict.items():
        st.write(f"**{label}:** {confidence[i]*100:.2f}%")