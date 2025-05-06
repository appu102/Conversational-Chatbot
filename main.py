import streamlit as st
from image_recognition import recognize_image
from nlp_model import generate_response
from PIL import Image

st.title("🗺 Conversational Landmark Recognition Chatbot 🤖")

# Upload Image
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)
    st.write("🔍 Recognizing landmarks...")

    # Recognize the Image
    predictions = recognize_image(uploaded_file)

    if predictions:
        image_caption = ', '.join([f"{pred[1]} ({pred[2]*100:.2f}%)" for pred in predictions])
        st.write(f"📌 **Predictions:** {image_caption}")

        # User Query
        user_query = st.text_input("💬 Ask a question about the image:")
        if user_query:
            response = generate_response(image_caption, user_query)
            st.write(f"🤖 **Chatbot:** {response}")
    else:
        st.write("⚠️ No landmark detected, please try another image.")
