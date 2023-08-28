from roboflow import Roboflow
import streamlit as st
from PIL import Image
import os


def main():
    st.write('#### Select an image to upload.')
    uploaded_file = st.file_uploader('', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
    
    if uploaded_file is not None:
        # Save the uploaded file
        with open(os.path.join("default", uploaded_file.name), "wb") as f: 
            f.write(uploaded_file.getbuffer())
        
        # Rest of your code
        rf = Roboflow(api_key="L1fLCebyFsX8pYKg7N0t")
        project = rf.workspace().project("kitesboundingbox")
        model = project.version(1).model
        generate_button = st.button('Generate prediction')
        
        if generate_button:
            # Use the saved image for prediction
            prediction_path = os.path.join("default", "uploaded_image.jpg")
            model.predict(prediction_path, confidence=10, overlap=30).save("prediction.jpg")
            prediction = Image.open("prediction.jpg")
            st.image(prediction)

if __name__ == '__main__':
    main()
