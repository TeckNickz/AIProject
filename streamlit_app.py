from roboflow import Roboflow
import streamlit as st
from PIL import Image
import os

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    img.tobytes()
    return img
  
  
def main():
    st.write('#### Select an image to upload.')
    uploaded_file = st.file_uploader('',type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        st.write(file_details)
        with open(os.path.join("default",uploaded_file.name),"wb") as f: 
            f.write(uploaded_file.getbuffer())
    rf = Roboflow(api_key="L1fLCebyFsX8pYKg7N0t")
    project = rf.workspace().project("kitesboundingbox")
    model = project.version(1).model
    generatebutton = st.button('Generatie prediction')
    image = load_image(uploaded_file)
    if generatebutton:
        # visualize your prediction
        model.predict(image, confidence=10, overlap=30).save("prediction.jpg")
        prediction = Image.open("prediction.jpg")
        st.image(prediction)
if __name__ == '__main__':
    main()
