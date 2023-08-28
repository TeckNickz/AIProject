from roboflow import Roboflow
import streamlit as st
from PIL import Image
import os


rf = Roboflow(api_key="L1fLCebyFsX8pYKg7N0t")
project = rf.workspace().project("kitesboundingbox")
model = project.version(1).model

def load_image(uploaded_file):
    img = Image.open(uploaded_file)
    return img
  
def main():
  st.sidebar.write('#### Select an image to upload.')
  uploaded_file = st.sidebar.file_uploader('',type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)
  if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
    img = load_image(uploaded_file)
    with open(os.path.join("default",uploaded_file.name),"wb") as f: 
        f.write(uploaded_file.getbuffer())  
  # visualize your prediction
  model.predict(uploaded_file, confidence=40, overlap=30).save("./prediction.jpg")
  prediction = Image.open("./prediction.jpg")
        #st.image(prediction)
if __name__ == '__main__':
    main()
