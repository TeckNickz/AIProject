from roboflow import Roboflow
import streamlit as st
from PIL import Image
import os

  
def main():
    st.write('#### Select an image to upload.')
    uploaded_file = st.file_uploader('',type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        st.write(file_details)
        with open(os.path.join("default",uploaded_file.name),"wb") as f: 
            f.write(uploaded_file.getbuffer())
        oldext = os.path.splitext("default/"+uploaded_file.name)[1]
        os.rename("default/"+uploaded_file.name, "default" + oldext)
    rf = Roboflow(api_key="L1fLCebyFsX8pYKg7N0t")
    project = rf.workspace().project("kitesboundingbox")
    model = project.version(1).model
    generatebutton = st.button('Generatie prediction')
    if generatebutton:
        # visualize your prediction
        model.predict("./default/"+ uploaded_file.name, confidence=10, overlap=30).save("prediction.jpg")
        prediction = Image.open("prediction.jpg")
        st.image(prediction)
if __name__ == '__main__':
    main()
