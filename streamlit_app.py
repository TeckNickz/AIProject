from roboflow import Roboflow
import streamlit as st

rf = Roboflow(api_key="L1fLCebyFsX8pYKg7N0t")
project = rf.workspace().project("kitesboundingbox")
model = project.version(1).model

def main():
  # Add in location to select image.
  st.sidebar.write('#### Select an image to upload.')
  uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)
  # visualize your prediction
  model.predict(uploaded_file, confidence=40, overlap=30).save("prediction.jpg")
  prediction = Image.open("prediction.jpg")
        #st.image(prediction)
if __name__ == '__main__':
    main()
