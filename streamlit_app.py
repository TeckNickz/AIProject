from roboflow import Roboflow
rf = Roboflow(api_key="L1fLCebyFsX8pYKg7N0t")
project = rf.workspace().project("kitesboundingbox")
model = project.version(1).model


# Add in location to select image.

st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)


# infer on a local image
print(model.predict(uploaded_file, confidence=40, overlap=30).json())

# visualize your prediction
model.predict(uploaded_file, confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
