import streamlit as st
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import pandas as pd
import time

# Set up the Streamlit app title and description
st.title("PALDRON: Bacterial Colony Detection Web App")
st.write("Upload images of petri dishes and detect bacterial colonies.")

# Load the YOLO model
model = YOLO('best.pt')

# List of classes
classes = ['artifact', 'bubble', 'colony', 'gate', 'lock', 'sharpie', 'star', 'tape', 'unlock']

# Initialize an empty dictionary to store the results
detection_data = {
    "File Name": [],  # Image file name
    "Timestamp": [],  # Timestamp
}

# Initialize columns for each class with empty lists
for cls in classes:
    detection_data[cls] = []

# Allow users to upload multiple images
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:  # Check if any files have been uploaded
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name

        # Save file temporarily to the current directory
        with open(file_name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Perform inference using the YOLO model
        results = model(file_name)

        # Add filename and timestamp for this image
        detection_data["File Name"].append(file_name)
        detection_data["Timestamp"].append(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        # Initialize a dictionary to store detections for this image
        detected_in_image = {cls: 0 for cls in classes}

        # Get the detected results for the current image
        for result in results:
            boxes = result.boxes  # Bounding boxes for detected objects
            for box in boxes:
                cls_id = int(box.cls[0])  # Detected class (integer)
                cls_name = model.names[cls_id]  # Class name from YOLO model

                # If the detected class is one of the predefined classes, increment the count
                if cls_name in detected_in_image:
                    detected_in_image[cls_name] += 1

        # Append the detection results for this image to the detection_data dictionary
        for cls in classes:
            detection_data[cls].append(detected_in_image[cls])

        # Get the first result and render the image with bounding boxes
        result_image = results[0].plot()

        # Convert the result_image (NumPy array) to RGB for displaying in matplotlib
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # Display the image with bounding boxes using Streamlit
        st.image(result_image_rgb, caption=f"Processed Image: {file_name}", use_column_width=True)

    # Convert detection data to a pandas DataFrame
    df = pd.DataFrame(detection_data)

    # Display the DataFrame in the Streamlit app
    st.write("Detection Results:")
    st.dataframe(df)

    # Provide a button to download the CSV file
    csv_filename = 'detection_results.csv'
    df.to_csv(csv_filename, index=False)

    # Download button for CSV file
    st.download_button(
        label="Download Detection Results as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=csv_filename,
        mime='text/csv'
    )
