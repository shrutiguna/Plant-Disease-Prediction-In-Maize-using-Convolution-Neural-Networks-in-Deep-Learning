
# **Plant Disease Prediction In Maize using Convolution Neural Networks in Deep Learning**

## **Overview**
This project focuses on detecting plant diseases in maize crops using a convolutional neural network (CNN) model. The system provides an easy-to-use web interface built with Streamlit, allowing users to upload images of maize leaves and receive predictions about potential diseases. The application also offers tailored suggestions for managing specific diseases, enhancing usability for farmers and agricultural researchers.

---

## **Key Features**
- **Disease Detection**: Uses a TensorFlow CNN model to classify plant diseases.
- **Real-Time Prediction**: Accepts image uploads and provides instant disease diagnosis.
- **Disease Management Suggestions**: Offers disease-specific recommendations for treatment and care.
- **User-Friendly Interface**: Built with Streamlit, making it accessible and interactive.
- **Scalable**: Supports adding new diseases and suggestions by updating the `sugg.json` file.

---

## **Technologies Used**
- **TensorFlow**: For the CNN model used in disease prediction.
- **OpenCV**: For image processing and resizing.
- **Streamlit**: For creating an interactive web application.
- **Pillow**: For handling image uploads.
- **JSON**: For storing disease-specific suggestions.

---

## **File Structure**
- **`plant_disease_3.h5`**: Pre-trained CNN model for plant disease detection.
- **`sugg.json`**: JSON file containing disease-specific suggestions.
- **Streamlit Script**: Implements the user interface and handles predictions.
- **Test Image**: Uploaded by the user and processed for prediction.

---

## **How to Run the Project**
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7+ installed. Then, run:
   ```bash
   pip install streamlit tensorflow opencv-python pillow numpy
   ```

3. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

4. **Upload an Image**:
   - Use the web interface to upload an image of a maize leaf.
   - Click on the "Submit" button to view the prediction and suggestions.

---

## **Model Details**
- **Input Size**: The CNN model accepts images resized to 224x224 pixels.
- **Normalization**: Images are normalized to have pixel values between 0 and 1.
- **Output**: The model predicts the class of the disease, which is then mapped to a suggestion in the `sugg.json` file.

---

## **Usage**
1. Upload an image of a maize leaf affected by disease.
2. View the predicted disease class on the interface.
3. Receive actionable suggestions for managing the identified disease.

**Example Output:**
- **Prediction**: "Rust disease"
- **Suggestion**: "Apply fungicides early to manage rust effectively."

---

## **Customization**
- **Adding Diseases**: Update the `sugg.json` file with new disease names and suggestions.
- **Model Updates**: Replace the `plant_disease_3.h5` file with a newer or retrained model to improve prediction accuracy.

---

## **Acknowledgments**
This project is based on deep learning techniques for plant disease detection, aiming to assist farmers in managing maize crop diseases efficiently. It uses a pre-trained CNN model trained on a dataset of maize leaf images.

---

## **Future Enhancements**
- Incorporating real-time camera integration for live disease detection.
- Expanding the model to support multiple crop types and diseases.
- Improving accuracy with larger, more diverse datasets.

---
