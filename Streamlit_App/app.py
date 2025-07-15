import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np

# CONFIGURATION 
# IMPORTANT: This must be the path to your saved Keras model.
MODEL_PATH = 'teeth_classification_model_with_FineTune.keras'

# You must list your class names in the EXACT order the model was trained on.
CLASS_NAMES = [
    'CaS', 
    'CoS', 
    'Gum', 
    'MC', 
    'OC', 
    'OLP', 
    'OT'
] 

# MODEL LOADING 

# Use st.cache_resource to load the model only once, which is much faster.
@st.cache_resource
def load_keras_model():
    """
    Loads the saved Keras model from the specified path.
    """
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_keras_model()

# IMAGE PREPROCESSING & PREDICTION 
def preprocess_image_and_predict(image_to_process, target_size=(256, 256)):
    """
    Preprocesses the uploaded image and makes a prediction using the loaded model.
    1. Opens and converts image to RGB.
    2. Resizes image to the target size the model expects.
    3. Converts image to a NumPy array.
    4. Expands dimensions to create a "batch" of 1 image.
    5. Uses ResNet50's specific `preprocess_input` function.
    6. Makes a prediction.
    """
    # Ensure image is in RGB format
    img = image_to_process.convert('RGB')
    
    # Resize the image to match the model's expected input shape
    img = img.resize(target_size)
    
    # Convert the PIL image to a NumPy array
    img_array = np.array(img)
    
    # Expand the dimensions to create a batch of 1
    # Shape becomes (1, 256, 256, 3)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image using the same function as during training
    processed_img = preprocess_input(img_array_expanded)
    
    # Make a prediction
    prediction = model.predict(processed_img)
    
    return prediction

# STREAMLIT APP LAYOUT 

st.title("🦷 Teeth Disease Classification 🦷")
st.write("Upload a dental image and the AI will predict the condition.")
st.write("This app uses a fine-tuned ResNet50 model.")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose a tooth image...", type=["jpg", "jpeg", "png"])

if model is None:
    st.stop() # Don't run the rest of the app if the model failed to load

if uploaded_file is not None:
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Add a "Classify" button
    if st.button('Classify Image'):
        with st.spinner('Analyzing the image...'):
            # Preprocess the image and get the prediction
            prediction = preprocess_image_and_predict(image)
            
            # Get the index of the highest probability
            predicted_class_index = np.argmax(prediction)
            
            # Get the corresponding class name
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            
            # Get the confidence score (probability)
            confidence_score = np.max(prediction)
            
            # Display the result
            st.success(f"**Prediction:** {predicted_class_name}")
            st.info(f"**Confidence:** {confidence_score:.2%}")

            # Display prediction probabilities for all classes
            st.write("Prediction Probabilities:")
            prob_df = {CLASS_NAMES[i]: prediction[0][i] for i in range(len(CLASS_NAMES))}
            st.bar_chart(prob_df)