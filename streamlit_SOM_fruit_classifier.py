import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Import PyTorch for Feature Extraction
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="SOM Fruit Classifier", page_icon="🍎", layout="centered")

# ==========================================
# CONSTANTS
# ==========================================
MODEL_FILE = "som_model.pkl"
CSV_FILE = "fruit_nutrients.csv"

# ==========================================
# CACHED LOADERS (Runs only once)
# ==========================================
@st.cache_resource
def load_feature_extractor():
    """Loads the exact same MobileNetV2 model used during training."""
    model = models.mobilenet_v2(weights=models.MobileNetV2_Weights.DEFAULT)
    model.classifier = torch.nn.Identity() # Remove classification head
    model.eval() # Set to evaluation mode
    return model

@st.cache_resource
def load_som_model():
    """Loads the pre-trained SOM model, label map, and threshold from the .pkl file."""
    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ Could not find '{MODEL_FILE}'. Please ensure it is in the same folder as this script.")
        return None, None, None
        
    with open(MODEL_FILE, 'rb') as f:
        data = pickle.load(f)
    return data["model"], data["map"], data["threshold"]

@st.cache_data
def load_nutrients():
    """Loads fruit nutritional data from the CSV file."""
    if not os.path.exists(CSV_FILE):
        return {}
    
    df = pd.read_csv(CSV_FILE)
    df['Fruit'] = df['Fruit'].str.lower().str.strip()
    df.set_index('Fruit', inplace=True)
    return df.to_dict('index')

def extract_features(img, extractor):
    """Preprocesses the image and extracts the 1280-dimensional feature vector."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        features = extractor(tensor)
    return features.squeeze().numpy()

# ==========================================
# INITIALIZATION
# ==========================================
# Load all required components into memory
feature_extractor = load_feature_extractor()
som_model, neuron_label_map, outlier_threshold = load_som_model()
fruit_nutrients = load_nutrients()

if som_model is None:
    st.stop() # Halt execution if the model file is missing

# ==========================================
# MAIN USER INTERFACE
# ==========================================
st.title("🍎 SOM Fruit Classifier")
st.markdown("Upload a fruit image, and our Self-Organizing Map will identify it and provide nutritional facts.")

# File Uploader
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(raw_image, caption="Uploaded Image", use_container_width=True)
        classify_btn = st.button("🔍 Classify Fruit", type="primary", use_container_width=True)

    with col2:
        if classify_btn:
            with st.spinner("Analyzing image features..."):
                
                # 1. Extract Features using PyTorch MobileNetV2
                img_vector = extract_features(raw_image, feature_extractor)

                # 2. Find the winning neuron on the SOM
                winner = som_model.winner(img_vector)
                
                # 3. Check distance against the dynamic threshold
                winning_weights = som_model.get_weights()[winner]
                quantization_error = np.linalg.norm(img_vector - winning_weights)
                
                if quantization_error > outlier_threshold:
                    st.warning(f"⚠️ **Unidentified Object**\n\nThis image does not match known fruit patterns closely enough. \n\n*(Distance: {quantization_error:.2f} | Max allowed: {outlier_threshold:.2f})*")
                else:
                    # 4. Map the winning neuron to a fruit label
                    if winner in neuron_label_map:
                        predicted_label = neuron_label_map[winner]
                    else:
                        # Fallback: Find the closest known neuron if the exact winner has no label
                        min_dist = float('inf')
                        closest_label = "Unknown"
                        for position, label in neuron_label_map.items():
                            dist = np.linalg.norm(np.array(winner) - np.array(position))
                            if dist < min_dist:
                                min_dist = dist
                                closest_label = label
                        predicted_label = closest_label

                    # 5. Display the Classification Result
                    st.success(f"### ✅ Identified: {predicted_label.upper()}")
                    
                    # 6. Fetch and Display Nutritional Data
                    fruit_name = predicted_label.lower().strip()
                    if fruit_name in fruit_nutrients:
                        st.markdown("#### Nutritional Facts")
                        data = fruit_nutrients[fruit_name]
                        
                        nut_cols = st.columns(2)
                        for i, (key, value) in enumerate(data.items()):
                            nut_cols[i % 2].metric(label=key, value=value)
                    else:
                        st.info(f"Nutritional data not found for '{predicted_label.upper()}' in the CSV.")