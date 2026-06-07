import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from minisom import MiniSom
from collections import Counter
import streamlit as st

# Import TensorFlow for Feature Extraction
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="SOM Fruit Classifier", page_icon="🍎", layout="centered")

# ==========================================
# CONSTANTS
# ==========================================
FRUIT_CLASSES = ['Apple', 'Banana', 'Orange', 'Dragon Fruit', 'Lichi', 'Mango', 'Pineapple', 'Watermelon', 'Guava', 'Jack fruit']
IMAGE_SIZE = (224, 224) # MobileNetV2 optimal size
IMAGE_FOLDER = "./fruits"
CSV_FILE = "fruit_nutrients.csv"
MODEL_FILE = "som_model.pkl"

# ==========================================
# FEATURE EXTRACTOR (CNN)
# ==========================================
@st.cache_resource
def get_feature_extractor():
    """Loads a pre-trained MobileNetV2 model to extract 1280-dimensional features instead of raw pixels."""
    return MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img, extractor):
    """Converts an image into a meaningful feature vector."""
    img = img.resize(IMAGE_SIZE)
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = extractor.predict(x, verbose=0)
    return features[0]

# ==========================================
# DATA & MODEL LOADING (CACHED)
# ==========================================
@st.cache_data
def load_nutrients_from_csv(csv_path):
    """Loads fruit nutritional data from a CSV file into a dictionary."""
    try:
        df = pd.read_csv(csv_path)
        df['Fruit'] = df['Fruit'].str.lower().str.strip()
        df.set_index('Fruit', inplace=True)
        return df.to_dict('index')
    except FileNotFoundError:
        st.error(f"CRITICAL ERROR: CSV file '{csv_path}' not found.")
        return {}
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return {}

def load_images_from_folder(folder, extractor):
    """Loads images and extracts their features."""
    features_list = []
    labels = []
    
    if not os.path.exists(folder):
        st.error(f"Error: The folder '{folder}' does not exist.")
        return np.array([]), np.array([])

    # Get total files for progress bar
    total_files = sum([len(files) for r, d, files in os.walk(folder)])
    if total_files == 0:
        return np.array([]), np.array([])

    progress_bar = st.progress(0, text="Extracting image features...")
    processed = 0

    for label in FRUIT_CLASSES:
        class_folder = os.path.join(folder, label)
        if not os.path.exists(class_folder):
            class_folder = os.path.join(folder, label.lower())
        
        if not os.path.exists(class_folder):
            continue
            
        for filename in os.listdir(class_folder):
            path = os.path.join(class_folder, filename)
            try:
                img = Image.open(path).convert('RGB')
                features = extract_features(img, extractor)
                features_list.append(features)
                labels.append(label)
            except Exception:
                pass
            finally:
                processed += 1
                progress_bar.progress(min(processed / total_files, 1.0), text=f"Processing {label}...")
                
    progress_bar.empty()
    return np.array(features_list), np.array(labels)

def train_som_model(features_array, labels):
    """Trains the SOM, generates the Label Map, and calculates a dynamic threshold."""
    # Initialize SOM
    som_model = MiniSom(10, 10, features_array.shape[1], sigma=3.0, learning_rate=0.5)
    som_model.random_weights_init(features_array)
    som_model.train_random(features_array, 5000)
    
    # Map neurons to labels
    raw_map = som_model.labels_map(features_array, labels)
    neuron_label_map = {}
    for position, label_list in raw_map.items():
        most_common_label = Counter(label_list).most_common(1)[0][0]
        neuron_label_map[position] = most_common_label
        
    # Calculate Dynamic Threshold (99th percentile of training distances)
    errors = []
    for vec in features_array:
        winner = som_model.winner(vec)
        weights = som_model.get_weights()[winner]
        errors.append(np.linalg.norm(vec - weights))
    
    dynamic_threshold = np.percentile(errors, 99) # 99% of valid fruits will pass this distance
    
    return som_model, neuron_label_map, dynamic_threshold

@st.cache_resource
def initialize_model():
    """Loads the model from disk, or trains it if it doesn't exist."""
    extractor = get_feature_extractor()

    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
            return data["model"], data["map"], data["threshold"], extractor
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None, None, None
    else:
        with st.spinner("No saved model found. Training SOM model... please wait."):
            train_features, train_labels = load_images_from_folder(IMAGE_FOLDER, extractor)
            if len(train_features) == 0:
                st.error("No training images found. Please ensure the 'fruits' folder exists and has images.")
                return None, None, None, None
            
            som_model, label_map, threshold = train_som_model(train_features, train_labels)
            
            # Save the trained model
            data = {"model": som_model, "map": label_map, "threshold": threshold}
            try:
                with open(MODEL_FILE, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                st.error(f"Error saving model: {e}")
                
            return som_model, label_map, threshold, extractor

# ==========================================
# STREAMLIT UI & LOGIC
# ==========================================
# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Force Retrain Model", type="primary", use_container_width=True):
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        st.cache_resource.clear()
        st.rerun()
    st.caption("Click this if you added new images to the 'fruits' folder.")

st.title("🍎 SOM Fruit Classifier")
st.markdown("Upload a fruit image, and our Self-Organizing Map will identify it and provide nutritional facts.")

# Initialize Data & Model
fruit_nutrients = load_nutrients_from_csv(CSV_FILE)
som, neuron_label_map, outlier_threshold, feature_extractor = initialize_model()

if som is None or not neuron_label_map:
    st.stop() # Halt execution if the model failed to load/train

# File Uploader
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(raw_image, caption="Uploaded Image", use_container_width=True)
        classify_btn = st.button("🔍 Classify Fruit", type="primary", use_container_width=True)

    with col2:
        if classify_btn:
            with st.spinner("Analyzing image features..."):
                # 1. Extract Features using MobileNetV2
                img_vector = extract_features(raw_image, feature_extractor)

                # 2. Find the winning neuron
                winner = som.winner(img_vector)
                
                # 3. Check Distance against Dynamic Threshold
                winning_weights = som.get_weights()[winner]
                quantization_error = np.linalg.norm(img_vector - winning_weights)
                
                if quantization_error > outlier_threshold:
                    st.warning(f"⚠️ **Unidentified Object**\n\nThe image features did not match known fruit patterns closely enough. (Distance: {quantization_error:.2f} | Max allowed: {outlier_threshold:.2f})")
                else:
                    # 4. Fast Classification using Map
                    if winner in neuron_label_map:
                        predicted_label = neuron_label_map[winner]
                    else:
                        min_dist = float('inf')
                        closest_label = "Unknown"
                        for position, label in neuron_label_map.items():
                            dist = np.linalg.norm(np.array(winner) - np.array(position))
                            if dist < min_dist:
                                min_dist = dist
                                closest_label = label
                        predicted_label = closest_label

                    # 5. Display Results
                    st.success(f"### ✅ Identified: {predicted_label.upper()}")
                    
                    fruit_name = predicted_label.lower().strip()
                    if fruit_name in fruit_nutrients:
                        st.markdown("#### Nutritional Facts")
                        data = fruit_nutrients[fruit_name]
                        
                        # Display nutrients in a nice grid format
                        nut_cols = st.columns(2)
                        for i, (key, value) in enumerate(data.items()):
                            nut_cols[i % 2].metric(label=key, value=value)
                    else:
                        st.info(f"Nutritional data not found for '{predicted_label.upper()}' in the CSV.")