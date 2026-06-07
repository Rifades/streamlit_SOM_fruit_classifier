import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from minisom import MiniSom
from collections import Counter
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="SOM Fruit Classifier", page_icon="🍎", layout="centered")

# ==========================================
# CONSTANTS
# ==========================================
FRUIT_CLASSES = ['Apple', 'Banana', 'Orange', 'Dragon Fruit', 'Lichi', 'Mango', 'Pineapple', 'Watermelon', 'Guava', 'Jack fruit']
IMAGE_SIZE = (64, 64)
IMAGE_FOLDER = "./fruits"
CSV_FILE = "fruit_nutrients.csv"
MODEL_FILE = "som_model.pkl"
UNIDENTIFIED_THRESHOLD = 60 

# ==========================================
# DATA & MODEL LOADING (CACHED)
# ==========================================
@st.cache_data
def load_nutrients_from_csv(csv_path):
    """Loads fruit nutritional data from a CSV file into a dictionary."""
    try:
        df = pd.read_csv(csv_path)
        df['Fruit'] = df['Fruit'].str.lower()
        df.set_index('Fruit', inplace=True)
        return df.to_dict('index')
    except FileNotFoundError:
        st.error(f"CRITICAL ERROR: CSV file '{csv_path}' not found.")
        return {}
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return {}

def load_images_from_folder(folder, size=IMAGE_SIZE):
    """Loads and processes training images."""
    images = []
    labels = []
    
    if not os.path.exists(folder):
        st.error(f"Error: The folder '{folder}' does not exist.")
        return np.array([]), np.array([])

    for label in FRUIT_CLASSES:
        class_folder = os.path.join(folder, label)
        if not os.path.exists(class_folder):
            class_folder = os.path.join(folder, label.lower())
        
        if not os.path.exists(class_folder):
            st.warning(f"Class folder for '{label}' not found. Skipping.")
            continue
            
        for filename in os.listdir(class_folder):
            path = os.path.join(class_folder, filename)
            try:
                img = Image.open(path).convert('RGB').resize(size)
                images.append(np.array(img).flatten() / 255.0)
                labels.append(label)
            except Exception:
                pass
                
    return np.array(images), np.array(labels)

def train_som_model(images, labels):
    """Trains the SOM and generates the Label Map."""
    som_model = MiniSom(10, 10, images.shape[1], sigma=3.0, learning_rate=0.5)
    som_model.random_weights_init(images)
    som_model.train_random(images, 5000)
    
    raw_map = som_model.labels_map(images, labels)
    neuron_label_map = {}
    
    for position, label_list in raw_map.items():
        most_common_label = Counter(label_list).most_common(1)[0][0]
        neuron_label_map[position] = most_common_label
        
    return som_model, neuron_label_map

@st.cache_resource
def initialize_model():
    """Loads the model from disk, or trains it if it doesn't exist."""
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
            return data["model"], data["map"]
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    else:
        with st.spinner("No saved model found. Training SOM model... please wait."):
            train_images, train_labels = load_images_from_folder(IMAGE_FOLDER)
            if len(train_images) == 0:
                st.error("No training images found. Please ensure the 'fruits' folder exists and has images.")
                return None, None
            
            som_model, label_map = train_som_model(train_images, train_labels)
            
            # Save the trained model
            data = {"model": som_model, "map": label_map}
            try:
                with open(MODEL_FILE, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                st.error(f"Error saving model: {e}")
                
            return som_model, label_map

# ==========================================
# STREAMLIT UI & LOGIC
# ==========================================
st.title("🍎 SOM Fruit Classifier")
st.markdown("Upload a fruit image, and our Self-Organizing Map will identify it and provide nutritional facts.")

# Initialize Data & Model
fruit_nutrients = load_nutrients_from_csv(CSV_FILE)
som, neuron_label_map = initialize_model()

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
            with st.spinner("Analyzing image..."):
                # Prepare image vector
                processed_img = raw_image.resize(IMAGE_SIZE)
                img_vector = np.array(processed_img).flatten() / 255.0

                # 1. Find the winning neuron
                winner = som.winner(img_vector)
                
                # 2. Check Distance (Outlier Detection)
                winning_weights = som.get_weights()[winner]
                quantization_error = np.linalg.norm(img_vector - winning_weights)
                
                if quantization_error > UNIDENTIFIED_THRESHOLD:
                    st.warning(f"⚠️ **Unidentified Object**\n\nThe distance ({quantization_error:.2f}) was too far from known categories.")
                else:
                    # 3. Fast Classification using Map
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

                    # 4. Display Results
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