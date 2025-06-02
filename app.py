import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Import custom modules
from src.preprocessing.noise_reduction import apply_gaussian_filter, apply_median_filter
from src.preprocessing.enhancement import apply_histogram_equalization, apply_contrast_stretching
from src.feature_extraction.color_features import extract_color_histogram
from src.feature_extraction.texture_features import extract_glcm_features, extract_lbp_features
from src.feature_extraction.shape_features import extract_hog_features
from src.classification.evaluation import evaluate_model
from src.utils.visualization import plot_confusion_matrix

# Set page configuration
st.set_page_config(page_title="Tubes PCD", layout="wide")

# Buat direktori jika belum ada
for directory in ['data/raw', 'data/processed', 'data/features', 'models', 'results']:
    os.makedirs(directory, exist_ok=True)

# Path ke dataset train
DATASET_FOLDER_PATH = "data/dataset_gambar" 

# Initialize session state variables
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'extracted_features' not in st.session_state:
    st.session_state.extracted_features = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

# ---LOAD DATASET DAN EXTRAKSI FITUR DATASET---
def load_and_preprocess_dataset(dataset_path, apply_noise_reduction, noise_reduction_method, kernel_size, sigma,
                                apply_enhancement, enhancement_method, min_percentile, max_percentile,
                                target_size=(128, 128)): # target size untuk resize gambar
    """Load gambar dari subdirektori/Dataset, preprocess, kemudian simpan label classnya"""
    data = []
    labels = []
    
    if not os.path.exists(dataset_path):
        st.error(f"Path ke Dataset tidak ditemukan: {dataset_path}. Please check the path.")
        return None, None, None

    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    if not class_names:
        st.warning(f"Tidak ditemukan class dalam {dataset_path}. Please organize your dataset into class folders.")
        return None, None, None

    st.info(f"Loading images from {dataset_path} with {len(class_names)} classes...")
    progress_bar = st.progress(0)
    
    total_images = 0
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            total_images += len(os.listdir(class_path))
    
    if total_images == 0:
        st.warning(f"Tidak ada gambar dalam dataset path: {dataset_path}. Please add images to class folders.")
        return None, None, None

    processed_count = 0
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize gambar untuk konsistensi
                    img = cv2.resize(img, target_size)

                    # Terapkan preprocessing pada dataset
                    processed_img = img.copy()
                    if apply_noise_reduction:
                        if noise_reduction_method == "Gaussian Filter":
                            processed_img = apply_gaussian_filter(processed_img, kernel_size, sigma)
                        elif noise_reduction_method == "Median Filter":
                            processed_img = apply_median_filter(processed_img, kernel_size)
                    
                    if apply_enhancement:
                        if enhancement_method == "Histogram Equalization":
                            processed_img = apply_histogram_equalization(processed_img)
                        elif enhancement_method == "Contrast Stretching":
                            processed_img = apply_contrast_stretching(processed_img, min_percentile, max_percentile)
                    
                    data.append(processed_img)
                    labels.append(class_name)
                    processed_count += 1
                    # Update progress bar only if total_images is not zero to avoid division by zero
                    if total_images > 0:
                        progress_bar.progress(processed_count / total_images)

                except Exception as e:
                    st.error(f"Error processing {img_path}: {e}")
                    continue
    st.success(f"Selesai load dan preprocessing {len(data)} gambar.")
    return np.array(data), np.array(labels), class_names

def extract_features_from_dataset(images, use_color_histogram, color_histogram_bins,
                                  use_glcm, use_lbp, use_hog, hog_orientations, hog_pixels_per_cell):
    """Ekstraksi fitur untuk gambar2 di dataset"""
    all_features = []
    feature_names = []
    
    st.info("Mengekstrak fitur gambar dari dataset...")
    progress_bar = st.progress(0)
    
    for i, image in enumerate(images):
        features_single_image = []
        
        if use_color_histogram:
            color_hist = extract_color_histogram(image, bins=color_histogram_bins)
            features_single_image.extend(color_hist)
        
        if use_glcm:
            glcm_features = extract_glcm_features(image)
            features_single_image.extend(glcm_features)
        
        if use_lbp:
            lbp_features = extract_lbp_features(image)
            features_single_image.extend(lbp_features)
        
        if use_hog:
            # Check if the image is suitable for HOG (needs to be grayscale)
            if len(image.shape) == 3 and image.shape[2] == 3: # If RGB
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else: # Already grayscale
                image_gray = image
            
            hog_features = extract_hog_features(image_gray, orientations=hog_orientations, 
                                                 pixels_per_cell=(hog_pixels_per_cell, hog_pixels_per_cell))
            features_single_image.extend(hog_features)
                
        all_features.append(features_single_image)
        progress_bar.progress((i + 1) / len(images))
        
    st.success(f"Selesai mengekstraksi fitur untuk {len(all_features)} gambar.")

    # Generate feature names dynamically based on what was extracted
    temp_feature_names = []
    if use_color_histogram and all_features:
        temp_feature_names.extend([f'color_hist_{j}' for j in range(len(extract_color_histogram(images[0], bins=color_histogram_bins)))])
    if use_glcm and all_features:
        temp_feature_names.extend([f'glcm_{j}' for j in range(len(extract_glcm_features(images[0])))])
    if use_lbp and all_features:
        temp_feature_names.extend([f'lbp_{j}' for j in range(len(extract_lbp_features(images[0])))])
    if use_hog and all_features:
        sample_img_for_hog = images[0]
        if len(sample_img_for_hog.shape) == 3 and sample_img_for_hog.shape[2] == 3:
            sample_img_for_hog = cv2.cvtColor(sample_img_for_hog, cv2.COLOR_RGB2GRAY)
        temp_feature_names.extend([f'hog_{j}' for j in range(len(extract_hog_features(sample_img_for_hog, orientations=hog_orientations, pixels_per_cell=(hog_pixels_per_cell, hog_pixels_per_cell))))])
    
    feature_names = temp_feature_names
    
    return np.array(all_features), feature_names


# App title and description
st.title("Aplikasi Digital Image Processing & Classification | Kelompok 14 🏎️")
st.write("""
App ini digunakan untuk memproses gambar digital dan klasifikasi jenis mobil dengan model KNN
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload", "Preprocessing", "Feature Extraction", "Classification", "Results"])

# Upload Page
if page == "Upload":
    st.header("Image Upload")
    
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.session_state.original_image = image
        
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Menyimpan gambar ke direktori data/raw
        pil_image = Image.fromarray(image)
        img_path = f"data/raw/{uploaded_file.name}"
        pil_image.save(img_path)
        st.success(f"Hasil gambar tersimpan di: {img_path}")

# Preprocessing Page
elif page == "Preprocessing":
    st.header("Image Preprocessing")
    
    if 'original_image' not in st.session_state:
        st.warning("Lakukan preprocess gambar terlebih dahulu...")
    else:
        st.subheader("Original Image")
        st.image(st.session_state.original_image, use_container_width=True)
        
        st.subheader("Preprocessing Options")
        
        preprocessing_col1, preprocessing_col2 = st.columns(2)
        
        with preprocessing_col1:
            st.write("Noise Reduction")
            apply_noise_reduction = st.checkbox("Apply Noise Reduction")
            
            noise_reduction_method = st.selectbox(
                "Noise Reduction Method",
                ["Gaussian Filter", "Median Filter"],
                disabled=not apply_noise_reduction
            )
            
            if noise_reduction_method == "Gaussian Filter":
                kernel_size = st.slider("Kernel Size", 1, 15, 5, step=1)
                sigma = st.slider("Sigma", 0.1, 10.0, 1.0, step=0.1)
            elif noise_reduction_method == "Median Filter":
                kernel_size = st.slider("Kernel Size", 1, 15, 5, step=1)
        
        with preprocessing_col2:
            st.write("Image Enhancement")
            apply_enhancement = st.checkbox("Apply Enhancement")
            
            enhancement_method = st.selectbox(
                "Enhancement Method",
                ["Histogram Equalization", "Contrast Stretching"],
                disabled=not apply_enhancement
            )
            
            if enhancement_method == "Contrast Stretching":
                min_percentile = st.slider("Min Percentile", 0, 100, 2)
                max_percentile = st.slider("Max Percentile", 0, 100, 98)
        
        # Eksekusi preprocessing lalu tampilkan hasil 
        if st.button("Apply Preprocessing"):
            processed_image = st.session_state.original_image.copy()
            
            if apply_noise_reduction:
                if noise_reduction_method == "Gaussian Filter":
                    processed_image = apply_gaussian_filter(processed_image, kernel_size, sigma)
                elif noise_reduction_method == "Median Filter":
                    processed_image = apply_median_filter(processed_image, kernel_size)
            
            if apply_enhancement:
                if enhancement_method == "Histogram Equalization":
                    processed_image = apply_histogram_equalization(processed_image)
                elif enhancement_method == "Contrast Stretching":
                    processed_image = apply_contrast_stretching(processed_image, min_percentile, max_percentile)
            
            st.session_state.processed_image = processed_image
            
            st.subheader("Processed Image")
            st.image(processed_image, use_container_width=True)
            
            processed_path = "data/processed/processed_image.jpg"
            Image.fromarray(processed_image).save(processed_path)
            st.success(f"Hasil gambar tersimpan di: {processed_path}")

# --- MODIFIED FEATURE EXTRACTION PAGE ---
elif page == "Feature Extraction":
    st.header("Feature Extraction")
    
    # Kondisi jika belum dilakukan preproses gambar
    if 'processed_image' not in st.session_state or st.session_state.processed_image is None:
        st.warning("Lakukan preprocess gambar terlebih dahulu...")
    else:
        st.subheader("Processed Image")
        st.image(st.session_state.processed_image, use_container_width=True)
        
        st.subheader("Feature Extraction Options")
        
        # Ambil nilai default dari Classification Page untuk konsistensi
        # Pastikan nilai-nilai ini sama persis dengan yang ada di Classification Page
        default_color_histogram_bins = 128
        default_hog_orientations = 16
        default_hog_pixels_per_cell = 8

        feature_col1, feature_col2 = st.columns(2)
        
        # Ekstraksi fitur warna gambar (Color Histogram)
        with feature_col1:
            st.write("Color Features")
            use_color_histogram = st.checkbox("Color Histogram", value=True)
            if use_color_histogram:
                st.info(f"Menggunakan Color Histogram Bins: {default_color_histogram_bins} (fixed)")
                color_histogram_bins = default_color_histogram_bins 

        # Ekstraksi fitur tekstur gambar (GLCM)
        with feature_col2:
            st.write("Texture Features")
            use_glcm = st.checkbox("GLCM Features", value=True)
            # use_lbp = st.checkbox("Local Binary Patterns", value=False)
        
        # Ekstraksi fitur bentuk gambar (HOG)
        st.write("Shape Features")
        use_hog = st.checkbox("Histogram of Oriented Gradients (HOG)", value=True)
        if use_hog:
            st.info(f"Menggunakan HOG Orientations: {default_hog_orientations} (fixed)")
            st.info(f"Menggunakan HOG Pixels Per Cell: {default_hog_pixels_per_cell} (fixed)")
            hog_orientations = default_hog_orientations 
            hog_pixels_per_cell = default_hog_pixels_per_cell 
            
        # Eksekusi ekstraksi fitur dan tampilkan hasil dalam barplot
        if st.button("Extract Features"):
            image = st.session_state.processed_image
            features = []
            feature_names = []
            
            if use_color_histogram:
                color_hist = extract_color_histogram(image, bins=color_histogram_bins)
                features.extend(color_hist)
                feature_names.extend([f'color_hist_{i}' for i in range(len(color_hist))])
            
            if use_glcm:
                glcm_features = extract_glcm_features(image)
                features.extend(glcm_features)
                feature_names.extend([f'glcm_{i}' for i in range(len(glcm_features))])
            
            # if use_lbp:
            #     lbp_features = extract_lbp_features(image)
            #     features.extend(lbp_features)
            #     feature_names.extend([f'lbp_{i}' for i in range(len(lbp_features))])
            
            if use_hog:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    image_gray = image
                
                hog_features = extract_hog_features(image_gray, orientations=hog_orientations, 
                                                     pixels_per_cell=(hog_pixels_per_cell, hog_pixels_per_cell))
                features.extend(hog_features)
                feature_names.extend([f'hog_{i}' for i in range(len(hog_features))])
            
            st.session_state.extracted_features = features
            st.session_state.feature_names = feature_names
            
            st.subheader("Feature Summary")
            st.write(f"Jumlah fitur yang terekstraksi: {len(features)}")
            
            features_df = pd.DataFrame([features], columns=feature_names)
            features_df.to_csv("data/features/extracted_features.csv", index=False)
            st.success("Features saved to data/features/extracted_features.csv")
            
            st.subheader("Feature Visualization")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(feature_names[:20], features[:20])
            ax.set_xticklabels(feature_names[:20], rotation=90)
            ax.set_ylabel("Feature Value")
            ax.set_title("First 20 Features")
            st.pyplot(fig)

# --- CLASSIFICATION PAGE ---
elif page == "Classification":
    st.header("Klasifikasi Jenis Mobil dengan K-Nearest Neighbors (KNN)")

    # Auto-load Dataset lalu Ekstraksi fiturnya
    # Default preprocessing and feature extraction settings for the dataset
    default_apply_noise_reduction_ds = True
    default_noise_reduction_method_ds = "Gaussian Filter"
    default_kernel_size_ds = 3
    default_sigma_ds = 0.5
    
    default_apply_enhancement_ds = False
    default_enhancement_method_ds = "Contrast Stretching"
    default_min_percentile_ds = 3
    default_max_percentile_ds = 98

    default_target_size_ds = 128 # Resize gambar menjadi 128x128 agar konsisten

    default_use_color_histogram_ds = True
    default_color_histogram_bins_ds = 128
    default_use_glcm_ds = True
    default_use_lbp_ds = False
    default_use_hog_ds = True
    default_hog_orientations_ds = 16
    default_hog_pixels_per_cell_ds = 8

    # Check if dataset has already been loaded and features extracted
    if 'X_dataset' not in st.session_state or st.session_state.X_dataset is None:
        st.info("Loading dan ekstraksi fitur dari dataset...")
        
        # Load dan preprocess dataset
        dataset_images, dataset_labels, class_names = load_and_preprocess_dataset(
            DATASET_FOLDER_PATH, 
            default_apply_noise_reduction_ds, default_noise_reduction_method_ds, default_kernel_size_ds, default_sigma_ds, 
            default_apply_enhancement_ds, default_enhancement_method_ds, default_min_percentile_ds, default_max_percentile_ds,
            target_size=(default_target_size_ds, default_target_size_ds)
        )
        
        if dataset_images is not None:
            st.session_state.dataset_images = dataset_images
            st.session_state.dataset_labels = dataset_labels
            st.session_state.class_names = class_names

            # Extract features dari dataset
            X_dataset, feature_names_dataset = extract_features_from_dataset(
                st.session_state.dataset_images,
                default_use_color_histogram_ds, default_color_histogram_bins_ds,
                default_use_glcm_ds, default_use_lbp_ds, default_use_hog_ds, 
                default_hog_orientations_ds, default_hog_pixels_per_cell_ds
            )
            st.session_state.X_dataset = X_dataset
            st.session_state.feature_names_dataset = feature_names_dataset
            st.success(f"Dataset features loaded and extracted: {X_dataset.shape[0]} samples, {X_dataset.shape[1]} features.")
        else:
            st.error("Failed to load or process the dataset. Please ensure the path is correct and contains images.")
            st.stop() 

    else:
        st.success(f"Using pre-loaded dataset features: {st.session_state.X_dataset.shape[0]} samples, {st.session_state.X_dataset.shape[1]} features.")

    # --- Training section untuk model KNN ---
    if 'X_dataset' in st.session_state and st.session_state.X_dataset is not None and \
       'dataset_labels' in st.session_state and st.session_state.dataset_labels is not None:
        
        st.subheader("Parameter KNN Classifier")
        
        # KNN parameters
        n_neighbors = st.slider("Number of Neighbors (k)", 1, 20, 3, key='n_neighbors_clf')
        weights = st.selectbox("Weight Function", ["uniform", "distance"], key='weights_clf')
        
        if st.button("Train KNN Classifier pada loaded Dataset"):
            X = st.session_state.X_dataset
            y = st.session_state.dataset_labels
            class_names = st.session_state.class_names
            
            label_to_int = {name: i for i, name in enumerate(class_names)}
            y_encoded = np.array([label_to_int[label] for label in y])

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
            
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            
            st.info("Training KNN classifier...")
            classifier.fit(X_train, y_train)
            st.success("KNN Classifier training complete!")
            
            y_pred = classifier.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            st.session_state.trained_model = classifier
            st.session_state.evaluation_results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm,
                'class_names': class_names 
            }
            
            model_filename = 'models/trained_knn_classifier.pkl' # Nama file model spesifik untuk KNN
            with open(model_filename, 'wb') as f:
                pickle.dump(classifier, f)
            st.success(f"KNN model trained and saved as {model_filename}!")
            
            st.subheader("Training Results Overview")
            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"Precision: {precision:.4f}")
            st.write(f"Recall: {recall:.4f}")
            st.write(f"F1 Score: {f1:.4f}")
            
            st.info("Go to Results tab for detailed evaluation, including confusion matrix.")

        # --- Prediksi gambar yang diupload ---
        st.subheader("Class Prediksi untuk Gambar yang diupload dengan model KNN")
        if 'extracted_features' in st.session_state and st.session_state.extracted_features is not None and \
           'trained_model' in st.session_state and st.session_state.trained_model is not None:
            
            current_image_features = np.array(st.session_state.extracted_features).reshape(1, -1)
            trained_classifier = st.session_state.trained_model
            
            if current_image_features.shape[1] != st.session_state.X_dataset.shape[1]:
                st.warning("Warning: The number of features for the uploaded image does not match the features used for training the model. This might lead to incorrect predictions or errors.")
                st.write(f"Uploaded image features: {current_image_features.shape[1]}, Dataset features: {st.session_state.X_dataset.shape[1]}")
                st.info("Please ensure the same feature extraction options are selected for both the single image and the dataset.")
            else:
                try:
                    predicted_label_index = trained_classifier.predict(current_image_features)[0]
                    if 'class_names' in st.session_state and st.session_state.class_names:
                        predicted_class_name = st.session_state.class_names[predicted_label_index]
                        st.write(f"Prediksi jenis mobil dari gambar yang diupload: **{predicted_class_name}**")
                    else:
                        st.write(f"Prediksi jenis mobil dari gambar yang diupload: **Class {predicted_label_index}** (Class names not available)")
                    
                    if hasattr(trained_classifier, "predict_proba"):
                        probs = trained_classifier.predict_proba(current_image_features)[0]
                        st.write("Class probabilities:")
                        if 'class_names' in st.session_state and st.session_state.class_names:
                            prob_df = pd.DataFrame({
                                'Class': st.session_state.class_names,
                                'Probability': probs
                            }).sort_values(by='Probability', ascending=False)
                            st.dataframe(prob_df)
                        else:
                            st.write(probs)
                except Exception as e:
                    st.error(f"Error during single image prediction: {e}")
                    st.info("Please ensure a KNN model has been trained and feature dimensions match.")

        else:
            st.info("Upload an image, preprocess, and extract features on their respective pages to get a prediction for it after training.")
            if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
                st.info("No KNN model trained yet. The dataset will be loaded and features extracted automatically.")
    else:
        st.warning("Dataset not loaded or features not extracted. Please ensure the `DATASET_FOLDER_PATH` is correct.")

# Results Page (unchanged)
elif page == "Results":
    st.header("Classification Results")
    
    if 'evaluation_results' not in st.session_state or st.session_state.evaluation_results is None:
        st.warning("Please train a classifier first on the 'Classification' page.")
    else:
        results = st.session_state.evaluation_results
        
        st.subheader("Performance Metrics")
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
            st.metric("Precision", f"{results['precision']:.4f}")
        
        with metrics_col2:
            st.metric("Recall", f"{results['recall']:.4f}")
            st.metric("F1 Score", f"{results['f1']:.4f}")
        
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))

        plot_confusion_matrix(results['confusion_matrix'], 
                                classes=results['class_names'], 
                                normalize=False, 
                                title='Confusion Matrix', 
                                ax=ax)

        st.pyplot(fig)
        
        # # Feature Importance (KNN does not have feature_importances_)
        # # This block will now effectively not run as KNN does not provide feature_importances_
        # if hasattr(st.session_state.trained_model, 'feature_importances_'):
        #     st.subheader("Feature Importance (Not available for KNN)")
        #     st.info("K-Nearest Neighbors (KNN) does not provide feature importance scores directly.")
        
        st.subheader("Export Results")
        
        if st.button("Export Results to CSV"):
            results_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Value': [results['accuracy'], results['precision'], results['recall'], results['f1']]
            })
            
            results_path = "results/classification_results.csv"
            results_df.to_csv(results_path, index=False)
            st.success(f"Results exported to {results_path}")
            
        if st.button("Save Confusion Matrix"):
            cm_path = "results/confusion_matrix.png"
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_confusion_matrix(results['confusion_matrix'], 
                                classes=results['class_names'],
                                normalize=False,
                                title='Confusion Matrix',
                                ax=ax)
            plt.savefig(cm_path)
            st.success(f"Confusion matrix saved to {cm_path}")