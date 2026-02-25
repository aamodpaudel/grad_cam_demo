import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import os

# CAM Imports
from pytorch_grad_cam import (GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
                              AblationCAM, XGradCAM, EigenCAM, LayerCAM, FullGrad)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ViT Explainability Import
VIT_EXPLAIN_AVAILABLE = False
try:
    from vit_grad_rollout import VITAttentionGradRollout # Assumes vit_grad_rollout.py is in the same directory
    VIT_EXPLAIN_AVAILABLE = True
except ImportError:
    # Warning will be shown in the sidebar if import fails
    pass

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        cnn_base = resnet50(weights=None) 
        self.cnn = nn.Sequential(*list(cnn_base.children())[:-2])
        
        self.feature_projection = nn.Conv2d(
            in_channels=2048,
            out_channels=768, 
            kernel_size=1
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.positional_embedding = nn.Parameter(torch.randn(1, 49 + 1, 768)) 
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=3072,
                activation="gelu",
                batch_first=False 
            ),
            num_layers=4
        )
        
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x): 
        # --- DEBUG PRINTS to investigate torch.cat error ---
        print(f"HYBRID_FORWARD_DEBUG: Input x.shape: {x.shape}")

        features_after_cnn = self.cnn(x) 
        print(f"HYBRID_FORWARD_DEBUG: After self.cnn, features_after_cnn.shape: {features_after_cnn.shape}")

        features_after_projection = self.feature_projection(features_after_cnn) 
        print(f"HYBRID_FORWARD_DEBUG: After self.feature_projection, features_after_projection.shape: {features_after_projection.shape}")
        
        features_for_vit = features_after_projection.flatten(2).permute(0, 2, 1) # [B, 49, 768]
        print(f"HYBRID_FORWARD_DEBUG: After flatten & permute (features_for_vit).shape: {features_for_vit.shape}")
        
        cls_tokens_expanded = self.cls_token.expand(x.shape[0], -1, -1) # [B, 1, 768]
        print(f"HYBRID_FORWARD_DEBUG: cls_tokens_expanded.shape (based on x.shape[0]): {cls_tokens_expanded.shape}")
        
        concatenated_features = torch.cat((cls_tokens_expanded, features_for_vit), dim=1) # [B, 50, 768]
        print(f"HYBRID_FORWARD_DEBUG: After torch.cat, concatenated_features.shape: {concatenated_features.shape}")
        
        concatenated_features += self.positional_embedding
        
        permuted_for_transformer = concatenated_features.permute(1, 0, 2)  # (50, B, 768)
        print(f"HYBRID_FORWARD_DEBUG: After permute for transformer, permuted_for_transformer.shape: {permuted_for_transformer.shape}")
        
        transformer_output = self.transformer(permuted_for_transformer) # (50, B, 768)
        print(f"HYBRID_FORWARD_DEBUG: After self.transformer, transformer_output.shape: {transformer_output.shape}")

        cls_output = transformer_output[0] # CLS token output [B, 768]
        print(f"HYBRID_FORWARD_DEBUG: cls_output.shape: {cls_output.shape}")

        final_output = self.classifier(cls_output)
        print(f"HYBRID_FORWARD_DEBUG: final_output.shape: {final_output.shape}")
        return final_output

def reshape_transform_cam_vit_layers(tensor):
    if tensor.ndim == 3 and tensor.shape[0] == 50: 
        tensor_permuted = tensor.permute(1, 0, 2) 
        image_tokens = tensor_permuted[:, 1:, :]    
        batch_size, num_patches, C = image_tokens.shape
        side_len = int(num_patches**0.5)
        if side_len * side_len != num_patches: 
            st.warning(f"CAM Reshape (S,B,E): Cannot make square from {num_patches} patches. Check tensor shape {tensor.shape}")
            return tensor 
        return image_tokens.reshape(batch_size, side_len, side_len, C).permute(0, 3, 1, 2)
    elif tensor.ndim == 3 and tensor.shape[1] == 50: 
        image_tokens = tensor[:, 1:, :]  
        batch_size, num_patches, C = image_tokens.shape
        side_len = int(num_patches**0.5)
        if side_len * side_len != num_patches:
            st.warning(f"CAM Reshape (B,S,E): Cannot make square from {num_patches} patches. Check tensor shape {tensor.shape}")
            return tensor
        return image_tokens.reshape(batch_size, side_len, side_len, C).permute(0, 3, 1, 2)

    st.warning(f"CAM Reshape: Tensor shape {tensor.shape} not directly handled for ViT layer reshape. CAM might be incorrect.")
    return tensor

def analyze_facial_features(image):
    image_np = np.array(image.convert('RGB')) 
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    try:
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            st.warning(f"Could not load haarcascade. Using direct image analysis.")
            return analyze_image_directly(image_bgr)
    except Exception as e: 
        st.warning(f"Face detector error: {e}. Using direct image analysis.")
        return analyze_image_directly(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    if len(faces) == 0:
        st.warning("Using approx image analysis.")
        return analyze_image_directly(image_bgr)
    x_face, y_face, w_face, h_face = faces[0] 
    face_region = gray[y_face:y_face+h_face, x_face:x_face+w_face]
    if face_region.size == 0: return analyze_image_directly(image_bgr)
    face_height, _ = face_region.shape
    eye_region = face_region[0:int(face_height/3), :]
    if eye_region.size == 0: eye_white_ratio = 0.0
    else:
        _, eye_binary = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
        eye_white_ratio = np.sum(eye_binary == 255) / (eye_region.size + 1e-10)
    mouth_region = face_region[int(face_height/2):, :]
    if mouth_region.size == 0: mouth_dark_ratio = 0.0
    else:
        _, mouth_binary = cv2.threshold(mouth_region, 50, 255, cv2.THRESH_BINARY_INV)
        mouth_dark_ratio = np.sum(mouth_binary == 255) / (mouth_region.size + 1e-10)
    ear = max(0, min(1, eye_white_ratio * 2))
    mar = max(0, min(1, mouth_dark_ratio * 3))
    return ear, mar

def analyze_image_directly(image_bgr): 
    if len(image_bgr.shape) > 2 and image_bgr.shape[2] == 3: gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif len(image_bgr.shape) == 2: gray = image_bgr
    else: return 0.5, 0.5 
    height, _ = gray.shape
    eye_region = gray[0:int(height/3), :]
    if eye_region.size == 0: eye_std_norm, eye_edge_density = 0.0, 0.0
    else:
        eye_std_norm = (np.std(eye_region) / 128.0) * 0.5 
        eye_edges = cv2.Canny(eye_region, 100, 200)
        eye_edge_density = np.sum(eye_edges == 255) / (eye_region.size + 1e-10)
    mouth_region = gray[int(height/2):, :]
    if mouth_region.size == 0: mouth_std_norm, mouth_edge_density = 0.0, 0.0
    else:
        mouth_std_norm = (np.std(mouth_region) / 128.0) * 0.5 
        mouth_edges = cv2.Canny(mouth_region, 100, 200)
        mouth_edge_density = np.sum(mouth_edges == 255) / (mouth_region.size + 1e-10)
    ear = max(0, min(1, eye_std_norm + eye_edge_density * 2))
    mar = max(0, min(1, mouth_std_norm + mouth_edge_density * 2))
    return ear, mar

def process_image(pil_image):
    ear, mar = analyze_facial_features(pil_image) 
    return ear, mar

def plot_emotion(x_coord, y_coord, emotion_name_main, specific_emotion_label):
    fig, ax = plt.subplots(figsize=(8, 8))
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='-')
    ax.add_patch(circle)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Valence', fontsize=10, fontweight='bold')
    ax.set_ylabel('Arousal', fontsize=10, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    quad_labels = {'Q1 (e.g. Happy)': (0.7, 0.7), 'Q2 (e.g. Angry)': (-0.7, 0.7), 
                   'Q3 (e.g. Sad)': (-0.7, -0.7), 'Q4 (e.g. Relaxed)': (0.7, -0.7)}
    for label, pos in quad_labels.items(): ax.text(pos[0], pos[1], label, fontsize=9, ha='center', va='center')
    for angle_deg_major in range(0, 360, 45):
        rad_major = math.radians(angle_deg_major)
        x_line, y_line = math.cos(rad_major), math.sin(rad_major)
        ax.plot([0, x_line], [0, y_line], 'k--', alpha=0.2)
        if angle_deg_major % 90 == 0: ax.text(1.1*x_line, 1.1*y_line, f"{angle_deg_major}°", fontsize=7, ha='center', va='center')
    emotion_plot_labels = [
        (15, "Pleased"), (45, "Happy"), (75, "Excited"), (105, "Annoyed"), (135, "Angry"), 
        (165, "Enraged"), (195, "Bored"), (225, "Sad"), (255, "Depressed"), (285, "Calm"), 
        (315, "Relaxed"), (345, "Content")]
    for angle_deg, text in emotion_plot_labels:
        rad = math.radians(angle_deg)
        rot = angle_deg-90 if (90 < angle_deg < 270) else angle_deg+90
        ax.text(0.85*math.cos(rad), 0.85*math.sin(rad), text, fontsize=7, ha='center', va='center', rotation=rot, rotation_mode='anchor')
    ax.scatter(x_coord, y_coord, s=100, color='red', zorder=5, label=f"({x_coord:.2f}, {y_coord:.2f})")
    ax.plot([0, x_coord], [0, y_coord], color='red', linestyle='-', alpha=0.7)
    angle_plot = math.degrees(math.atan2(y_coord, x_coord))
    if angle_plot < 0: angle_plot += 360
    ax.set_title(f'Emotion: {emotion_name_main} ({specific_emotion_label})\nAngle: {angle_plot:.1f}°', fontsize=11)
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3); plt.tight_layout()
    return fig

@st.cache_resource
def load_model(model_path, num_classes, device):
    model = HybridCNNViT(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model

def main():
    st.set_page_config(page_title="Pet Sentiment Analyzer", layout="wide")
    st.title("🐾 Pet Sentiment Analyzer")
    st.write("Upload a pet face image to analyze its emotion and see model explanations.")
    
    if not VIT_EXPLAIN_AVAILABLE: 
        st.sidebar.error("`vit_grad_rollout.py` not found. ViT Attention Rollout methods are disabled.")

    st.sidebar.title("About")
    st.sidebar.info("Hybrid CNN-ViT model for pet sentiment analysis with CAM and ViT explainability.")
    st.sidebar.title("Emotion Classification Guide")
    st.sidebar.markdown("- Q1 (Top-Right): Happy, Excited\n- Q2 (Top-Left): Angry, Annoyed\n- Q3 (Bottom-Left): Sad, Bored\n- Q4 (Bottom-Right): Relaxed, Calm")

    st.sidebar.title("⚙️ Explainability")
    explain_method_options = ["GradCAM", "HiResCAM", "ScoreCAM", "GradCAMPlusPlus", 
                              "AblationCAM", "XGradCAM", "EigenCAM", "LayerCAM"]
    if VIT_EXPLAIN_AVAILABLE:
        explain_method_options.extend(["ViT Attention Rollout", "ViT Gradient Rollout"])
    
    selected_explain_method = st.sidebar.selectbox("Choose Method:", explain_method_options)

    cam_target_layer_option = "CNN: Last Conv Block" 
    if selected_explain_method not in ["ViT Attention Rollout", "ViT Gradient Rollout"]:
        cam_target_layer_option = st.sidebar.selectbox(
            "Target Layer for CAM:",
            ("CNN: Last Conv Block", "ViT: Last Transformer Norm (after FFN)", "ViT: Last Transformer Norm (after Attention)"),
            key="cam_layer_select"
        )
    
    vit_rollout_discard_ratio = 0.9
    if selected_explain_method in ["ViT Attention Rollout", "ViT Gradient Rollout"] and VIT_EXPLAIN_AVAILABLE:
        vit_rollout_discard_ratio = st.sidebar.slider("ViT Discard Ratio:", 0.0, 1.0, 0.9, 0.05, key="vit_discard")
        # head_fusion widget is removed as it's not a parameter in the user's provided vit_grad_rollout.py's __init__

    use_aug_smooth = st.sidebar.checkbox("Enable AugSmooth (CAMs only)", False, key="cam_augsmooth")
    use_eigen_smooth = st.sidebar.checkbox("Enable EigenSmooth (CAMs only)", False, key="cam_eigensmooth")

    uploaded_file = st.file_uploader("Choose an image file (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        pil_image_original = Image.open(uploaded_file).convert('RGB')
        
        col_orig_img_display, col_explain_img_display = st.columns(2)
        with col_orig_img_display:
            st.image(pil_image_original, caption="Uploaded Image", use_container_width=True)
        
        transform_for_model = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor_for_model = transform_for_model(pil_image_original).unsqueeze(0)
        
        vis_image_pil_resized_for_explain = pil_image_original.resize((224,224))
        vis_image_numpy_for_explain = np.array(vis_image_pil_resized_for_explain) / 255.0

        current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor_for_model = input_tensor_for_model.to(current_device)

        try:
            # Use a workspace-relative path so the app can find the model when run from the repo
            model_file_path = os.path.join(os.path.dirname(__file__), 'pet_sentiment_model.pth')
            pet_model = load_model(model_file_path, num_classes=4, device=current_device)
        except FileNotFoundError:
            st.error(f"Model file not found: {model_file_path}. Update path if needed."); return
        except Exception as e_model_load: 
            st.error(f"Error loading model: {e_model_load}")
            st.error("Please check the model file, PyTorch compatibility, and ensure the model definition in this app matches the one used for training.")
            return
        
        predicted_class_index = -1 
        try:
            st.write("Running model inference...") 
            with torch.no_grad(): 
                model_output = pet_model(input_tensor_for_model)
                _, predicted_idx = torch.max(model_output, 1)
                predicted_class_index = predicted_idx.item()
            st.write(f"Model Inference successful. Predicted class index: {predicted_class_index}") 
        except RuntimeError as e_runtime: 
            st.error(f"RuntimeError during model inference: {e_runtime}")
            st.error("Check terminal for DEBUG prints from the model's forward pass (lines starting with HYBRID_FORWARD_DEBUG:).")
            st.error(f"DEBUG_INFO: Shape of input_tensor_for_model passed to model: {input_tensor_for_model.shape}")
            return 

        explanation_image = None
        explanation_caption = selected_explain_method

        if predicted_class_index != -1: # Proceed only if inference was successful
            try:
                if selected_explain_method in ["ViT Attention Rollout", "ViT Gradient Rollout"]:
                    if not VIT_EXPLAIN_AVAILABLE:
                        st.error("ViT Explainability library (vit_grad_rollout.py) not loaded.")
                    else:
                        # VITAttentionGradRollout is called without head_fusion
                        # It will use its internal default for attention_layer_name ('attn_drop')
                        grad_rollout_explainer = VITAttentionGradRollout(
                            pet_model, 
                            discard_ratio=vit_rollout_discard_ratio
                        )
                        
                        if selected_explain_method == "ViT Attention Rollout":
                            st.info("Note: Your `vit_grad_rollout.py` seems to perform a gradient-based rollout. "
                                    "Displaying result for the predicted class.")
                        
                        # The __call__ method of user's VITAttentionGradRollout always expects category_index
                        attention_mask = grad_rollout_explainer(input_tensor_for_model, category_index=predicted_class_index)
                        
                        if attention_mask.shape != vis_image_numpy_for_explain.shape[:2]:
                            attention_mask_resized = cv2.resize(attention_mask, (vis_image_numpy_for_explain.shape[1], vis_image_numpy_for_explain.shape[0]))
                        else:
                            attention_mask_resized = attention_mask

                        explanation_image = show_cam_on_image(vis_image_numpy_for_explain, attention_mask_resized, use_rgb=True)
                        explanation_caption = f"{selected_explain_method} (Class: {predicted_class_index})"

                else: # CAM Methods
                    selected_target_layers_cam = None
                    reshape_fn_cam = None 
                    if cam_target_layer_option == "CNN: Last Conv Block":
                        selected_target_layers_cam = [pet_model.cnn[-1]]
                    elif cam_target_layer_option == "ViT: Last Transformer Norm (after FFN)":
                        selected_target_layers_cam = [pet_model.transformer.layers[-1].norm2]
                        reshape_fn_cam = reshape_transform_cam_vit_layers 
                    elif cam_target_layer_option == "ViT: Last Transformer Norm (after Attention)":
                        selected_target_layers_cam = [pet_model.transformer.layers[-1].norm1]
                        reshape_fn_cam = reshape_transform_cam_vit_layers
                    
                    if selected_target_layers_cam:
                        CAM_ALGORITHMS_MAP = {"GradCAM": GradCAM, "HiResCAM": HiResCAM, "ScoreCAM": ScoreCAM, 
                                            "GradCAMPlusPlus": GradCAMPlusPlus, "AblationCAM": AblationCAM, 
                                            "XGradCAM": XGradCAM, "EigenCAM": EigenCAM, "LayerCAM": LayerCAM}
                        ChosenCAMAlgorithm = CAM_ALGORITHMS_MAP[selected_explain_method]
                        cam_targets = [ClassifierOutputTarget(predicted_class_index)]

                        with ChosenCAMAlgorithm(model=pet_model, target_layers=selected_target_layers_cam, 
                                                reshape_transform=reshape_fn_cam) as cam_instance:
                            grayscale_cam = cam_instance(input_tensor=input_tensor_for_model, targets=cam_targets,
                                                        aug_smooth=use_aug_smooth, eigen_smooth=use_eigen_smooth)
                            grayscale_cam = grayscale_cam[0, :] 
                            explanation_image = show_cam_on_image(vis_image_numpy_for_explain, grayscale_cam, use_rgb=True)
                            explanation_caption = f"{selected_explain_method} on {cam_target_layer_option}"
                
                if explanation_image is not None:
                    with col_explain_img_display:
                        st.image(explanation_image, caption=explanation_caption, use_container_width=True)
                else:
                    with col_explain_img_display:
                        st.warning("Could not generate explanation for the selected method/layer.")

            except Exception as e_explain:
                with col_explain_img_display:
                    st.error(f"Error generating explanation: {e_explain}")
                    st.caption("Ensure model structure is compatible with the chosen ViT explainability method if selected. "
                               "The `vit_grad_rollout.py` script you are using might have specific expectations for layer names or attention map formats.")
            
        # --- Emotion Processing and Plotting ---
        if predicted_class_index != -1: # Only proceed if prediction was successful
            st.write("---") 
            emotion_class_map = {0: ("Angry", 2), 1: ("Happy", 1), 2: ("Relaxed", 4), 3: ("Sad", 3)}
            main_emotion_name, main_quadrant = emotion_class_map[predicted_class_index]
            ear_value, mar_value = process_image(pil_image_original) 
            current_magnitude = math.sqrt(ear_value**2 + mar_value**2)
            if current_magnitude == 0: norm_x, norm_y = 0.7071, 0.7071 
            else: norm_x, norm_y = ear_value / current_magnitude, mar_value / current_magnitude
            plot_x, plot_y = 0.0, 0.0
            if main_quadrant == 1: plot_x, plot_y = abs(norm_x), abs(norm_y)    
            elif main_quadrant == 2: plot_x, plot_y = -abs(norm_x), abs(norm_y) 
            elif main_quadrant == 3: plot_x, plot_y = -abs(norm_x), -abs(norm_y) 
            else: plot_x, plot_y = abs(norm_x), -abs(norm_y)                    
            plot_angle_deg = math.degrees(math.atan2(plot_y, plot_x))
            if plot_angle_deg < 0: plot_angle_deg += 360
            specific_emotions_angle_map = {
                (0, 30): "Pleased", (30, 60): "Happy", (60, 90): "Excited", (90, 120): "Annoyed", 
                (120, 150): "Angry", (150, 180): "Enraged", (180, 210): "Bored", (210, 240): "Sad", 
                (240, 270): "Depressed", (270, 300): "Calm", (300, 330): "Relaxed", (330, 360): "Content"}
            assigned_specific_emotion = main_emotion_name 
            for (low, upp), emo in specific_emotions_angle_map.items():
                if low <= plot_angle_deg < upp: assigned_specific_emotion = emo; break
            if math.isclose(plot_angle_deg, 0.0) or math.isclose(plot_angle_deg, 360.0):
                 assigned_specific_emotion = "Pleased" if main_quadrant in [1,4] else "Enraged"
            st.write(f"### Sentiment Analysis Results:")
            st.write(f"**Predicted Emotion Category:** {main_emotion_name} (Quadrant {main_quadrant})")
            st.write(f"**Specific Emotion Detail:** {assigned_specific_emotion}")
            col_metrics, col_plot = st.columns(2)
            with col_metrics:
                st.write("#### Facial Metrics (approximate):")
                st.write(f"EAR-like: **{ear_value:.3f}** | MAR-like: **{mar_value:.3f}**")
                st.write(f"Plot Coords: **({plot_x:.3f}, {plot_y:.3f})** | Angle: **{plot_angle_deg:.1f}°**")
            with col_plot:
                emotion_figure = plot_emotion(plot_x, plot_y, main_emotion_name, assigned_specific_emotion)
                st.pyplot(emotion_figure)
        elif predicted_class_index == -1 and uploaded_file: # if inference failed but file was uploaded
             st.error("Could not proceed to full analysis because model prediction failed.")

    else:
        st.info("✨ Please upload an image of a pet face to begin!")

if __name__ == "__main__":
    main()