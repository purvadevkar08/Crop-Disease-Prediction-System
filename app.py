import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, T5ForConditionalGeneration, T5TokenizerFast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Load Models (once at start)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Load T5
t2t_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
t2t_tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-small")
t2t_model.eval()

# Classes
classes = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy",
    "Potato___Bacterial_spot"
]
text_prompts = [f"This is a photo of {c.replace('___', ' ')}" for c in classes]

# -----------------------------
# Rule-based pesticide suggestions
# -----------------------------
pesticide_recommendations = {
    "Potato___Early_blight": {
        "pesticide": "Spray Chlorothalonil (2 g/L) or Mancozeb (2.5 g/L) every 7-10 days.",
        "care": "Use certified disease-free seeds and avoid water stagnation."
    },
    "Potato___Late_blight": {
        "pesticide": "Spray Metalaxyl + Mancozeb (2 g/L) or Cymoxanil (1.5 g/L) every 10 days.",
        "care": "Remove infected leaves and avoid overhead irrigation."
    },
    "Potato___Bacterial_spot": {
        "pesticide": "Apply Copper Oxychloride (3 g/L) or Streptocycline (0.1 g/L) weekly.",
        "care": "Use crop rotation and ensure field sanitation."
    },
    "Potato___Healthy": {
        "pesticide": "No pesticide required.",
        "care": "Maintain good irrigation and monitor for disease symptoms regularly."
    }
}

def get_rule_based_recommendation(disease_name):
    rec = pesticide_recommendations.get(disease_name, None)
    if rec:
        return f"**Pesticide Suggestion:** {rec['pesticide']}\n\n**Care Tips:** {rec['care']}"
    else:
        return "No specific recommendation found. Please consult an agricultural expert."

# -----------------------------
# Helper functions
# -----------------------------
def predict_clip(image):
    inputs = clip_processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    idx = probs.argmax()
    predicted_class = classes[idx]
    return predicted_class, probs

def generate_recommendation(disease_name):
    prompt = f"Suggest pesticide and management recommendations for {disease_name} in potato plants."
    inputs = t2t_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = t2t_model.generate(**inputs, max_new_tokens=120)
    return t2t_tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("🌿 Potato Leaf Disease Detector & Pesticide Recommendation System")
st.write("Upload a potato leaf image to predict the disease and get pesticide recommendations.")

uploaded_file = st.file_uploader("📁 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # 1️⃣ CLIP Disease Prediction
    disease, probs = predict_clip(image)
    st.subheader(f"🦠 Predicted Disease: {disease.replace('___', ' ')}")

    # Show probabilities
    prob_df = pd.DataFrame({
        "Disease Class": [c.replace("___", " ") for c in classes],
        "Probability (%)": [round(p*100, 2) for p in probs]
    }).sort_values("Probability (%)", ascending=False)

    st.subheader("📊 Prediction Probabilities")
    st.table(prob_df)

    # Bar plot
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x="Probability (%)", y="Disease Class", data=prob_df, palette="viridis", ax=ax)
    ax.set_title("Model Confidence by Disease Class")
    st.pyplot(fig)

    # 2️⃣ Rule-based recommendation
    st.subheader("🌾 Verified Pesticide Recommendation")
    rule_based_text = get_rule_based_recommendation(disease)
    st.markdown(rule_based_text)

    # 3️⃣ AI-generated (T5) recommendation
    st.subheader("🤖 AI-Generated Management Advice")
    recommendation = generate_recommendation(disease.replace("___", " "))
    st.write(recommendation)
