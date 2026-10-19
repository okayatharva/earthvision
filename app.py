
import streamlit as st
from PIL import Image

from src.predict import load_model, predict_image


st.set_page_config(
    page_title="EarthVision | Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ EarthVision")
st.subheader("Satellite Image Classification with PyTorch")

st.write(
    "Upload an RGB satellite image to predict its likely "
    "land-use or land-cover category using a convolutional "
    "neural network."
)

st.info(
    "This demo uses the EuroSAT RGB dataset. It classifies "
    "optical satellite images and is not a SAR/radar classifier."
)

with st.expander("About the project"):
    st.write(
        "EarthVision is a computer vision project built with "
        "PyTorch, Torchvision, scikit-learn, and Streamlit. "
        "The model is trained on EuroSAT RGB imagery and reports "
        "its three highest-scoring categories."
    )


@st.cache_resource
def get_model_bundle():
    return load_model()


try:
    model, class_names, image_transform, device = get_model_bundle()
    st.caption(f"Model loaded successfully. Inference device: {device}")

except FileNotFoundError:
    st.warning(
        "The trained model has not been created yet. Run "
        "`python -m src.train` in your project Terminal, "
        "then refresh this page."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.image(
                image,
                caption="Uploaded satellite image",
                use_container_width=True,
            )

        with right:
            st.subheader("Prediction")

            with st.spinner("Analysing image..."):
                results = predict_image(
                    image,
                    model,
                    class_names,
                    image_transform,
                    device,
                )

            best = results[0]

            st.metric(
                "Top predicted class",
                best["class"].replace("_", " ").title(),
                f"{best['probability']:.1%} model score",
            )

            st.write("Top three predictions")

            for result in results:
                label = result["class"].replace("_", " ").title()
                probability = result["probability"]

                st.write(f"**{label}** — {probability:.1%}")
                st.progress(probability)

            st.caption(
                "Scores are model confidence estimates, not verified "
                "probabilities of real-world land use. Performance "
                "may be lower on images unlike the training dataset."
            )

    except Exception as error:
        st.error(f"Could not process this image: {error}")

else:
    st.write(
        "Upload an image to get started. Images resembling "
        "EuroSAT satellite imagery are most suitable for this demo."
    )
