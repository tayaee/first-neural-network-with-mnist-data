import streamlit as st
from resample import get_new_samples
from predict import MNISTPredictor

st.set_page_config(page_title="MNIST Classifier")

if "predictor" not in st.session_state:
    st.session_state.predictor = MNISTPredictor()
if "samples" not in st.session_state:
    st.session_state.samples = get_new_samples(10)
if "predictions" not in st.session_state:
    st.session_state.predictions = [None] * 10  # type: ignore
if "results" not in st.session_state:
    st.session_state.results = [None] * 10  # type: ignore


def refresh_samples():
    st.session_state.samples = get_new_samples(10)
    st.session_state.predictions = [None] * 10  # type: ignore
    st.session_state.results = [None] * 10  # type: ignore


st.title("MNIST Hands-on Neural Network")

st.markdown("https://github.com/tayaee/first-neural-network-with-mnist-data")
correct = sum(1 for r in st.session_state.results if r == "correct")
wrong = sum(1 for r in st.session_state.results if r == "wrong")
unpredicted = sum(1 for r in st.session_state.results if r is None)
st.subheader(f"Correct: {correct}, Wrong: {wrong}, Unpredicted: {unpredicted}")

if st.button("Re-sample"):
    refresh_samples()
    st.rerun()

cols = st.columns(5)
for i in range(10):
    with cols[i % 5]:
        st.image(st.session_state.samples[i]["image"], width=100)

        # Determine button label
        if st.session_state.predictions[i] is not None:
            button_label = str(st.session_state.predictions[i])
        else:
            button_label = "Predict #"

        if st.button(
            button_label,
            disabled=st.session_state.predictions[i] is not None,
            key=f"btn_{i}",
        ):
            pred, conf = st.session_state.predictor.predict(
                st.session_state.samples[i]["flat"]
            )
            actual = st.session_state.samples[i]["label"]

            st.session_state.predictions[i] = pred  # type: ignore
            if pred == actual:
                st.session_state.results[i] = "correct"  # type: ignore
                st.success(f"Correct! ({pred})")
            else:
                st.session_state.results[i] = "wrong"  # type: ignore
                st.error(f"Wrong! Pred:{pred} (Actual:{actual})")
            st.rerun()
