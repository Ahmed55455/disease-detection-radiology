import sys
sys.path.append("model")

import pandas as pd
import streamlit as st
from PIL import Image

import db
from predict import predict

st.set_page_config(page_title="Chest X-ray Assistant", layout="wide")
db.init_db()

st.title("AI-Assisted Chest X-ray Analysis")
st.warning(
    "This tool assists radiologists and does not replace them. "
    "It is limited by its training data and is not a medical diagnosis."
)

page = st.sidebar.radio("Page", ["Upload & Analyze", "Worklist"])

if page == "Upload & Analyze":
    file = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])
    if file:
        key = f"{file.name}-{file.size}"
        if st.session_state.get("key") != key:       # analyze once per file
            img = Image.open(file).convert("RGB")
            with st.spinner("Analyzing..."):
                res = predict(img)
            st.session_state.update(key=key, img=img, res=res, saved=None)

        img, res = st.session_state["img"], st.session_state["res"]
        probs = res["probabilities"]

        col1, col2 = st.columns(2)
        col1.image(img, caption="Original", width=400)
        col2.image(res["heatmap"], caption="Grad-CAM heatmap", width=400)

        st.subheader(f"Prediction: {res['class']}")
        st.metric("Confidence", f"{probs[res['class']]:.1%}")
        st.metric("Urgency score", f"{db.urgency_score(probs):.2f}")
        st.bar_chart(pd.Series(probs))

        if st.session_state["saved"]:
            st.success(f"Saved as case #{st.session_state['saved']}")
        elif st.button("Add to worklist"):
            st.session_state["saved"] = db.save_case(img, res)
            st.rerun()

else:
    st.header("Radiologist Worklist (highest urgency first)")
    rows = db.get_worklist()
    if not rows:
        st.info("No pending cases.")
    else:
        df = pd.DataFrame(rows)[["case_id", "uploaded_at", "predicted_class",
                                 "confidence", "urgency_score", "status"]]
        st.dataframe(df, hide_index=True)
        case_id = st.selectbox("Case", df["case_id"])
        c1, c2 = st.columns(2)
        row = next(r for r in rows if r["case_id"] == case_id)
        c1.image(row["image_path"], caption="Original", width=400)
        c2.image(row["heatmap_path"], caption="Grad-CAM heatmap", width=400)
        if st.button("Mark as read"):
            db.mark_done(case_id)
            st.rerun()