"""Streamlit UI only. Logic lives in inference.py (model) and db.py (database).

Run from the repository root:  streamlit run app/app.py
"""
import io

import pandas as pd
import streamlit as st

import db
from inference import InvalidImageError, analyze, load_image

st.set_page_config(page_title="Chest X-ray Assistant", layout="wide")
db.init_db()

st.title("AI-Assisted Chest X-ray Analysis")
st.warning(
    "This tool assists radiologists and does not replace them. "
    "It is limited by its training data and is not a medical diagnosis."
)

def highlight_critical(row):
    """Red background for rows the AI predicts as COVID (highest severity)."""
    style = "background-color: rgba(220, 53, 69, 0.18)" if row["predicted_class"] == "COVID" else ""
    return [style] * len(row)


page = st.sidebar.radio("Page", ["Upload & Analyze", "Worklist", "Export"])

# ---------------------------------------------------------------- Upload
if page == "Upload & Analyze":
    file = st.file_uploader("Upload a chest X-ray (PNG or JPG, max 10 MB)",
                            type=["png", "jpg", "jpeg"])
    if file:
        key = f"{file.name}-{file.size}"
        if st.session_state.get("key") != key:          # analyze once per file
            st.session_state.update(key=key, img=None, res=None, saved=None, error=None)
            try:
                img = load_image(file.name, file.getvalue())
                with st.spinner("Analyzing..."):
                    res = analyze(img)
                st.session_state.update(img=img, res=res)
            except InvalidImageError as e:
                st.session_state["error"] = str(e)
            except Exception as e:                       # model error: no crash
                st.session_state["error"] = f"Analysis failed: {e}"

        if st.session_state["error"]:
            st.error(st.session_state["error"])
        else:
            img, res = st.session_state["img"], st.session_state["res"]
            for w in res["warnings"]:
                st.warning(f"Image check: {w}")

            col1, col2 = st.columns(2)
            col1.image(img, caption="Original", width=400)
            col2.image(res["heatmap"], caption="Grad-CAM heatmap (approximate)", width=400)

            st.subheader(f"Prediction: {res['class']}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{res['confidence']:.1%}")
            m2.metric("Urgency score", f"{db.urgency_score(res['probabilities']):.2f}")
            m3.metric("Model", res["model_name"])
            st.bar_chart(pd.Series(res["probabilities"]))

            if st.session_state["saved"]:
                st.success(f"Saved as case #{st.session_state['saved']}")
            elif st.button("Add to worklist"):
                st.session_state["saved"] = db.save_case(img, res)
                st.rerun()

# ---------------------------------------------------------------- Worklist
elif page == "Worklist":
    st.header("Radiologist Worklist")
    view = st.radio("Reading order", ["AI priority", "FIFO (arrival order)", "Compare FIFO vs AI"],
                    horizontal=True)
    show_done = st.checkbox("Show done cases")
    rows = db.get_worklist(include_done=show_done,
                           order="fifo" if view.startswith("FIFO") else "ai")
    if not rows:
        st.info("No cases.")
    else:
        if view == "Compare FIFO vs AI":
            st.info("Visual illustration of the reading order only. "
                    "Waiting-time results come from the simulation study.")
            st.caption("Position 1 is read first. 'moved_up' = how many places earlier "
                       "the AI order reads the case compared with FIFO.")
            df = pd.DataFrame(db.compare_orders(include_done=show_done))
            st.dataframe(df.style.apply(highlight_critical, axis=1), hide_index=True)
            crit = df[df["predicted_class"] == "COVID"]
            if len(crit):
                a, b = st.columns(2)
                a.metric("COVID cases: average position (FIFO)",
                         f"{crit['fifo_position'].mean():.1f}")
                b.metric("COVID cases: average position (AI)",
                         f"{crit['ai_position'].mean():.1f}",
                         delta=f"{crit['ai_position'].mean() - crit['fifo_position'].mean():.1f}",
                         delta_color="inverse")
        else:
            df = pd.DataFrame(rows)[["case_id", "uploaded_at", "predicted_class",
                                     "confidence", "urgency_score", "status",
                                     "radiologist_label", "read_at"]]
            df.insert(0, "position", range(1, len(df) + 1))
            st.dataframe(df.style.apply(highlight_critical, axis=1), hide_index=True)
        st.caption("Red rows: AI predicts COVID (most urgent class). The radiologist decides.")

        case_id = st.selectbox("Case", df["case_id"])
        row = db.get_case(int(case_id))
        c1, c2 = st.columns(2)
        c1.image(row["image_path"], caption="Original", width=400)
        c2.image(row["heatmap_path"], caption="Grad-CAM heatmap (approximate)", width=400)
        st.write(f"**Status:** {row['status']}  |  **AI prediction:** {row['predicted_class']}"
                 f" ({row['confidence']:.1%})")

        try:
            if row["status"] == "pending":
                if st.button("Start review"):
                    db.start_review(row["case_id"])
                    st.rerun()
            elif row["status"] == "in_review":
                label = st.selectbox("Radiologist label (final decision)", db.CLASSES,
                                     index=db.CLASSES.index(row["predicted_class"]))
                b1, b2 = st.columns(2)
                if b1.button("Mark as done"):
                    db.mark_done(row["case_id"], label)
                    st.rerun()
                if b2.button("Back to pending"):
                    db.set_status(row["case_id"], "pending")
                    st.rerun()
            else:
                st.success(f"Read at {row['read_at']}, label: {row['radiologist_label']}")
        except ValueError as e:
            st.error(str(e))

# ---------------------------------------------------------------- Export
else:
    st.header("Export worklist to CSV (for the simulation)")
    st.write("Columns: " + ", ".join(db.EXPORT_COLUMNS) + ". All cases, including done.")
    csv_text = db.export_csv()
    st.dataframe(pd.read_csv(io.StringIO(csv_text)), hide_index=True)
    st.download_button("Download worklist.csv", csv_text, file_name="worklist.csv",
                       mime="text/csv")
