"""Streamlit UI only. Logic lives in inference.py (model) and db.py (database).

Run from the repository root:  streamlit run app/app.py
"""
import io
from datetime import datetime

import pandas as pd
import streamlit as st

import db
from inference import InvalidImageError, analyze, load_image

st.set_page_config(page_title="Chest X-ray Assistant", page_icon="🩻", layout="wide")
db.init_db()

# Display names are for the screen only; code and database keep the exact class strings
# (docs/interfaces.md v0.2, section 1).
DISPLAY = {"COVID": "COVID-19", "Lung_Opacity": "Lung Opacity",
           "Normal": "Normal", "Viral Pneumonia": "Viral Pneumonia"}
STATUS_ICON = {"pending": "🕒 pending", "in_review": "🔍 in review", "done": "✅ done"}


def show(cls):
    return DISPLAY.get(cls, cls) if cls else ""


def short_time(iso):
    """'2026-10-21T09:22:58+03:00' -> '21 Oct 09:22' (screen only; stored value stays ISO)."""
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %H:%M")
    except ValueError:
        return iso


def minutes_between(start, end):
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
    except (TypeError, ValueError):
        return None


def highlight_critical(row):
    """Red background for rows the AI predicts as COVID (highest severity)."""
    crit = row.get("AI prediction") == DISPLAY["COVID"]
    return ["background-color: rgba(220, 53, 69, 0.15)" if crit else ""] * len(row)


def case_table(rows, with_position=True):
    """Readable table for the screen."""
    df = pd.DataFrame([{
        "case": r["case_id"],
        "⚠️": "⚠️" if r.get("warnings") else "",
        "arrived": short_time(r["uploaded_at"]),
        "AI prediction": show(r["predicted_class"]),
        "confidence": r["confidence"],
        "urgency": r["urgency_score"],
        "status": STATUS_ICON.get(r["status"], r["status"]),
        "radiologist": show(r["radiologist_label"]),
    } for r in rows])
    if with_position:
        df.insert(0, "#", range(1, len(df) + 1))
    return df


TABLE_CONFIG = {
    "confidence": st.column_config.NumberColumn("confidence", format="percent"),
    "urgency": st.column_config.ProgressColumn("urgency", min_value=0.0, max_value=1.0,
                                               format="%.2f"),
}

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("## 🩻 Chest X-ray Assistant")
    st.caption("AI triage for a radiologist worklist")
    page = st.radio("Page", ["Dashboard", "Upload & Analyze", "Worklist", "Export"],
                    label_visibility="collapsed")
    st.divider()
    st.caption("**Model:** EfficientNet-B0 (provisional)  \n"
               "**Classes:** COVID-19, Lung Opacity, Normal, Viral Pneumonia  \n"
               "**Explanation:** Grad-CAM (approximate)")
    st.caption("Istanbul Arel University · Interdisciplinary Engineering Project · Fall 2026")

st.warning("⚕️ This tool assists radiologists and does not replace them. It is limited by "
           "its training data and is not a medical diagnosis.")

# ---------------------------------------------------------------- Dashboard
if page == "Dashboard":
    st.title("Dashboard")
    rows = db.get_worklist(include_done=True)
    if not rows:
        st.info("No cases yet. Go to **Upload & Analyze** to add the first X-ray.")
    else:
        df = pd.DataFrame(rows)
        counts = df["status"].value_counts()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total cases", len(df))
        k2.metric("🕒 Pending", int(counts.get("pending", 0)))
        k3.metric("🔍 In review", int(counts.get("in_review", 0)))
        k4.metric("✅ Done", int(counts.get("done", 0)))
        k5.metric("⚠️ Flagged", int(df["warnings"].fillna("").astype(bool).sum()))

        left, right = st.columns(2)
        with left:
            st.subheader("Cases by AI prediction")
            by_class = (df["predicted_class"].map(show).value_counts()
                        .reindex([DISPLAY[c] for c in db.CLASSES], fill_value=0))
            st.bar_chart(by_class, horizontal=True, x_label="cases", y_label="")
        with right:
            st.subheader("Next cases to read (AI priority)")
            nxt = db.get_worklist()[:5]
            if nxt:
                st.dataframe(case_table(nxt)[["#", "case", "AI prediction", "urgency", "status"]],
                             hide_index=True, column_config=TABLE_CONFIG,
                             width="stretch")
            else:
                st.success("All cases are done.")

        done = df[df["status"] == "done"].copy()
        st.subheader("Reading results")
        r1, r2, r3 = st.columns(3)
        if len(done):
            done["ttd"] = [minutes_between(u, r) for u, r in zip(done["uploaded_at"],
                                                                 done["read_at"])]
            agree = (done["radiologist_label"] == done["predicted_class"]).mean()
            r1.metric("Radiologist agreed with AI", f"{agree:.0%}",
                      help="Share of done cases where the radiologist label equals the AI prediction.")
            ttd = done["ttd"].dropna()
            r2.metric("Average time to diagnosis", f"{ttd.mean():.0f} min" if len(ttd) else "-",
                      help="read_at minus uploaded_at, for done cases.")
            crit = done[done["predicted_class"] == "COVID"]["ttd"].dropna()
            r3.metric("… for COVID-19 cases", f"{crit.mean():.0f} min" if len(crit) else "-")
        else:
            r1.info("No case is done yet. Results appear after the radiologist reads cases.")
        st.caption("Live numbers from this demo database, for illustration. The waiting-time "
                   "study (FIFO vs AI) is done with the simulation model.")

# ---------------------------------------------------------------- Upload
elif page == "Upload & Analyze":
    st.title("Upload & Analyze")
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
                st.warning(f"⚠️ Image check: {w}")

            col1, col2 = st.columns(2)
            col1.image(img, caption="Original", width=400)
            col2.image(res["heatmap"], caption="Grad-CAM heatmap (approximate)", width=400)

            st.subheader(f"AI prediction: {show(res['class'])}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{res['confidence']:.1%}")
            m2.metric("Urgency score", f"{db.urgency_score(res['probabilities']):.2f}")
            m3.metric("Model", res["model_name"])
            probs = pd.Series({show(c): p for c, p in res["probabilities"].items()})
            st.bar_chart(probs, horizontal=True, x_label="probability", y_label="")

            if st.session_state["saved"]:
                st.success(f"Saved as case #{st.session_state['saved']}. "
                           "Open the **Worklist** to see its place in the queue.")
            elif st.button("➕ Add to worklist", type="primary"):
                st.session_state["saved"] = db.save_case(img, res)
                st.rerun()

# ---------------------------------------------------------------- Worklist
elif page == "Worklist":
    st.title("Radiologist Worklist")
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
            cmp = pd.DataFrame(db.compare_orders(include_done=show_done))
            crit = cmp[cmp["predicted_class"] == "COVID"]
            if len(crit):
                a, b, c = st.columns(3)
                a.metric("COVID-19 cases: average position (FIFO)",
                         f"{crit['fifo_position'].mean():.1f}")
                b.metric("COVID-19 cases: average position (AI)",
                         f"{crit['ai_position'].mean():.1f}",
                         delta=f"{crit['ai_position'].mean() - crit['fifo_position'].mean():.1f}",
                         delta_color="inverse")
                c.metric("Places moved up (average)", f"{crit['moved_up'].mean():.1f}")
            table = pd.DataFrame({
                "case": cmp["case_id"],
                "⚠️": cmp["flag"],
                "arrived": cmp["uploaded_at"].map(short_time),
                "AI prediction": cmp["predicted_class"].map(show),
                "urgency": cmp["urgency_score"],
                "FIFO position": cmp["fifo_position"],
                "AI position": cmp["ai_position"],
                "moved up": cmp["moved_up"],
            })
            st.dataframe(table.style.apply(highlight_critical, axis=1), hide_index=True,
                         column_config=TABLE_CONFIG, width="stretch")
            st.caption("Position 1 is read first. 'moved up' = how many places earlier the AI "
                       "order reads the case compared with FIFO.")
            ids = list(cmp["case_id"])
        else:
            table = case_table(rows)
            st.dataframe(table.style.apply(highlight_critical, axis=1), hide_index=True,
                         column_config=TABLE_CONFIG, width="stretch")
            ids = list(table["case"])
        st.caption("Red rows: AI predicts COVID-19 (most urgent class). ⚠️ = image check "
                   "warning (priority is NOT changed). The radiologist decides.")

        st.divider()
        case_id = st.selectbox("Open case", ids, format_func=lambda i: f"Case #{i}")
        row = db.get_case(int(case_id))
        c1, c2 = st.columns(2)
        c1.image(row["image_path"], caption="Original", width=400)
        c2.image(row["heatmap_path"], caption="Grad-CAM heatmap (approximate)", width=400)
        st.write(f"**Status:** {STATUS_ICON.get(row['status'], row['status'])}  |  "
                 f"**AI prediction:** {show(row['predicted_class'])} ({row['confidence']:.1%})  |  "
                 f"**Arrived:** {short_time(row['uploaded_at'])}  |  "
                 f"**Model:** {row['model_name'] or 'unknown'}")
        for w in db.warnings_list(row):
            st.warning(f"⚠️ Image check: {w} The case keeps its priority; please check the image.")

        try:
            if row["status"] == "pending":
                if st.button("🔍 Start review", type="primary"):
                    db.start_review(row["case_id"])
                    st.rerun()
            elif row["status"] == "in_review":
                label = st.selectbox("Radiologist label (final decision)", db.CLASSES,
                                     index=db.CLASSES.index(row["predicted_class"]),
                                     format_func=show)
                b1, b2 = st.columns(2)
                if b1.button("✅ Mark as done", type="primary"):
                    db.mark_done(row["case_id"], label)
                    st.rerun()
                if b2.button("↩️ Back to pending"):
                    db.set_status(row["case_id"], "pending")
                    st.rerun()
            else:
                st.success(f"Read at {short_time(row['read_at'])}, radiologist label: "
                           f"{show(row['radiologist_label'])}")
        except ValueError as e:
            st.error(str(e))

# ---------------------------------------------------------------- Export
else:
    st.title("Export for the simulation")
    st.write("Columns: `" + ", ".join(db.EXPORT_COLUMNS) + "`. All cases, including done. "
             "Times are ISO 8601.")
    csv_text = db.export_csv()
    st.dataframe(pd.read_csv(io.StringIO(csv_text)), hide_index=True, width="stretch")
    st.download_button("⬇️ Download worklist.csv", csv_text, file_name="worklist.csv",
                       mime="text/csv", type="primary")
