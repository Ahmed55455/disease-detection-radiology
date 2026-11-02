"""Streamlit UI only. Logic lives in inference.py (model) and db.py (database).

Run from the repository root:  streamlit run app/app.py
"""
import io
from datetime import datetime

import pandas as pd
import streamlit as st

import db
from inference import (InvalidImageError, analyze, ask_result, explain_result, load_image,
                       make_report, result_from_case)

st.set_page_config(page_title="Chest X-ray Assistant", page_icon="🩻", layout="wide")
db.init_db()

# Display names are for the screen only; code and database keep the exact class strings
# (docs/interfaces.md v0.2, section 1).
DISPLAY = {"COVID": "COVID-19", "Lung_Opacity": "Lung Opacity",
           "Normal": "Normal", "Viral Pneumonia": "Viral Pneumonia"}
STATUS_ICON = {"pending": "🕒 pending", "in_review": "🔍 in review", "done": "✅ done"}
SOURCE_ICON = {"web": "💻", "phone": "📱", "api": "🔌", "demo": "🧪"}
API_PORT = 8000            # the phone page is served by the FastAPI app (app/api.py)


def local_ip():
    """IP address of this laptop on the Wi-Fi (what the phone must open)."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))          # no data is sent; only picks the network card
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def qr_png(text):
    """QR code image (PNG bytes), or None if the 'qrcode' library is not installed."""
    try:
        import qrcode
    except ImportError:
        return None
    buf = io.BytesIO()
    qrcode.make(text, box_size=6, border=2).save(buf, format="PNG")
    return buf.getvalue()


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
        "from": SOURCE_ICON.get(r.get("source"), ""),
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


EXAMPLE_QUESTIONS = ["Why this result?", "How sure is the model?", "What does the heat map show?",
                     "What are the model's limits?"]


def explanation_box(res, key):
    """AI explanation of one result + questions about it (model/explain.py)."""
    with st.container(border=True):
        st.markdown("#### 🤖 AI explanation")
        try:
            st.write(explain_result(res))
        except Exception as e:                       # never break the page
            st.error(f"Explanation not available: {e}")
            return
        st.caption("Ask about this result. The assistant explains the model output only; it does "
                   "not diagnose or advise on treatment.")
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, q in zip(cols, EXAMPLE_QUESTIONS):
            if col.button(q, key=f"{key}-{q}"):
                st.session_state[f"{key}-q"] = q
        q = st.text_input("Your question", key=f"{key}-q",
                          placeholder="e.g. What is the second most likely class?")
        if q:
            try:
                out = ask_result(res, q)
                if out["source"] == "refused":
                    st.warning(out["answer"])
                else:
                    st.info(out["answer"])
                st.caption(f"Answer source: {out['source']}")
            except Exception as e:
                st.error(f"Could not answer: {e}")


def report_buttons(image, heatmap, res, case, key):
    """Download the result as a printable PDF report or as a PNG image."""
    cache_key = f"report-{key}-{case['status'] if case else 'unsaved'}"
    if cache_key not in st.session_state:               # build once, not on every click
        with st.spinner("Preparing the report..."):
            st.session_state[cache_key] = {fmt: make_report(image, heatmap, res, case, fmt)
                                           for fmt in ("pdf", "png")}
    files = st.session_state[cache_key]
    name = f"case_{case['case_id']}" if case else "xray_result"
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("🖨️ Report (PDF)", files["pdf"], file_name=f"{name}_report.pdf",
                       mime="application/pdf", key=f"{cache_key}-pdf", width="stretch")
    c2.download_button("🖼️ Report (PNG)", files["png"], file_name=f"{name}_report.png",
                       mime="image/png", key=f"{cache_key}-png", width="stretch")


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
    with st.expander("📱 Phone capture"):
        phone_url = f"http://{local_ip()}:{API_PORT}/mobile"
        qr = qr_png(phone_url)
        if qr:
            st.image(qr, caption="Scan with the phone camera", width=170)
        st.code(phone_url, language=None)
        st.caption("The phone must be on the same Wi-Fi, and the API must be running:  \n"
                   "`uvicorn app.api:app --host 0.0.0.0 --port 8000`")
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
            explanation_box(res, key=f"up-{key}")

            saved = st.session_state["saved"]
            report_buttons(img, res["heatmap"], res, db.get_case(saved) if saved else None,
                           key=f"up-{key}")
            if saved:
                st.success(f"Saved as case #{saved}. "
                           "Open the **Worklist** to see its place in the queue.")
            elif res["warnings"]:
                # Possibly not a chest X-ray: do not add it by default (priority is never
                # changed automatically; the user decides whether it is a real X-ray).
                st.error("⛔ **This may not be a chest X-ray.** The model always picks one of its "
                         "4 classes, even for an image that is not an X-ray, so this prediction is "
                         "not reliable. It was not added to the worklist.")
                if st.button("It is a chest X-ray: add anyway"):
                    st.session_state["saved"] = db.save_case(img, res, source="web")
                    st.rerun()
            elif st.button("➕ Add to worklist", type="primary"):
                st.session_state["saved"] = db.save_case(img, res, source="web")
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
                "from": cmp["source"].map(lambda x: SOURCE_ICON.get(x, "")),
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
                   "warning (priority is NOT changed). From: 💻 laptop, 📱 phone, 🧪 demo. "
                   "The radiologist decides.")

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
        with st.expander("🤖 Explain this result"):
            explanation_box(result_from_case(row), key=f"case-{row['case_id']}")
        report_buttons(row["image_path"], row["heatmap_path"], result_from_case(row), row,
                       key=f"case-{row['case_id']}")
        with st.popover("🗑️ Remove this case"):
            st.write("Only for test or invalid images (for example a photo that is not an X-ray). "
                     "The case and its images are deleted.")
            if st.button("Yes, remove it", key=f"del-{row['case_id']}", type="primary"):
                db.delete_case(row["case_id"])
                st.rerun()

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
