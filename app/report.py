"""Printable one-page report of one AI result (A4, PDF or PNG).

Built with matplotlib only (already installed), so no new library is needed.
Contains: case info, the X-ray and the Grad-CAM heatmap, the prediction, confidence, urgency,
the class probabilities, the AI explanation, worklist status, a box for the radiologist's
decision, and the disclaimer. No patient name or ID is stored in this system, so the report
shows only the case number.
"""
import io
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DISPLAY = {"COVID": "COVID-19", "Lung_Opacity": "Lung Opacity",
           "Normal": "Normal", "Viral Pneumonia": "Viral Pneumonia"}
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
DISCLAIMER = ("This report was produced by an AI tool that assists radiologists and does not replace "
              "them. It is limited by its training data, is not clinically validated and is not a "
              "medical diagnosis. The final decision belongs to the radiologist.")
RED, GREY, DARK = "#c0392b", "#9aa5b1", "#1f2933"


def _img(x):
    if isinstance(x, Image.Image):
        return np.asarray(x.convert("RGB"))
    if isinstance(x, (str, bytes)) or hasattr(x, "__fspath__"):
        return np.asarray(Image.open(x).convert("RGB"))
    return np.asarray(x)


def _short_time(iso):
    if not iso:
        return "-"
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return iso


def build_report(image, heatmap, result, urgency, explanation="", case=None, fmt="pdf"):
    """Return the report as bytes. fmt: 'pdf' or 'png'.
    result: predict()-style dict (class, confidence, probabilities, model_name, warnings).
    case: optional stored case row (case_id, uploaded_at, status, radiologist_label, read_at)."""
    fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait, inches
    fig.patch.set_facecolor("white")

    # ---- header
    fig.text(0.06, 0.955, "AI-Assisted Chest X-ray Report", fontsize=18, weight="bold", color=DARK)
    created = datetime.now().astimezone().strftime("%d %b %Y, %H:%M")
    sub = f"Generated {created}   |   Model: {result.get('model_name', 'unknown')}"
    if case:
        sub = f"Case #{case['case_id']}   |   Arrived {_short_time(case.get('uploaded_at'))}   |   " + sub
    fig.text(0.06, 0.935, sub, fontsize=8.5, color="#52606d")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.925, 0.925], color=DARK, lw=1))

    # ---- images
    for k, (arr, title) in enumerate([(_img(image), "Chest X-ray"),
                                      (_img(heatmap), "Grad-CAM heatmap (approximate)")]):
        ax = fig.add_axes([0.06 + k * 0.45, 0.635, 0.43, 0.26])
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        ax.set_title(title, fontsize=9, color=DARK)
        ax.axis("off")

    # ---- result box
    cls = result["class"]
    fig.text(0.06, 0.605, "AI prediction", fontsize=9, color="#52606d")
    fig.text(0.06, 0.578, DISPLAY.get(cls, cls), fontsize=17, weight="bold",
             color=RED if cls == "COVID" else DARK)
    fig.text(0.40, 0.605, "Confidence", fontsize=9, color="#52606d")
    fig.text(0.40, 0.578, f"{result['confidence']:.1%}", fontsize=17, weight="bold", color=DARK)
    fig.text(0.66, 0.605, "Urgency score (0-1)", fontsize=9, color="#52606d")
    fig.text(0.66, 0.578, f"{urgency:.2f}", fontsize=17, weight="bold", color=DARK)

    # ---- probabilities
    ax = fig.add_axes([0.24, 0.43, 0.66, 0.12])
    probs = [result["probabilities"][c] for c in CLASSES]
    ax.barh(range(4), probs, color=[RED if c == cls else GREY for c in CLASSES], height=0.6)
    ax.set_yticks(range(4), [DISPLAY[c] for c in CLASSES], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=8)
    ax.tick_params(axis="x", labelsize=7.5)
    for i, p in enumerate(probs):
        inside = p > 0.85
        ax.text(p - 0.01 if inside else p + 0.01, i, f"{p:.1%}", va="center",
                ha="right" if inside else "left", fontsize=7.5, color="white" if inside else DARK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.06, 0.56, "Class probabilities", fontsize=9, weight="bold", color=DARK)

    # ---- explanation + warnings
    y = 0.385
    fig.text(0.06, y, "AI explanation", fontsize=9, weight="bold", color=DARK)
    body = explanation or "No explanation available."
    lines = []
    for para in body.split("\n"):
        # the warnings and the disclaimer are printed separately below
        if para.startswith(("Warnings about the input image", "This tool assists radiologists")):
            continue
        lines += textwrap.wrap(para, 110)
    for w in result.get("warnings") or []:
        lines += textwrap.wrap(f"WARNING (image check): {w}", 105)
    for line in lines[:9]:
        y -= 0.017
        fig.text(0.06, y, line, fontsize=8, color=DARK,
                 weight="bold" if line.startswith("WARNING") else "normal")

    # ---- worklist status and radiologist box (fixed place at the bottom)
    y = 0.2
    fig.add_artist(plt.Rectangle((0.06, 0.112), 0.88, 0.115, fill=False, lw=0.8,
                                 edgecolor="#52606d", transform=fig.transFigure))
    fig.text(0.08, y, "Radiologist review", fontsize=9, weight="bold", color=DARK)
    if case and case.get("status") == "done":
        decision = (f"Final label: {DISPLAY.get(case.get('radiologist_label'), case.get('radiologist_label'))}"
                    f"      Read at: {_short_time(case.get('read_at'))}")
    else:
        status = case.get("status", "not saved") if case else "not saved to worklist"
        decision = f"Status: {status}.      Final label: ______________________"
    fig.text(0.08, y - 0.03, decision, fontsize=8.5, color=DARK)
    fig.text(0.08, y - 0.065, "Radiologist name / signature: ______________________________",
             fontsize=8.5, color=DARK)

    # ---- disclaimer
    for k, line in enumerate(textwrap.wrap(DISCLAIMER, 115)):
        fig.text(0.06, 0.085 - k * 0.015, line, fontsize=7.5, style="italic", color="#52606d")
    fig.text(0.06, 0.03, "Istanbul Arel University · Interdisciplinary Engineering Project · Fall 2026",
             fontsize=7, color="#7b8794")

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=150 if fmt == "png" else None)
    plt.close(fig)
    return buf.getvalue()
