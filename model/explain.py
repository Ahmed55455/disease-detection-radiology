"""Explanation assistant for predict() results (English only).

It explains the model's output in plain language and answers questions about
ONE case. It never diagnoses, never recommends treatment, and never looks at
the image itself. Only the structured result of predict() is used.

Two layers:
  1. "template" (default): fixed text and keyword rules. Works offline, no
     extra libraries, cannot invent facts.
  2. "ollama" or "gemini" (optional): a language model answers free questions,
     but only from the case data and the facts written in this file. If it fails
     for any reason, the template layer answers instead.

Usage (from Python, repository root):
    from predict import predict
    from explain import explain, ask
    result = predict("xray.png")
    print(explain(result))
    print(ask(result, "How sure is the model?")["answer"])

The `lang` argument is kept for compatibility and is ignored (English only).

Environment variables for the optional layer:
    OLLAMA_MODEL   name of a model you pulled with Ollama (for example "qwen2.5:3b")
    GEMINI_API_KEY and GEMINI_MODEL   for the Gemini API (free tier, needs internet)
Only the text built by build_context() is sent to a language model. The image,
the file name and any patient information are never sent.

Demo:
    python model/explain.py
"""
import json
import os
import re
import urllib.error
import urllib.request

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

DISPLAY = {"COVID": "COVID-19", "Lung_Opacity": "Lung Opacity", "Normal": "Normal",
           "Viral Pneumonia": "Viral Pneumonia"}

CLASS_INFO = {
    "COVID": "The image resembles chest X-rays labeled COVID-19 in the training data.",
    "Lung_Opacity": "Lung opacity means areas of the lungs that look denser (whiter) than normal "
                    "on the X-ray. It has several possible causes, so this label does not name one.",
    "Normal": "The image resembles chest X-rays labeled normal in the training data.",
    "Viral Pneumonia": "The image resembles chest X-rays labeled viral pneumonia in the training data.",
}

DISCLAIMER = ("This tool assists radiologists and does not replace them. It is not a diagnosis. "
              "The final decision belongs to the radiologist.")

# Facts the assistant may use (measured in this project, see docs/model_findings.md).
LIMITATIONS = (
    "Known limits of this model: it was trained on public datasets only and has not been clinically "
    "validated. On the original test set it reaches about 96% accuracy, but it can be wrong with very "
    "high confidence. On new children's X-rays from another source, it rarely missed pneumonia but "
    "often labeled healthy images as abnormal. It always outputs one of four classes, even for an image "
    "that is not a chest X-ray. The heat map is approximate."
)

GRADCAM = (
    "The colored heat map (Grad-CAM) shows the image regions that influenced the prediction most. "
    "Red/yellow means stronger influence. It is low resolution and approximate: it shows where the "
    "model looked, not proof of why it decided, and it may highlight areas outside the lungs."
)

REFUSAL = ("I can't diagnose or advise on treatment or medication. I can only explain what the model "
           "predicted for this image. Please discuss the case with a physician.")

HELP = ("I can explain this result. You can ask about: why this class, how sure the model is, "
        "what a class means, the heat map, the warnings, or the model's limits.")

# Questions that ask for a diagnosis or treatment are refused in code, before any model is called.
REFUSE_RE = re.compile(
    r"(do i have|am i (sick|infected|ill)|should i (take|start|stop)|treat|medicat|medicine|drug|dose|dosage|"
    r"prescri|cure|therapy|surgery|antibiotic|"
    r"(is|are|was) (this|the|that|my) (patient|person|child|man|woman)\b.{0,40}\b(sick|ill|infected|healthy|positive|negative)|"
    r"(does|did|do) (this|the|that|my) (patient|person|child)\b.{0,30}\bhave)",
    re.IGNORECASE,
)


# A language-model answer that mentions causes or findings we never gave it is not used.
UNSUPPORTED_RE = re.compile(
    r"\b(inflammation|infection|infections|infectious|cancer|tumou?rs?|edema|fluid|bleeding|fibrosis|"
    r"scarring|previous conditions|detected|shows? (areas|opacit)|showing (areas|opacit)|appears? to show|"
    r"areas of concern|points? to where|interprets? the x-?ray|high confident)\b",
    re.IGNORECASE,
)


def _name(cls):
    return DISPLAY.get(cls, cls)


def confidence_level(conf):
    if conf >= 0.90:
        return "high"
    if conf >= 0.70:
        return "moderate"
    return "low"


def _conf_text(result):
    conf = result["confidence"]
    lvl = confidence_level(conf)
    pct = f"{conf * 100:.1f}%"
    base = f"The model's confidence in this class is {lvl} ({pct})."
    extra = {
        "high": " High confidence does not mean the label is correct: in our tests the model was wrong "
                "with very high confidence.",
        "moderate": " The model is not fully sure, so human review matters even more.",
        "low": " The model is undecided. This result should not be relied on alone.",
    }[lvl]
    return base + extra


def _second_class(result):
    ranked = sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and ranked[1][1] >= 0.20:
        return ranked[1]
    return None


def _warnings_text(result):
    warns = result.get("warnings") or []
    if not warns:
        return ""
    return "Warnings about the input image: " + " ".join(warns)


def explain(result, lang="en"):
    """Plain-language summary of one predict() result."""
    cls = result["class"]
    parts = [f"The model classified this image as: {_name(cls)}.", CLASS_INFO[cls], _conf_text(result)]
    second = _second_class(result)
    if second:
        parts.append(f"The second most likely class is {_name(second[0])} ({second[1] * 100:.1f}%).")
    w = _warnings_text(result)
    if w:
        parts.append(w)
    parts.append(DISCLAIMER)
    return "\n".join(parts)


def _template_answer(result, question):
    q = question.lower()
    mentioned = None
    for c in CLASSES:
        if c.lower() in q or c.lower().replace("_", " ") in q or DISPLAY[c].lower() in q:
            mentioned = c
            break

    if re.search(r"(warning|alert)", q):
        return _warnings_text(result) or "There are no warnings for this image."
    if re.search(r"(heat ?map|grad|colou?r map|highlight)", q):
        return GRADCAM
    if re.search(r"(sure|confiden|trust|reliab|certain|accura)", q):
        return _conf_text(result)
    if re.search(r"(limit|wrong|error|mistake|fail|valid)", q):
        return LIMITATIONS
    if re.search(r"(why|reason|because|explain)", q) and not mentioned:
        return explain(result) + "\n" + GRADCAM
    if mentioned or re.search(r"(what is|what does|mean|meaning)", q):
        c = mentioned or result["class"]
        return f"{_name(c)}: {CLASS_INFO[c]}"
    return HELP + "\n\n" + explain(result)


def _fmt_p(x):
    """Do not show 1.0 or 0.0 for probabilities that are only close to them."""
    x = float(x)
    if x >= 0.9995:
        return ">0.999"
    if x < 0.0005:
        return "<0.001"
    return round(x, 3)


def build_context(result):
    """The ONLY case data given to a language model (no image, no file name, no patient data)."""
    ctx = {
        "predicted_class": result["class"],
        "confidence": _fmt_p(result["confidence"]),
        "probabilities": {k: _fmt_p(v) for k, v in result["probabilities"].items()},
        "warnings": list(result.get("warnings") or []),
        "model": result.get("model_name", "unknown"),
    }
    return json.dumps(ctx, indent=2)


SYSTEM_PROMPT = """You are an explanation assistant inside a radiology worklist tool. A deep learning model
classified one chest X-ray into: COVID, Lung_Opacity, Normal or Viral Pneumonia. You only explain this
model output to a radiology staff member.

Rules:
1. Use only the CASE DATA and the FACTS below. If the answer is not there, say you do not know.
2. Never diagnose, never give treatment, medication, or patient advice. If asked, refuse and suggest
   discussing with a physician.
3. You cannot see the image. Never describe what the X-ray looks like.
4. Always say the tool assists radiologists and the final decision is theirs.
5. High confidence does not mean the label is correct. Never say the model is "certain" or "sure"; say "very high confidence".
6. Answer in English, in at most 6 short sentences.
7. Never answer yes or no about whether a patient is sick or healthy. Say that the model output is only a
   suggestion for the radiologist.
8. The model cannot explain its own reasoning. For "why" questions, say the exact reason cannot be known,
   say what the predicted class means, and point to the heat map.
9. Do not list possible causes, diseases or conditions. Do not say what the X-ray shows or what the model
   "detected"; the model only assigns a class to the whole image.

FACTS:
- Lung_Opacity means lung areas that look denser (whiter) than normal; it has several possible causes.
- The heat map (Grad-CAM) is low resolution and approximate; it shows where the model looked, not proof of its reasoning.
- The model trained on public datasets only and is not clinically validated. Test accuracy is about 96%
  on the original test set, but it can be wrong with very high confidence.
- On new children's X-rays from another source the model rarely missed pneumonia but often labeled
  healthy images as abnormal (low specificity).
- The model always outputs one of four classes, even for an image that is not a chest X-ray.
- Confidence values like ">0.999" mean extremely close to 1, but the label can still be wrong.
"""


def _http_json(url, payload, headers=None, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ask_ollama(result, question):
    model = os.environ.get("OLLAMA_MODEL")
    if not model:
        raise RuntimeError("OLLAMA_MODEL is not set")
    out = _http_json("http://localhost:11434/api/chat", {
        "model": model, "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CASE DATA:\n{build_context(result)}\n\nQUESTION: {question}"},
        ],
    })
    return out["message"]["content"].strip()


def _ask_gemini(result, question):
    key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL")
    if not key or not model:
        raise RuntimeError("GEMINI_API_KEY and GEMINI_MODEL must be set")
    out = _http_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [
                {"text": f"CASE DATA:\n{build_context(result)}\n\nQUESTION: {question}"}]}],
        },
        headers={"x-goog-api-key": key},
    )
    return out["candidates"][0]["content"]["parts"][0]["text"].strip()


def ask(result, question, lang="en", backend="template"):
    """Answer a question about one case.

    Returns {"answer": str, "source": "template" | "ollama" | "gemini" | "refused"}.
    Diagnosis and treatment questions are refused here, before any language model is called.
    `lang` is ignored (English only).
    """
    if REFUSE_RE.search(question):
        return {"answer": REFUSAL + "\n" + DISCLAIMER, "source": "refused"}
    if backend in ("ollama", "gemini"):
        try:
            text = _ask_ollama(result, question) if backend == "ollama" else _ask_gemini(result, question)
            if text and not UNSUPPORTED_RE.search(text):
                return {"answer": text + "\n\n" + DISCLAIMER, "source": backend}
        except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError, RuntimeError):
            pass  # fall back to the template layer
    return {"answer": _template_answer(result, question), "source": "template"}


if __name__ == "__main__":
    demo = {
        "class": "Lung_Opacity", "confidence": 0.64,
        "probabilities": {"COVID": 0.28, "Lung_Opacity": 0.64, "Normal": 0.05, "Viral Pneumonia": 0.03},
        "heatmap": None, "model_name": "efficientnet_b0",
        "warnings": ["Image is colored. Chest X-rays are normally grayscale; this may not be a chest X-ray."],
    }
    print(explain(demo), "\n")
    for q in ["Why this result?", "How sure is the model?", "What does Lung Opacity mean?",
              "What is the warning?", "Should I take antibiotics?", "What does the heat map show?"]:
        r = ask(demo, q)
        print(f"Q: {q}\n[{r['source']}] {r['answer']}\n")