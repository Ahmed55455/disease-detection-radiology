"""Explanation assistant for predict() results.

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
    print(explain(result, lang="en"))
    print(ask(result, "How sure is the model?", lang="en")["answer"])

Environment variables for the optional layer:
    OLLAMA_MODEL   name of a model you pulled with Ollama (for example "llama3.2")
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

DISPLAY = {
    "en": {"COVID": "COVID-19", "Lung_Opacity": "Lung Opacity", "Normal": "Normal",
           "Viral Pneumonia": "Viral Pneumonia"},
    "ar": {"COVID": "كوفيد-19", "Lung_Opacity": "عتامة رئوية", "Normal": "طبيعي",
           "Viral Pneumonia": "التهاب رئوي فيروسي"},
}

CLASS_INFO = {
    "en": {
        "COVID": "The image resembles chest X-rays labeled COVID-19 in the training data.",
        "Lung_Opacity": "Lung opacity means areas of the lungs that look denser (whiter) than normal "
                        "on the X-ray. It has several possible causes, so this label does not name one.",
        "Normal": "The image resembles chest X-rays labeled normal in the training data.",
        "Viral Pneumonia": "The image resembles chest X-rays labeled viral pneumonia in the training data.",
    },
    "ar": {
        "COVID": "تشبه الصورة صور أشعة الصدر المصنّفة كوفيد-19 في بيانات التدريب.",
        "Lung_Opacity": "العتامة الرئوية تعني مناطق في الرئتين تبدو أكثر كثافة (أكثر بياضاً) من الطبيعي في الأشعة. "
                        "لها أسباب عدة، فهذا التصنيف لا يحدد سبباً معيناً.",
        "Normal": "تشبه الصورة صور أشعة الصدر المصنّفة طبيعية في بيانات التدريب.",
        "Viral Pneumonia": "تشبه الصورة صور أشعة الصدر المصنّفة التهاباً رئوياً فيروسياً في بيانات التدريب.",
    },
}

DISCLAIMER = {
    "en": "This tool assists radiologists and does not replace them. It is not a diagnosis. "
          "The final decision belongs to the radiologist.",
    "ar": "هذه الأداة تساعد أطباء الأشعة ولا تحل محلهم، وهي ليست تشخيصاً. القرار النهائي للطبيب.",
}

# Facts the assistant may use (measured in this project, see docs/model_findings.md).
LIMITATIONS = {
    "en": (
        "Known limits of this model: it was trained on public datasets only and has not been clinically "
        "validated. On the original test set it reaches about 96% accuracy, but it can be wrong with very "
        "high confidence. On new children's X-rays from another source, it rarely missed pneumonia but "
        "often labeled healthy images as abnormal. It always outputs one of four classes, even for an image "
        "that is not a chest X-ray. The heat map is approximate."
    ),
    "ar": (
        "حدود هذا النموذج: دُرّب على بيانات عامة فقط ولم يُختبر سريرياً. على مجموعة الاختبار الأصلية "
        "دقته حوالي 96%، لكنه قد يخطئ بثقة عالية جداً. وعلى صور أشعة أطفال جديدة من مصدر آخر نادراً ما "
        "فاته الالتهاب الرئوي، لكنه كثيراً ما صنّف صوراً سليمة على أنها غير طبيعية. وهو يعطي دائماً "
        "واحداً من أربعة أصناف حتى لو لم تكن الصورة أشعة صدر. والخريطة الحرارية تقريبية."
    ),
}

GRADCAM = {
    "en": ("The colored heat map (Grad-CAM) shows the image regions that influenced the prediction most. "
           "Red/yellow means stronger influence. It is low resolution and approximate: it shows where the "
           "model looked, not proof of why it decided, and it may highlight areas outside the lungs."),
    "ar": ("الخريطة الحرارية الملونة (Grad-CAM) تُظهر المناطق التي أثّرت أكثر في التنبؤ. اللون الأحمر أو "
           "الأصفر يعني تأثيراً أقوى. دقتها منخفضة وتقريبية: تبيّن أين نظر النموذج، لا أنها دليل على سبب "
           "قراره، وقد تبرز مناطق خارج الرئتين."),
}

REFUSAL = {
    "en": ("I can't diagnose or advise on treatment or medication. I can only explain what the model "
           "predicted for this image. Please discuss the case with a physician."),
    "ar": ("لا أستطيع التشخيص ولا تقديم نصيحة عن العلاج أو الدواء. أستطيع فقط شرح ما تنبأ به النموذج لهذه "
           "الصورة. يُرجى مناقشة الحالة مع الطبيب."),
}

HELP = {
    "en": ("I can explain this result. You can ask about: why this class, how sure the model is, "
           "what a class means, the heat map, the warnings, or the model's limits."),
    "ar": ("أستطيع شرح هذه النتيجة. يمكنك السؤال عن: سبب هذا التصنيف، ومدى ثقة النموذج، ومعنى أي صنف، "
           "والخريطة الحرارية، والتحذيرات، وحدود النموذج."),
}

# Questions that ask for a diagnosis or treatment are refused in code, before any model is called.
REFUSE_RE = re.compile(
    r"(do i have|am i (sick|infected|ill)|should i (take|start|stop)|treat|medicat|medicine|drug|dose|dosage|"
    r"prescri|cure|therapy|surgery|antibiotic|"
    r"علاج|دواء|ادوية|أدوية|جرعة|عملية جراحية|هل أنا مصاب|هل عندي|هل لدي|وصفة)",
    re.IGNORECASE,
)


def _name(cls, lang):
    return DISPLAY[lang].get(cls, cls)


def confidence_level(conf):
    if conf >= 0.90:
        return "high"
    if conf >= 0.70:
        return "moderate"
    return "low"


def _conf_text(result, lang):
    conf = result["confidence"]
    lvl = confidence_level(conf)
    pct = f"{conf * 100:.1f}%"
    if lang == "ar":
        lvl_ar = {"high": "عالية", "moderate": "متوسطة", "low": "منخفضة"}[lvl]
        base = f"ثقة النموذج في هذا التصنيف {lvl_ar} ({pct})."
        extra = {
            "high": " لكن الثقة العالية لا تعني أن التصنيف صحيح؛ فقد أخطأ النموذج في اختباراتنا بثقة عالية جداً.",
            "moderate": " هذا يعني أن النموذج غير متأكد تماماً، فالمراجعة البشرية مهمة بشكل خاص.",
            "low": " هذا يعني أن النموذج متردد، ويجب عدم الاعتماد على هذه النتيجة وحدها.",
        }[lvl]
        return base + extra
    base = f"The model's confidence in this class is {lvl} ({pct})."
    extra = {
        "high": " High confidence does not mean the label is correct: in our tests the model was wrong "
                "with very high confidence.",
        "moderate": " The model is not fully sure, so human review matters even more.",
        "low": " The model is undecided. This result should not be relied on alone.",
    }[lvl]
    return base + extra


def _second_class(result):
    probs = result["probabilities"]
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and ranked[1][1] >= 0.20:
        return ranked[1]
    return None


def _warnings_text(result, lang):
    warns = result.get("warnings") or []
    if not warns:
        return ""
    head = "تنبيهات على الصورة:" if lang == "ar" else "Warnings about the input image:"
    return head + " " + " ".join(warns)


def explain(result, lang="en"):
    """Plain-language summary of one predict() result."""
    lang = "ar" if lang == "ar" else "en"
    cls = result["class"]
    parts = []
    if lang == "ar":
        parts.append(f"صنّف النموذج هذه الصورة على أنها: {_name(cls, lang)}.")
    else:
        parts.append(f"The model classified this image as: {_name(cls, lang)}.")
    parts.append(CLASS_INFO[lang][cls])
    parts.append(_conf_text(result, lang))
    second = _second_class(result)
    if second:
        s_name, s_p = second
        if lang == "ar":
            parts.append(f"الصنف الثاني الأقرب هو {_name(s_name, lang)} باحتمال {s_p * 100:.1f}%.")
        else:
            parts.append(f"The second most likely class is {_name(s_name, lang)} ({s_p * 100:.1f}%).")
    w = _warnings_text(result, lang)
    if w:
        parts.append(w)
    parts.append(DISCLAIMER[lang])
    return "\n".join(parts)


def _template_answer(result, question, lang):
    q = question.lower()
    mentioned = None
    for c in CLASSES:
        names = {c.lower(), c.lower().replace("_", " "), DISPLAY["en"][c].lower(), DISPLAY["ar"][c]}
        if any(n in q for n in names):
            mentioned = c
            break

    if re.search(r"(warning|alert|تحذير|تنبيه)", q):
        w = _warnings_text(result, lang)
        if w:
            return w
        return "لا توجد تحذيرات لهذه الصورة." if lang == "ar" else "There are no warnings for this image."
    if re.search(r"(heat ?map|grad|colou?r map|highlight|الخريطة|حرارية|الألوان)", q):
        return GRADCAM[lang]
    if re.search(r"(sure|confiden|trust|reliab|certain|accura|ثقة|متأكد|موثوق|دقة|يوثق)", q):
        return _conf_text(result, lang)
    if re.search(r"(limit|wrong|error|mistake|fail|valid|حدود|خطأ|يخطئ|قيود|يفشل)", q):
        return LIMITATIONS[lang]
    if re.search(r"(why|reason|because|explain|لماذا|ليش|سبب|اشرح|فسر)", q) and not mentioned:
        base = explain(result, lang)
        return base + "\n" + GRADCAM[lang]
    if mentioned or re.search(r"(what is|what does|mean|meaning|ما هو|ما معنى|معنى|يعني)", q):
        c = mentioned or result["class"]
        return f"{_name(c, lang)}: {CLASS_INFO[lang][c]}"
    return HELP[lang] + "\n\n" + explain(result, lang)


def build_context(result):
    """The ONLY case data given to a language model (no image, no file name, no patient data)."""
    ctx = {
        "predicted_class": result["class"],
        "confidence": round(float(result["confidence"]), 3),
        "probabilities": {k: round(float(v), 3) for k, v in result["probabilities"].items()},
        "warnings": list(result.get("warnings") or []),
        "model": result.get("model_name", "unknown"),
    }
    return json.dumps(ctx, ensure_ascii=False, indent=2)


SYSTEM_PROMPT = """You are an explanation assistant inside a radiology worklist tool. A deep learning model
classified one chest X-ray into: COVID, Lung_Opacity, Normal or Viral Pneumonia. You only explain this
model output to a radiology staff member.

Rules:
1. Use only the CASE DATA and the FACTS below. If the answer is not there, say you do not know.
2. Never diagnose, never give treatment, medication, or patient advice. If asked, refuse and suggest
   discussing with a physician.
3. You cannot see the image. Never describe what the X-ray looks like.
4. Always say the tool assists radiologists and the final decision is theirs.
5. High confidence does not mean the label is correct. Never say the result is certain.
6. Answer in the language of the question (Arabic or English), in at most 6 short sentences.

FACTS:
- Lung_Opacity means lung areas that look denser (whiter) than normal; it has several possible causes.
- The heat map (Grad-CAM) is low resolution and approximate; it shows where the model looked, not proof of its reasoning.
- The model trained on public datasets only and is not clinically validated. Test accuracy is about 96%
  on the original test set, but it can be wrong with very high confidence.
- On new children's X-rays from another source the model rarely missed pneumonia but often labeled
  healthy images as abnormal (low specificity).
- The model always outputs one of four classes, even for an image that is not a chest X-ray.
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
    """
    lang = "ar" if lang == "ar" else "en"
    if REFUSE_RE.search(question):
        return {"answer": REFUSAL[lang] + "\n" + DISCLAIMER[lang], "source": "refused"}
    if backend in ("ollama", "gemini"):
        try:
            text = _ask_ollama(result, question) if backend == "ollama" else _ask_gemini(result, question)
            if text:
                return {"answer": text + "\n\n" + DISCLAIMER[lang], "source": backend}
        except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError, RuntimeError):
            pass  # fall back to the template layer
    return {"answer": _template_answer(result, question, lang), "source": "template"}


if __name__ == "__main__":
    demo = {
        "class": "Lung_Opacity", "confidence": 0.64,
        "probabilities": {"COVID": 0.28, "Lung_Opacity": 0.64, "Normal": 0.05, "Viral Pneumonia": 0.03},
        "heatmap": None, "model_name": "efficientnet_b0",
        "warnings": ["Image is colored. Chest X-rays are normally grayscale; this may not be a chest X-ray."],
    }
    print(explain(demo, "en"), "\n")
    for q in ["Why this result?", "How sure is the model?", "What does Lung Opacity mean?",
              "What is the warning?", "Should I take antibiotics?", "ما معنى الخريطة الحرارية؟"]:
        lang = "ar" if re.search(r"[؀-ۿ]", q) else "en"
        r = ask(demo, q, lang)
        print(f"Q: {q}\n[{r['source']}] {r['answer']}\n")