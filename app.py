import json
import pickle
import re
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
DEPLOY = ROOT / "deploy"

st.set_page_config(
    page_title="Kinyarwanda News Classifier",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .result-card {
            background: rgba(62, 227, 122, 0.08);
            border: 1px solid rgba(62, 227, 122, 0.45);
            border-radius: 10px;
            padding: 20px 24px;
            margin: 8px 0 20px 0;
        }
        .result-label {
            color: #AAB1BA;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
        }
        .result-class {
            color: #3EE37A;
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 2px;
            text-transform: capitalize;
        }
        .result-confidence {
            color: #DADEE5;
            font-size: 1em;
        }
        .bar-row {
            display: flex;
            align-items: center;
            margin: 6px 0;
            gap: 10px;
        }
        .bar-name {
            width: 120px;
            text-align: right;
            color: #DADEE5;
            font-size: 0.95em;
            text-transform: capitalize;
        }
        .bar-track {
            flex: 1;
            height: 10px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 5px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            background: #3EE37A;
            border-radius: 5px;
        }
        .bar-pct {
            width: 60px;
            text-align: right;
            color: #AAB1BA;
            font-variant-numeric: tabular-nums;
            font-size: 0.9em;
        }
        .meta-line {
            color: #7A818A;
            font-size: 0.88em;
            margin-top: -8px;
            margin-bottom: 24px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle():
    model = pickle.load(open(DEPLOY / "model.pkl", "rb"))
    vectorizer = pickle.load(open(DEPLOY / "vectorizer.pkl", "rb"))
    labels = {int(k): v for k, v in json.load(open(DEPLOY / "label_map.json")).items()}
    config = json.load(open(DEPLOY / "config.json"))
    stops = {w.strip().lower() for w in open(DEPLOY / "stopwords.txt") if w.strip()}
    return model, vectorizer, labels, config, stops


model, vectorizer, labels, config, stops = load_bundle()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
NONA_RE = re.compile(r"[^a-z\s]")
MS_RE = re.compile(r"\s+")


def preprocess(text):
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = NONA_RE.sub(" ", t)
    t = MS_RE.sub(" ", t).strip()
    return " ".join(w for w in t.split() if w not in stops and len(w) >= 2)


st.title("Kinyarwanda News Classifier")
st.caption(
    "Enter a Kinyarwanda news article title and content. The model predicts one of 14 "
    "news categories and shows its top three guesses."
)
st.markdown(
    f"<div class='meta-line'>Model: {config['model_name']}  ·  "
    f"71.1% accuracy on the KINNEWS test set</div>",
    unsafe_allow_html=True,
)

title = st.text_input("Title", placeholder="Umukino wa FERWAFA ...")
content = st.text_area(
    "Content",
    height=220,
    placeholder="Full article content in Kinyarwanda...",
)
go = st.button("Predict", type="primary", use_container_width=True)

if go:
    text = f"{title} {content}".strip()
    if not text:
        st.warning("Enter the title and content first.")
    else:
        cleaned = preprocess(text)
        if not cleaned:
            st.error(
                "Nothing left after preprocessing. The input may be too short or made only "
                "of stopwords."
            )
        else:
            X = vectorizer.transform([cleaned])
            probs = model.predict_proba(X)[0]
            order = probs.argsort()[::-1]
            top_idx = order[:3]
            predicted = labels[int(top_idx[0])]
            confidence = probs[int(top_idx[0])]

            st.markdown(
                f"""
                <div class='result-card'>
                    <div class='result-label'>Predicted category</div>
                    <div class='result-class'>{predicted}</div>
                    <div class='result-confidence'>{confidence:.1%} confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### Top 3 guesses")
            top_max = float(probs[int(top_idx[0])])
            bars_html = ""
            for i in top_idx:
                name = labels[int(i)]
                pct = probs[int(i)]
                width = max(2.0, (pct / top_max) * 100.0)
                bars_html += (
                    f"<div class='bar-row'>"
                    f"<div class='bar-name'>{name}</div>"
                    f"<div class='bar-track'><div class='bar-fill' style='width:{width:.1f}%'></div></div>"
                    f"<div class='bar-pct'>{pct:.1%}</div>"
                    f"</div>"
                )
            st.markdown(bars_html, unsafe_allow_html=True)

            with st.expander("All 14 class probabilities"):
                all_html = ""
                for i in order:
                    name = labels[int(i)]
                    pct = probs[int(i)]
                    width = max(1.0, pct * 100.0)
                    all_html += (
                        f"<div class='bar-row'>"
                        f"<div class='bar-name'>{name}</div>"
                        f"<div class='bar-track'><div class='bar-fill' style='width:{width:.1f}%'></div></div>"
                        f"<div class='bar-pct'>{pct:.1%}</div>"
                        f"</div>"
                    )
                st.markdown(all_html, unsafe_allow_html=True)

            with st.expander("Preprocessed text sent to the model"):
                st.code(cleaned)
