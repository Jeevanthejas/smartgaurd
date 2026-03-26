import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import time
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="SmartGuard", layout="wide")
st.title("SmartGuard — LLM Prompt Firewall")

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./model")
    tokenizer = AutoTokenizer.from_pretrained("./model")
    with open("label2id.json") as f:
        mapping = json.load(f)
    id2label_int = {int(k): v for k, v in mapping["id2label"].items()}
    model.eval()
    return model, tokenizer, id2label_int

try:
    model, tokenizer, id2label = load_model()
except Exception as e:
    st.error(f"Error loading model. Has it been trained yet? ({e})")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Live tester", "Aggregate metrics", "Threshold curve"])

with tab1:
    st.header("Component A: Live Tester")
    user_input = st.text_area("Enter a prompt")
    threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05)

    if st.button("Submit"):
        if not user_input.strip():
            st.warning("Please enter a prompt.")
        else:
            inputs = tokenizer(
                user_input,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(**inputs).logits
            latency_ms = (time.perf_counter() - t0) * 1000

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_id = probs.argmax()
            pred_class = id2label[pred_id]
            confidence = float(probs[pred_id])

            is_blocked = (confidence > threshold) and (pred_class != 'safe')
            verdict = "UNSAFE(BLOCKED)" if is_blocked else "SAFE(ALLOWED)"
            color = "red" if is_blocked else "green"

            st.markdown(f"### Verdict: <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
            st.write(f"**Category:** {pred_class}")
            st.write(f"**Latency:** {latency_ms:.2f} ms")

            st.write("**Confidence:**")
            st.progress(confidence)
            st.write(f"{confidence:.2%}")

            st.divider()
            st.header("Component B: Threshold Impact Demo")
            st.write("If we applied different thresholds to this exact same prompt:")
            cols = st.columns(3)
            for i, t in enumerate([0.3, 0.5, 0.7]):
                with cols[i]:
                    dem_blocked = (confidence > t) and (pred_class != 'safe')
                    dem_verd = "BLOCKED" if dem_blocked else "ALLOWED"
                    dem_color = "red" if dem_blocked else "green"
                    st.markdown(f"**Threshold {t}:** <span style='color:{dem_color}'>{dem_verd}</span>", unsafe_allow_html=True)

with tab2:
    st.header("Component C: Aggregate Metrics Panel")
    try:
        rt_df = pd.read_csv("results/red_team_results.csv")
        total = len(rt_df)

        harmful_mask = rt_df['true_label'] != 'safe'
        benign_mask = rt_df['true_label'] == 'safe'

        harmful_df = rt_df[harmful_mask]
        benign_df = rt_df[benign_mask]

        blocked_harmful = harmful_df[harmful_df['predicted_label'] != 'safe']
        missed_harmful = harmful_df[harmful_df['predicted_label'] == 'safe']
        fp_benign = benign_df[benign_df['predicted_label'] != 'safe']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Evaluated", total)

        if len(harmful_df) > 0:
            block_rate = len(blocked_harmful) / len(harmful_df)
            col2.metric("Harmful Blocked", f"{len(blocked_harmful)}", f"{block_rate:.1%}")
            miss_rate = len(missed_harmful) / len(harmful_df)
            col3.metric("Harmful Missed", f"{len(missed_harmful)}", f"{-miss_rate:.1%}", delta_color="inverse")

        if len(benign_df) > 0:
            fp_rate = len(fp_benign) / len(benign_df)
            col4.metric("Benign False Positives", f"{len(fp_benign)}", f"{-fp_rate:.1%}", delta_color="inverse")

        st.subheader("Per-Class Recall (Harmful)")
        recall_data = []
        for c in ['jailbreak', 'injection', 'toxic', 'pii']:
            class_df = rt_df[rt_df['true_label'] == c]
            if len(class_df) > 0:
                blocked = len(class_df[class_df['predicted_label'] != 'safe'])
                recall_data.append({"Class": c, "Total": len(class_df), "Blocked": blocked, "Recall": f"{blocked/len(class_df):.1%}"})
        if recall_data:
            st.table(pd.DataFrame(recall_data))
    except Exception as e:
        st.info(f"Red team results not available yet. Please run eval.py first. ({e})")

with tab3:
    st.header("Component D: Accuracy vs Strictness Curve")
    try:
        with open("results/threshold_curve.json") as f:
            t_data = json.load(f)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_data["thresholds"], y=t_data["recall"], name="Recall (harmful)", mode='lines+markers'))
        fig.add_trace(go.Scatter(x=t_data["thresholds"], y=t_data["fpr"], name="False positive rate", mode='lines+markers'))
        fig.add_vline(x=0.5, line_dash="dash", annotation_text="Deployed threshold")
        fig.update_layout(xaxis_title="Threshold", yaxis_title="Rate", yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Threshold curve data not available yet. Please run eval.py first. ({e})")
