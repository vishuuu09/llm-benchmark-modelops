import os
import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="LLM Benchmarking & ModelOps",
    layout="wide"
)

st.title("LLM Benchmarking & ModelOps")
st.caption(
    "Compare LLM performance using latency and lexical evaluation proxies."
)

api = st.sidebar.text_input(
    "API URL",
    os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
)

try:
    health = requests.get(api + "/health", timeout=3).json()
    st.sidebar.success("API: " + health["status"])
except Exception:
    st.sidebar.error("Start API first: uvicorn backend.main:app --reload")
    st.stop()

if st.button("Run bundled benchmark", type="primary"):
    try:
        with st.spinner("Evaluating examples with the selected model..."):
            response = requests.post(
                api + "/benchmark",
                timeout=600
            )

        if response.ok:
            st.session_state.result = response.json()
        else:
            st.error(response.text)

    except requests.RequestException as exc:
        st.error(f"Benchmark request failed: {exc}")

res = st.session_state.get("result")

if res:
    st.divider()

    st.subheader("Benchmark overview")

    st.write(
        f"**Model:** {res['model']}  |  "
        f"**Provider:** {res['provider']}  |  "
        f"**Run ID:** {res['run_id']}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Test cases", res["total_items"])
    c2.metric("Mean latency", f"{res['average_latency']:.2f} ms")
    c3.metric(
        "Reference overlap proxy",
        f"{res['average_reference_overlap']:.3f}"
    )

    c4, c5 = st.columns(2)

    c4.metric(
        "Reference F1 proxy",
        f"{res['average_reference_f1']:.3f}"
    )

    c5.metric(
        "Keyword relevance proxy",
        f"{res['average_keyword_relevance']:.3f}"
    )

    st.divider()
    st.subheader("Quality gate")

    if res["passed_gate"]:
        st.success("PASS — All configured checks passed.")
    else:
        st.error("FAIL — One or more configured checks failed.")

    checks = res["gate_checks"]

    gate_df = pd.DataFrame([
        {"Check": "Reference F1 proxy", "Result": "PASS" if checks["reference_f1"] else "FAIL"},
        {"Check": "Keyword relevance proxy", "Result": "PASS" if checks["keyword_relevance"] else "FAIL"},
        {"Check": "Average latency", "Result": "PASS" if checks["average_latency"] else "FAIL"}
    ])

    st.dataframe(gate_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Individual test results")

    df = pd.DataFrame(res["results"])

    display_columns = [
        "id",
        "question",
        "answer",
        "latency_ms",
        "keyword_relevance",
        "context_overlap_proxy",
        "reference_overlap_proxy",
        "reference_precision_proxy",
        "reference_f1_proxy",
        "answer_token_count"
    ]

    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Latency by test case")
    st.bar_chart(df.set_index("id")["latency_ms"])

    st.subheader("Metric comparison")
    metric_columns = [
        "keyword_relevance",
        "reference_overlap_proxy",
        "reference_f1_proxy"
    ]

    st.bar_chart(df.set_index("id")[metric_columns])

    st.caption(
        "All lexical scores are proxies. They do not independently establish "
        "factual accuracy, semantic correctness, or hallucination."
    )

st.divider()
st.subheader("Previous benchmark runs")

try:
    history_response = requests.get(api + "/runs", timeout=5)
    history_response.raise_for_status()
    history = history_response.json()

    if history:
        st.dataframe(
            pd.DataFrame(history),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No saved runs yet.")

except Exception as exc:
    st.warning(str(exc))
