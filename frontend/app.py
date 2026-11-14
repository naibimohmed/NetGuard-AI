
import requests
import pandas as pd
import streamlit as st

from pathlib import Path


st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.ok
    except requests.RequestException:
        return False


def predict_traffic(data, model):
    response = requests.post(
        f"{API_URL}/predict-batch",
        json={
            "model": model,
            "records": data.to_dict(orient="records")
        },
        timeout=120
    )

    if not response.ok:
        try:
            message = response.json().get("detail", response.text)
        except ValueError:
            message = response.text

        raise ValueError(message)

    return response.json()


def show_metrics(results):
    total = len(results)
    attacks = int((results["prediction"] == 1).sum())
    normal = total - attacks
    rate = attacks / total * 100 if total else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Traffic", total)
    col2.metric("Normal Traffic", normal)
    col3.metric("Detected Attacks", attacks)
    col4.metric("Detection Rate", f"{rate:.2f}%")


def show_results():
    results = st.session_state["last_results"]

    st.divider()
    st.markdown("### Detection Results")

    show_metrics(results)

    st.markdown("#### Traffic Distribution")

    distribution = pd.DataFrame({
        "Traffic": ["Normal", "Attack"],
        "Count": [
            int((results["prediction"] == 0).sum()),
            int((results["prediction"] == 1).sum())
        ]
    }).set_index("Traffic")

    st.bar_chart(distribution)

    st.markdown("#### Prediction Details")

    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download Results",
        data=results.to_csv(index=False),
        file_name="netguard_results.csv",
        mime="text/csv"
    )


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b1120;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263247;
}

.brand {
    font-size: 27px;
    font-weight: 800;
    color: #60a5fa;
    margin-bottom: 0;
}

.muted {
    color: #94a3b8;
    font-size: 13px;
}

.metric-card {
    background: #151f31;
    border: 1px solid #263247;
    border-radius: 12px;
    padding: 20px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
}

.metric-value {
    color: #f8fafc;
    font-size: 29px;
    font-weight: 700;
    margin-top: 8px;
}

div[data-testid="stFileUploader"] {
    background: #151f31;
    border: 1px dashed #3b82f6;
    border-radius: 12px;
    padding: 15px;
}

div.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    min-height: 42px;
}

div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown(
        '<p class="brand">🛡️ NetGuard AI</p>',
        unsafe_allow_html=True
    )

    st.caption("NETWORK SECURITY PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Traffic Analysis",
            "Detections",
            "Model Performance"
        ]
    )

    st.divider()

    model_name = st.selectbox(
        "Detection Model",
        ["rf", "xgboost"],
        format_func=lambda model: {
            "rf": "Random Forest",
            "xgboost": "XGBoost"
        }[model]
    )

    st.divider()

    api_status = check_api()

    if api_status:
        st.success("API Connected")
    else:
        st.error("API Disconnected")

    st.caption("NetGuard AI v1.0")


st.markdown(
    '<p class="brand">Security Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="muted">Monitor and analyze network traffic using machine learning.</p>',
    unsafe_allow_html=True
)

st.divider()


if page == "Overview":

    st.markdown("### Overview")

    if "last_results" in st.session_state:
        results = st.session_state["last_results"]

        total = len(results)
        attacks = int((results["prediction"] == 1).sum())
        normal = total - attacks
        rate = attacks / total * 100 if total else 0

        cards = [
            ("Total Traffic", total),
            ("Normal Traffic", normal),
            ("Detected Attacks", attacks),
            ("Detection Rate", f"{rate:.2f}%")
        ]
    else:
        cards = [
            ("Total Traffic", "—"),
            ("Normal Traffic", "—"),
            ("Detected Attacks", "—"),
            ("Detection Rate", "—")
        ]

    columns = st.columns(4)

    for column, (label, value) in zip(columns, cards):
        with column:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    left, right = st.columns([2, 1])

    with left:
        st.markdown("### Traffic Monitoring")

        if "last_results" in st.session_state:
            results = st.session_state["last_results"]

            distribution = pd.DataFrame({
                "Traffic": ["Normal", "Attack"],
                "Count": [
                    int((results["prediction"] == 0).sum()),
                    int((results["prediction"] == 1).sum())
                ]
            }).set_index("Traffic")

            st.bar_chart(distribution)
        else:
            st.info(
                "Upload a traffic CSV from the Traffic Analysis page "
                "to start monitoring."
            )

    with right:
        st.markdown("### Active Model")
        st.markdown(f"**{model_name.upper()}**")
        st.caption("Autoencoder feature extraction enabled")


elif page == "Traffic Analysis":

    st.markdown("### Traffic Analysis")

    st.write(
        "Upload a CSV file containing network traffic features."
    )

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            if data.empty:
                st.warning("The uploaded file is empty.")
            else:
                st.markdown("#### Dataset Preview")

                col1, col2 = st.columns(2)
                col1.metric("Rows", len(data))
                col2.metric("Columns", len(data.columns))

                st.dataframe(
                    data.head(10),
                    use_container_width=True
                )

                if st.button("Analyze Traffic", use_container_width=True):

                    if not api_status:
                        st.error("FastAPI is not running.")
                    else:
                        try:
                            with st.spinner("Analyzing network traffic..."):
                                prediction = predict_traffic(
                                    data,
                                    model_name
                                )

                            results = pd.DataFrame(
                                prediction["results"]
                            )

                            st.session_state["last_results"] = results
                            st.session_state["last_model"] = model_name

                            st.success(
                                f"Analysis completed: {prediction['total']} records."
                            )

                        except (requests.RequestException, ValueError) as error:
                            st.error(f"Analysis failed: {error}")

                        except Exception as error:
                            st.error(f"Unexpected error: {error}")

        except pd.errors.ParserError:
            st.error("Could not read the CSV file. Check its format.")

        except Exception as error:
            st.error(f"File error: {error}")

    if "last_results" in st.session_state:
        show_results()


elif page == "Detections":

    st.markdown("### Threat Detections")

    if "last_results" in st.session_state:
        results = st.session_state["last_results"]

        attacks = results[results["prediction"] == 1]

        if attacks.empty:
            st.success("No attacks detected in the last analysis.")
        else:
            st.error(f"{len(attacks)} potential attacks detected.")

            st.dataframe(
                attacks,
                use_container_width=True
            )
    else:
        st.info("Run a traffic analysis to display detected attacks.")


elif page == "Model Performance":

    st.markdown("### Model Performance")

    try:
        rf = pd.read_json(
            RESULTS_DIR / "ae_rf_metrics.json",
            typ="series"
        )

        xgb = pd.read_json(
            RESULTS_DIR / "ae_xgb_metrics.json",
            typ="series"
        )

        comparison = pd.DataFrame({
            "Random Forest": rf,
            "XGBoost": xgb
        })

        metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ]

        available_metrics = [
            metric for metric in metrics
            if metric in comparison.index
        ]

        comparison = comparison.loc[available_metrics]

        st.markdown("#### Performance Comparison")

        st.dataframe(
            comparison,
            use_container_width=True
        )

        st.bar_chart(comparison)

    except FileNotFoundError:
        st.warning("Train both models to display their performance.")

    except Exception as error:
        st.error(f"Could not load model metrics: {error}")
