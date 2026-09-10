import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MedAI Heart Disease Decision Support",
    page_icon="🫀",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f9fd;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ================= MAIN HEADER ================= */

.main-title {
    background: linear-gradient(135deg, #075985, #0ea5e9);
    color: white;
    padding: 22px 28px;
    border-radius: 16px;
    font-size: 34px;
    font-weight: 800;
    box-shadow: 0 5px 15px rgba(0,0,0,0.12);
}

.main-subtitle {
    background-color: #e6f4ff;
    color: #315b78;
    padding: 12px 20px;
    border-radius: 0 0 12px 12px;
    font-size: 15px;
    margin-bottom: 28px;
}

/* ================= HEADINGS ================= */

.patient-title {
    color: #075985;
    font-size: 29px;
    font-weight: 800;
    margin-top: 18px;
    margin-bottom: 6px;
}

.patient-description {
    color: #607d94;
    font-size: 15px;
    margin-bottom: 20px;
}

/* ================= SECTION BOX ================= */

.section-box {
    background: linear-gradient(135deg, #e1f3ff, #cceaff);
    border-left: 6px solid #087ea4;
    border-radius: 12px;
    padding: 15px 20px;
    margin-top: 22px;
    margin-bottom: 16px;
}

.section-title {
    color: #075985;
    font-size: 20px;
    font-weight: 750;
}

.section-description {
    color: #52758f;
    font-size: 13px;
    margin-top: 3px;
}

/* ================= FEATURE LABELS ================= */

.feature-name {
    color: #174a7e;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 3px;
}

.feature-description {
    color: #718096;
    font-size: 12px;
    margin-bottom: 7px;
}

/* ================= ANALYZE BUTTON ================= */

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #075985, #0ea5e9);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 13px;
    font-size: 16px;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0369a1, #0284c7);
    color: white;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #075985 0%,
        #086f9d 55%,
        #064e73 100%
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.3);
}

/* ================= REPORT CARDS ================= */

.report-card {
    background: linear-gradient(135deg, #075985, #0ea5e9);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}

.report-label {
    color: #dff6ff;
    font-size: 13px;
    font-weight: 600;
}

.report-value {
    color: white;
    font-size: 28px;
    font-weight: 800;
    margin-top: 5px;
}

/* ================= PREDICTION BOX ================= */

.prediction-box {
    background: white;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    border: 1px solid #d8eaf5;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.prediction-text {
    color: #075985;
    font-size: 25px;
    font-weight: 800;
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #718096;
    font-size: 13px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

try:
    model = joblib.load("knn_heart_disease_model.pkl")
    scaler = joblib.load("heart_disease_scaler.pkl")

except Exception as e:
    st.error("Unable to load model or scaler.")
    st.code(str(e))
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🫀 MedAI Heart Disease Decision Support'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Machine Learning based cardiovascular risk assessment '
    'using the K-Nearest Neighbors algorithm'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🩺 MedAI")

    st.write(
        "Trustworthy AI Medical Decision Support System"
    )

    st.divider()

    st.markdown("### 🤖 Model Information")

    st.write("**Algorithm:** K-Nearest Neighbors")
    st.write("**Test Accuracy:** 88.52%")
    st.write("**Input Features:** 13")
    st.write("**Dataset:** Cleveland Heart Disease")

    st.divider()

    st.markdown("### 📌 About")

    st.write(
        "This system uses machine learning to identify "
        "patterns associated with heart disease."
    )

    st.divider()

    st.warning(
        "Academic prototype only. "
        "Not a substitute for professional medical diagnosis."
    )


# ============================================================
# PATIENT ASSESSMENT
# ============================================================

st.markdown(
    '<div class="patient-title">👤 Patient Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="patient-description">'
    "Enter the patient's clinical information below. "
    "Each feature includes a description to make the "
    "required input easy to understand."
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# BASIC INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-box">
        <div class="section-title">👤 Basic Information</div>
        <div class="section-description">
            Patient demographic information
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        '<div class="feature-name">Age (age)</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Age of the patient in years'
        '</div>',
        unsafe_allow_html=True
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50,
        label_visibility="collapsed"
    )


with col2:

    st.markdown(
        '<div class="feature-name">Sex (sex)</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Biological sex of the patient'
        '</div>',
        unsafe_allow_html=True
    )

    sex_option = st.selectbox(
        "Sex",
        ["Female", "Male"],
        label_visibility="collapsed"
    )

sex = 0 if sex_option == "Female" else 1


# ============================================================
# CLINICAL INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-box">
        <div class="section-title">🫀 Clinical Information</div>
        <div class="section-description">
            Chest pain and exercise-related information
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        '<div class="feature-name">Chest Pain Type (cp)</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Type of chest pain experienced by the patient'
        '</div>',
        unsafe_allow_html=True
    )

    cp_option = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-Anginal Pain",
            "Asymptomatic"
        ],
        label_visibility="collapsed"
    )

cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}

cp = cp_mapping[cp_option]


with col2:

    st.markdown(
        '<div class="feature-name">'
        'Exercise Induced Angina (exang)'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Whether exercise causes chest discomfort'
        '</div>',
        unsafe_allow_html=True
    )

    exang_option = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"],
        label_visibility="collapsed"
    )

exang = 0 if exang_option == "No" else 1


# ============================================================
# CARDIOVASCULAR MEASUREMENTS
# ============================================================

st.markdown(
    """
    <div class="section-box">
        <div class="section-title">
            ❤️ Cardiovascular Measurements
        </div>
        <div class="section-description">
            Blood pressure, cholesterol and heart rate
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        '<div class="feature-name">'
        'Resting Blood Pressure (trestbps)'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Resting blood pressure measured in mmHg'
        '</div>',
        unsafe_allow_html=True
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=220,
        value=120,
        label_visibility="collapsed"
    )


with col2:

    st.markdown(
        '<div class="feature-name">Cholesterol (chol)</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Serum cholesterol level in mg/dL'
        '</div>',
        unsafe_allow_html=True
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=600,
        value=200,
        label_visibility="collapsed"
    )


with col3:

    st.markdown(
        '<div class="feature-name">'
        'Maximum Heart Rate (thalach)'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Maximum heart rate achieved during exercise'
        '</div>',
        unsafe_allow_html=True
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150,
        label_visibility="collapsed"
    )


# ============================================================
# ECG & BLOOD SUGAR
# ============================================================

st.markdown(
    """
    <div class="section-box">
        <div class="section-title">🩺 ECG & Blood Sugar</div>
        <div class="section-description">
            Resting ECG and fasting blood sugar information
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        '<div class="feature-name">Fasting Blood Sugar (fbs)</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Whether fasting blood sugar is greater than 120 mg/dL'
        '</div>',
        unsafe_allow_html=True
    )

    fbs_option = st.selectbox(
        "Fasting Blood Sugar",
        [
            "No — ≤ 120 mg/dL",
            "Yes — > 120 mg/dL"
        ],
        label_visibility="collapsed"
    )

fbs = 0 if fbs_option.startswith("No") else 1


with col2:

    st.markdown(
        '<div class="feature-name">Resting ECG (restecg)</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Resting electrocardiographic result'
        '</div>',
        unsafe_allow_html=True
    )

    restecg_option = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ],
        label_visibility="collapsed"
    )

restecg_mapping = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

restecg = restecg_mapping[restecg_option]


# ============================================================
# EXERCISE TEST
# ============================================================

st.markdown(
    """
    <div class="section-box">
        <div class="section-title">🏃 Exercise Test</div>
        <div class="section-description">
            Exercise-related heart measurements
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        '<div class="feature-name">'
        'ST Depression / Oldpeak (oldpeak)'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'ST depression induced by exercise'
        '</div>',
        unsafe_allow_html=True
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=7.0,
        value=1.0,
        step=0.1,
        label_visibility="collapsed"
    )


with col2:

    st.markdown(
        '<div class="feature-name">ST Segment Slope (slope)</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Slope of the peak exercise ST segment'
        '</div>',
        unsafe_allow_html=True
    )

    slope_option = st.selectbox(
        "Slope",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ],
        label_visibility="collapsed"
    )

slope_mapping = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

slope = slope_mapping[slope_option]


# ============================================================
# ADDITIONAL FEATURES
# ============================================================

st.markdown(
    """
    <div class="section-box">
        <div class="section-title">
            📊 Additional Clinical Features
        </div>
        <div class="section-description">
            Additional diagnostic information
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        '<div class="feature-name">Major Vessels (ca)</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Number of major vessels observed by fluoroscopy'
        '</div>',
        unsafe_allow_html=True
    )

    ca = st.selectbox(
        "Major Vessels",
        [0, 1, 2, 3, 4],
        label_visibility="collapsed"
    )


with col2:

    st.markdown(
        '<div class="feature-name">'
        'Thalassemia Category (thal)'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="feature-description">'
        'Diagnostic category for thalassemia-related findings'
        '</div>',
        unsafe_allow_html=True
    )

    thal_option = st.selectbox(
        "Thalassemia",
        [
            "Unknown / Not specified",
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ],
        label_visibility="collapsed"
    )

thal_mapping = {
    "Unknown / Not specified": 0,
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

thal = thal_mapping[thal_option]


# ============================================================
# ANALYZE REPORT BUTTON
# ============================================================

st.write("")

predict = st.button(
    "🔍 ANALYZE REPORT",
    use_container_width=True
)


# ============================================================
# COMPLETE ANALYSIS REPORT
# ONLY APPEARS AFTER BUTTON CLICK
# ============================================================

if predict:

    # ========================================================
    # CREATE PATIENT DATA
    # ========================================================

    feature_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ]

    patient_data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=feature_names
    )

    # ========================================================
    # SCALE + PREDICT
    # ========================================================

    patient_scaled = scaler.transform(patient_data)

    prediction = model.predict(patient_scaled)[0]

    # ========================================================
    # ANALYSIS REPORT HEADING
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="patient-title" style="text-align:center;">'
        '📋 Analysis Report'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # DISEASE PREDICTION
    # ========================================================

    if prediction == 0:

        st.markdown(
            """
            <div class="prediction-box">
                <div class="prediction-text">
                    ✅ No Heart Disease Detected
                </div>
                <p style="color:#52758f;">
                    The KNN model did not identify patterns
                    associated with heart disease in the
                    provided patient information.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="prediction-box">
                <div class="prediction-text">
                    ⚠️ Heart Disease Detected
                </div>
                <p style="color:#52758f;">
                    The KNN model identified patterns associated
                    with heart disease in the provided patient
                    information.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="patient-title" style="text-align:center;">'
        '🤖 Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="report-card">
                <div class="report-label">
                    TEST ACCURACY
                </div>
                <div class="report-value">
                    88.52%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="report-card">
                <div class="report-label">
                    MODEL USED
                </div>
                <div class="report-value">
                    KNN
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="report-card">
                <div class="report-label">
                    INPUT FEATURES
                </div>
                <div class="report-value">
                    13
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # PATIENT SUMMARY
    # ========================================================

    st.markdown(
        '<div class="patient-title" style="text-align:center;">'
        '📊 Patient Summary'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="patient-description" style="text-align:center;">'
        'Clinical information used for this prediction'
        '</div>',
        unsafe_allow_html=True
    )

    summary = pd.DataFrame({

        "Clinical Feature": [
            "Age",
            "Sex",
            "Chest Pain Type",
            "Resting Blood Pressure",
            "Cholesterol",
            "Fasting Blood Sugar",
            "Resting ECG",
            "Maximum Heart Rate",
            "Exercise Induced Angina",
            "ST Depression (Oldpeak)",
            "ST Segment Slope",
            "Major Vessels",
            "Thalassemia Category"
        ],

        "Patient Value": [
            f"{age} years",
            sex_option,
            cp_option,
            f"{trestbps} mmHg",
            f"{chol} mg/dL",
            fbs_option,
            restecg_option,
            f"{thalach} bpm",
            exang_option,
            oldpeak,
            slope_option,
            ca,
            thal_option
        ]
    })

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.info(
        "⚠️ Academic prototype only. This system is not "
        "a substitute for professional medical diagnosis."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <b>Trustworthy AI Medical Decision Support</b>
        <br><br>
        Machine Learning • KNN • Heart Disease Prediction
    </div>
    """,
    unsafe_allow_html=True
)