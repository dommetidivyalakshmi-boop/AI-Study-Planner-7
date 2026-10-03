import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

# Lightweight AI/ML engine
# Uses only Python standard library + pandas so the app can run
# on Streamlit Cloud without scikit-learn.
import math
import re
from collections import Counter

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="AI Study Planner Pro",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PASTEL UI
# ============================================================

st.markdown("""
<style>
/* =========================================================
   CLEAR PASTEL UI - HIGH CONTRAST
   ========================================================= */
.stApp { background:#F7F9FC !important; }
.main .block-container { background:#F7F9FC !important; padding-top:2rem !important; }

/* Sidebar */
[data-testid="stSidebar"] { background:#EEF3FF !important; border-right:2px solid #D6E2F5 !important; }
[data-testid="stSidebar"] > div:first-child { background:#EEF3FF !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color:#263D78 !important; font-weight:700 !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea { background:#FFFFFF !important; color:#263A68 !important; -webkit-text-fill-color:#263A68 !important; border:2px solid #C9D9F2 !important; border-radius:12px !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:#FFFFFF !important; color:#263A68 !important; border:2px solid #C9D9F2 !important; border-radius:12px !important; }
[data-testid="stSidebar"] [data-baseweb="select"] span { color:#263A68 !important; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] { background:#FFFFFF !important; border-radius:12px !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button { background:#DDEBFF !important; color:#203A78 !important; border:2px solid #B8CDEC !important; border-radius:14px !important; font-weight:800 !important; min-height:46px !important; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#E9DFFF !important; border-color:#BFA9E8 !important; color:#263D78 !important; }

/* Title */
.main-title { text-align:center; font-size:46px; font-weight:900; color:#203A78 !important; margin-bottom:6px; }
.subtitle { text-align:center; font-size:18px; font-weight:750; color:#76518E !important; margin-bottom:22px; }

/* Section headers */
.section { background:#E8DEFF !important; padding:13px 18px; border-radius:18px; color:#263D78 !important; font-size:23px; font-weight:850; margin:22px 0 15px; border:2px solid #CFC0F0; }

/* Cards */
.card { background:#FFFFFF !important; border:2px solid #D7E1F2 !important; border-radius:18px; padding:18px; margin-bottom:12px; color:#33466F !important; box-shadow:0 4px 14px rgba(52,75,120,.08); }
.card b { color:#263D78 !important; }

/* Info / status cards */
.info,.success,.warning,.alarm { padding:16px 18px; border-radius:16px; font-weight:650; margin:10px 0; }
.info { background:#EAF4FF !important; border:2px solid #BBD8F7 !important; color:#294A7D !important; }
.info b { color:#203A78 !important; }
.success { background:#E5F8EC !important; border:2px solid #A9DDBD !important; color:#21643D !important; }
.warning { background:#FFF5D9 !important; border:2px solid #E8CC83 !important; color:#705719 !important; }
.alarm { background:#F2E9FF !important; border:2px solid #D5C0F0 !important; color:#5A3D79 !important; }

/* Metrics - strong readable text */
[data-testid="stMetric"] { background:#FFFFFF !important; border:2px solid #D7E1F2 !important; border-radius:16px !important; padding:14px !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color:#52688F !important; -webkit-text-fill-color:#52688F !important; font-weight:750 !important; }
[data-testid="stMetricValue"] { color:#203A78 !important; -webkit-text-fill-color:#203A78 !important; font-weight:900 !important; }
[data-testid="stMetricDelta"] { color:#52688F !important; -webkit-text-fill-color:#52688F !important; }

/* Headings and text */
h1,h2,h3,h4 { color:#263D78 !important; }
.stMarkdown,.stMarkdown p,.stMarkdown li { color:#33466F; }

/* Dataframe */
[data-testid="stDataFrame"] { border:2px solid #D7E1F2 !important; border-radius:14px !important; overflow:hidden !important; background:#FFFFFF !important; }

/* Footer */
.footer { padding:20px; text-align:center; border-radius:20px; background:#EAF4FF !important; color:#293F78 !important; border:2px solid #C9DDF5 !important; margin-top:25px; }

/* ---------------- CUSTOM CLEAR METRICS ---------------- */

.clear-metric {
    background: #FFFFFF !important;
    border: 2px solid #D2DFF2 !important;
    border-radius: 16px !important;
    padding: 14px 16px !important;
    min-height: 92px !important;
    box-shadow: 0 3px 10px rgba(52, 75, 120, 0.07) !important;
}

.clear-metric-label {
    color: #52688F !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    margin-bottom: 8px !important;
}

.clear-metric-value {
    color: #203A78 !important;
    font-size: 25px !important;
    font-weight: 900 !important;
    line-height: 1.15 !important;
}

.table-scroll {
    width: 100%;
    overflow-x: auto;
    background: #FFFFFF !important;
    border: 2px solid #D7E1F2 !important;
    border-radius: 14px !important;
    margin: 12px 0 20px 0 !important;
}

.clear-table {
    width: 100% !important;
    min-width: 900px !important;
    border-collapse: collapse !important;
    background: #FFFFFF !important;
    color: #263A68 !important;
    font-size: 14px !important;
}

.clear-table thead th {
    background: #E8DEFF !important;
    color: #263D78 !important;
    border-bottom: 2px solid #CFC0F0 !important;
    padding: 11px 10px !important;
    text-align: left !important;
    font-weight: 850 !important;
    white-space: nowrap !important;
}

.clear-table tbody td {
    background: #FFFFFF !important;
    color: #263A68 !important;
    border-bottom: 1px solid #E1E7F1 !important;
    padding: 10px !important;
    vertical-align: middle !important;
    white-space: nowrap !important;
}

.clear-table tbody tr:nth-child(even) td {
    background: #F7FAFF !important;
}

.clear-table tbody tr:hover td {
    background: #F1EBFF !important;
}

.eval-card {
    background: #FFFFFF !important;
    border: 2px solid #D7E1F2 !important;
    border-radius: 15px !important;
    padding: 13px 10px !important;
    text-align: center !important;
    min-height: 88px !important;
}

.eval-name {
    color: #52688F !important;
    font-size: 13px !important;
    font-weight: 800 !important;
}

.eval-status {
    color: #197044 !important;
    font-size: 18px !important;
    font-weight: 900 !important;
    margin-top: 8px !important;
}

.eval-status.fail {
    color: #B33A3A !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">📚 AI Study Planner Pro</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Adaptive AI/ML Planning • 100% Subject Coverage • Evaluation • Regeneration • Smart Alarms</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.markdown("## 🎓 Student Profile")

level = st.sidebar.selectbox(
    "Learner Level",
    [
        "1st–5th Class",
        "6th–10th Class",
        "Intermediate / 11th–12th",
        "Undergraduate",
        "Postgraduate",
        "Competitive Exam"
    ],
    index=None,
    placeholder="Select learner level"
)

goal = st.sidebar.selectbox(
    "Study Goal",
    [
        "Regular Study",
        "Exam Preparation",
        "Assignment",
        "Project Work",
        "Revision",
        "Competitive Exam Preparation"
    ],
    index=None,
    placeholder="Select study goal"
)

st.sidebar.markdown("## 📌 Study Constraints")

# Dates are intentionally empty until the user selects them.
start_date = st.sidebar.date_input(
    "📅 Start Date",
    value=None,
    help="Select the date on which your study plan should start."
)

# IMPORTANT: The timetable is created ONLY inside this user-entered time window.
# No fixed 00:00 / 00:30 / 01:00 list is used.
# The student can type ANY minute: 2:00 PM, 2:10 PM, 2:14 PM, 2:25 PM, 3:37 PM, etc.

def parse_clock(value):
    """Parse flexible user-entered times such as 2:14 PM, 14:14 or 2 PM."""
    text = str(value or "").strip().upper()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    candidates = [text, text.replace(".", ":")]
    formats = ["%I:%M %p", "%I %p", "%H:%M"]
    for candidate in candidates:
        for fmt in formats:
            try:
                parsed = datetime.strptime(candidate, fmt)
                return parsed
            except ValueError:
                continue
    return None


def format_user_minutes(minutes):
    minutes = int(minutes)
    hour = minutes // 60
    minute = minutes % 60
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour:02d}:{minute:02d} {suffix}"


st.sidebar.markdown("**⏰ Enter the exact time you want**")
st.sidebar.caption("Examples: 2:00 PM, 2:10 PM, 2:14 PM, 2:25 PM, 3:37 PM or 14:25")

start_time_text = st.sidebar.text_input(
    "🕘 Study Start Time",
    value="",
    placeholder="Example: 2:14 PM",
    help="Type any exact start time. There are no fixed 30-minute options."
)

end_time_text = st.sidebar.text_input(
    "🕐 Study End Time",
    value="",
    placeholder="Example: 6:37 PM",
    help="Type any exact end time. There are no fixed 30-minute options."
)

start_clock = parse_clock(start_time_text)
end_clock = parse_clock(end_time_text)

if start_clock is not None and end_clock is not None:
    start_minutes = start_clock.hour * 60 + start_clock.minute
    end_minutes = end_clock.hour * 60 + end_clock.minute
    study_window_minutes = end_minutes - start_minutes
else:
    start_minutes = None
    end_minutes = None
    study_window_minutes = 0

study_hours = study_window_minutes / 60

if study_window_minutes > 0:
    st.sidebar.info(
        f"⏱️ Available time per study day: {study_hours:g} hours "
        f"({start_time_text} – {end_time_text})"
    )
else:
    st.sidebar.info("⏱️ Select both start and end time to create the timetable.")

subjects_text = st.sidebar.text_area(
    "📚 Subjects",
    value="",
    height=90,
    placeholder="Example: Python, Database, AI, Blockchain",
    help="Enter subjects separated by commas."
)

priority_text = st.sidebar.text_input(
    "🎯 Priority Subjects / Topics",
    value="",
    placeholder="Example: AI, Python"
)

weak_text = st.sidebar.text_input(
    "⚠️ Weak / Difficult Subjects",
    value="",
    placeholder="Example: Database"
)

difficulty = st.sidebar.selectbox(
    "📈 Overall Difficulty",
    [1, 2, 3, 4, 5],
    index=None,
    placeholder="Select difficulty"
)

exam_date = st.sidebar.date_input(
    "📝 Main Exam Date",
    value=None,
    help="Select the exam date."
)

study_days = st.sidebar.multiselect(
    "📆 Study Days",
    [
        "Monday","Tuesday","Wednesday","Thursday",
        "Friday","Saturday","Sunday"
    ],
    default=[],
    placeholder="Select study days"
)

st.sidebar.markdown("## 🔔 Smart Alarm")

alarm_enabled = st.sidebar.checkbox(
    "Enable Smart Alarms",
    value=False
)

alarm_before = st.sidebar.selectbox(
    "Reminder Time",
    [0,5,10,15],
    index=None,
    placeholder="Select reminder time",
    format_func=lambda x: "At session start" if x == 0 else f"{x} min before"
)

revision_enabled = st.sidebar.checkbox(
    "🔄 Add Revision / Mock Test",
    value=False
)

generate_button = st.sidebar.button(
    "🚀 Generate AI Plan",
    use_container_width=True
)

regenerate_button = st.sidebar.button(
    "🔄 Intelligent Regeneration",
    use_container_width=True
)

# ============================================================
# BASIC HELPERS
# ============================================================

def clean_list(text):
    """Convert comma-separated text into a clean unique list."""
    result = []

    for item in str(text).split(","):
        item = item.strip()

        if item and item.casefold() not in [
            x.casefold() for x in result
        ]:
            result.append(item)

    return result


subjects = clean_list(subjects_text)
priority_subjects = clean_list(priority_text)
weak_subjects = clean_list(weak_text)


def is_match(subject, item_list):
    """Flexible subject matching."""
    subject = subject.casefold()

    return any(
        subject == item.casefold()
        or subject in item.casefold()
        or item.casefold() in subject
        for item in item_list
    )


def exam_urgency():
    """Safe 1-to-5 urgency score. Returns None until both dates exist."""
    if start_date is None or exam_date is None:
        return None

    days_left = (exam_date - start_date).days
    if days_left <= 0:
        return 5.0

    score = 5.0 - (days_left / 14.0)
    return round(max(1.0, min(5.0, score)), 2)


def subject_category(subject):
    """Simple NLP-style subject category."""
    name = subject.casefold()

    if any(x in name for x in [
        "ai", "artificial intelligence",
        "machine learning", "deep learning",
        "data science"
    ]):
        return "AI / Data"

    if any(x in name for x in [
        "python", "java", "c++", "program",
        "coding", "software", "web", "javascript"
    ]):
        return "Programming"

    if any(x in name for x in [
        "database", "dbms", "sql"
    ]):
        return "Database"

    if any(x in name for x in [
        "math", "mathematics", "statistics", "calculus"
    ]):
        return "Mathematics"

    if any(x in name for x in [
        "english", "language", "literature",
        "telugu", "hindi"
    ]):
        return "Language"

    if any(x in name for x in [
        "physics", "chemistry", "biology", "science"
    ]):
        return "Science"

    return "General"


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs():
    errors = []

    if level is None:
        errors.append("Please select the learner level.")

    if goal is None:
        errors.append("Please select the study goal.")

    if not subjects:
        errors.append("Please enter at least one subject.")

    if start_date is None:
        errors.append("Please select the start date.")

    if exam_date is None:
        errors.append("Please select the exam date.")

    if not study_days:
        errors.append("Please select at least one study day.")

    if start_clock is None or end_clock is None:
        errors.append("Please select both Study Start Time and Study End Time.")
    elif end_minutes <= start_minutes:
        errors.append("Study End Time must be later than Study Start Time.")

    if difficulty is None:
        errors.append("Please select the overall difficulty.")

    if alarm_enabled and alarm_before is None:
        errors.append("Please select the Smart Alarm reminder time.")

    if start_date is not None and exam_date is not None and exam_date <= start_date:
        errors.append("Exam Date must be after Start Date.")

    return errors


# ============================================================
# AIML MODEL 1: LIGHTWEIGHT WORKLOAD CLASSIFIER
# ============================================================
# A small K-Nearest-Neighbors style classifier is trained from
# labelled examples. It is real supervised ML, but needs no
# third-party machine-learning package.

workload_X = [
    [1,2,1,1,1], [2,3,2,2,1], [3,4,2,2,2],
    [4,4,3,2,2], [5,5,4,3,3], [6,6,4,4,3],
    [8,7,5,5,4], [10,8,5,5,5], [2,6,4,5,4],
    [6,3,5,4,4]
]

workload_y = [
    "Light", "Light", "Moderate", "Moderate",
    "Heavy", "Heavy", "Critical", "Critical",
    "Critical", "Heavy"
]


class SimpleKNNClassifier:
    """Small, dependency-free KNN classifier for this app."""

    def __init__(self, X, y, k=3):
        self.X = X
        self.y = y
        self.k = k

    @staticmethod
    def distance(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def predict_one(self, row):
        distances = [
            (self.distance(row, sample), label)
            for sample, label in zip(self.X, self.y)
        ]
        distances.sort(key=lambda item: item[0])
        nearest = distances[:self.k]

        votes = Counter(label for _, label in nearest)
        return votes.most_common(1)[0][0]


workload_model = SimpleKNNClassifier(
    workload_X,
    workload_y,
    k=3
)


def predict_workload():
    # The AI/ML analysis section is displayed before the user has
    # necessarily filled in every sidebar field.  In particular,
    # `difficulty` is None until the user selects it.  Passing None
    # into the KNN distance calculation causes:
    # TypeError: unsupported operand type(s) for -: ...
    #
    # Keep the UI unchanged and simply wait until the required
    # numeric input is available.
    if difficulty is None:
        return "Not set"

    values = [
        float(study_hours),
        max(1, len(subjects)),
        float(difficulty),
        len(priority_subjects) + 1,
        len(weak_subjects) + 1
    ]

    return workload_model.predict_one(values)


# ============================================================
# AIML MODEL 2: LIGHTWEIGHT SESSION-TIME REGRESSOR
# ============================================================
# KNN regression predicts a study-session duration from similar
# historical examples. The result is rounded to a practical slot.

time_X = [
    [1,1,1,1,1], [2,1,2,1,2], [3,2,2,2,2],
    [4,3,3,2,3], [5,4,4,3,4], [7,5,5,4,5],
    [10,5,5,5,5], [14,4,5,4,4], [2,5,4,5,5],
    [8,1,3,2,2], [4,4,1,1,3]
]

time_y = [25,30,35,40,50,60,75,80,70,35,30]


class SimpleKNNRegressor:
    """Small, dependency-free KNN regressor for session prediction."""

    def __init__(self, X, y, k=3):
        self.X = X
        self.y = y
        self.k = k

    @staticmethod
    def distance(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def predict_one(self, row):
        distances = [
            (self.distance(row, sample), target)
            for sample, target in zip(self.X, self.y)
        ]
        distances.sort(key=lambda item: item[0])
        nearest = distances[:self.k]

        # Inverse-distance weighted average.
        weighted_total = 0.0
        weight_total = 0.0

        for distance, target in nearest:
            weight = 1.0 / max(distance, 0.001)
            weighted_total += target * weight
            weight_total += weight

        return weighted_total / weight_total


time_model = SimpleKNNRegressor(
    time_X,
    time_y,
    k=3
)


# ============================================================
# SMART IMPORTANCE / PRIORITY
# ============================================================

def importance_score(subject):
    score = 1.0

    if is_match(subject, priority_subjects):
        score += 3.0

    if is_match(subject, weak_subjects):
        score += 2.2

    if subject_category(subject) in [
        "AI / Data",
        "Programming",
        "Database",
        "Mathematics"
    ]:
        score += 0.4

    urgency = exam_urgency()
    score += 0.65 * (urgency if urgency is not None else 1.0)
    score += 0.35 * (difficulty if difficulty is not None else 1)

    return round(score, 2)


def priority_level(subject):
    score = importance_score(subject)

    if score >= 7:
        return "Critical"

    if score >= 5:
        return "High"

    if score >= 3:
        return "Medium"

    return "Normal"


# ============================================================
# NLP: SIMPLE TEXT SIMILARITY TASK CLASSIFIER
# ============================================================
# Token-based cosine similarity provides lightweight NLP without
# requiring scikit-learn/TfidfVectorizer.

task_library = [
    "concept learning theory definitions fundamentals",
    "practice exercises problems coding questions",
    "revision recap important points formulas",
    "implementation project build practical application",
    "mock test exam questions assessment"
]

task_labels = [
    "Concept Learning",
    "Practice",
    "Revision",
    "Implementation",
    "Mock Test"
]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).casefold())


def text_cosine_similarity(text_a, text_b):
    a = Counter(tokenize(text_a))
    b = Counter(tokenize(text_b))

    if not a or not b:
        return 0.0

    common = set(a) & set(b)
    numerator = sum(a[token] * b[token] for token in common)

    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return numerator / (norm_a * norm_b)


def classify_task(subject):
    similarities = [
        text_cosine_similarity(subject, task)
        for task in task_library
    ]

    best_index = max(
        range(len(similarities)),
        key=lambda index: similarities[index]
    )

    if similarities[best_index] == 0:
        return "Concept Learning"

    return task_labels[best_index]


# ============================================================
# TASK DECOMPOSITION
# ============================================================

def get_tasks(subject):
    task_sets = {
        "1st–5th Class": [
            "Learn Basics",
            "Examples / Story Learning",
            "Practice",
            "Quick Revision"
        ],

        "6th–10th Class": [
            "Concept Learning",
            "Worked Examples",
            "Practice Questions",
            "Revision"
        ],

        "Intermediate / 11th–12th": [
            "Concepts",
            "Problem Solving",
            "Exam Questions",
            "Revision"
        ],

        "Undergraduate": [
            "Concepts",
            "Technical Practice",
            "Application / Problems",
            "Revision"
        ],

        "Postgraduate": [
            "Advanced Concepts",
            "Critical Analysis",
            "Research / Implementation",
            "Revision"
        ],

        "Competitive Exam": [
            "Core Concepts",
            "Timed Practice",
            "Previous Questions",
            "Mock Test"
        ]
    }

    return task_sets[level]


def get_task(subject, index):
    tasks = get_tasks(subject)

    return f"{subject} - {tasks[index % len(tasks)]}"


# ============================================================
# AI SESSION TIME
# ============================================================

def predicted_session_minutes(subject):
    if start_date is None or exam_date is None:
        days_left = 30
    else:
        days_left = max(1, (exam_date - start_date).days)

    features = [[
        days_left,
        importance_score(subject),
        difficulty,
        2 if is_match(subject, priority_subjects) else 1,
        2 if is_match(subject, weak_subjects) else 1
    ]]

    prediction = time_model.predict_one(features[0])

    prediction = round(prediction / 15) * 15
    prediction = max(15, min(120, prediction))

    return int(prediction)


# ============================================================
# AVAILABLE STUDY DATES
# ============================================================

def get_available_dates():
    result = []

    current = start_date
    days_to_check = min(
        14,
        max(1, (exam_date - start_date).days)
    )

    for _ in range(days_to_check):

        if current >= exam_date:
            break

        if current.strftime("%A") in study_days:
            result.append(current)

        current += timedelta(days=1)

    return result


# ============================================================
# TIME DISPLAY
# ============================================================

def clock_time(minutes):
    return datetime.strptime(
        f"{minutes // 60:02d}:{minutes % 60:02d}",
        "%H:%M"
    ).strftime("%I:%M %p")


def duration_text(minutes):
    hours, mins = divmod(int(minutes), 60)

    if hours and mins:
        return f"{hours} hr {mins} min"

    if hours:
        return f"{hours} hr"

    return f"{mins} min"


# ============================================================
# MAIN AI STUDY PLANNER
# ============================================================

def generate_plan():

    errors = validate_inputs()

    if errors:
        return None, errors

    dates = get_available_dates()

    if not dates:
        return None, [
            "No selected study day is available before the exam date."
        ]

    # HARD TIME CONSTRAINT: use exactly the user's entered start/end window.
    daily_minutes = int(end_minutes - start_minutes)
    total_minutes = daily_minutes * len(dates)

    # HARD RULE: 100% SUBJECT COVERAGE
    minimum_required = len(subjects) * 15

    if minimum_required > total_minutes:
        return None, [
            f"100% coverage needs at least "
            f"{minimum_required} minutes. "
            f"Available time is {total_minutes} minutes. "
            f"Increase study hours or study days."
        ]

    # --------------------------------------------------------
    # Dynamic AI weights
    # --------------------------------------------------------

    weights = {
        subject: importance_score(subject)
        for subject in subjects
    }

    total_weight = sum(weights.values())

    target_minutes = {}

    for subject in subjects:
        extra = (
            (total_minutes - minimum_required)
            * weights[subject]
            / total_weight
        )

        target_minutes[subject] = (
            15 + int(extra)
        )

    # Correct rounding
    difference = (
        total_minutes
        - sum(target_minutes.values())
    )

    ranked_subjects = sorted(
        subjects,
        key=lambda x: weights[x],
        reverse=True
    )

    index = 0

    while difference >= 15:
        target_minutes[
            ranked_subjects[index % len(ranked_subjects)]
        ] += 15

        difference -= 15
        index += 1

    # --------------------------------------------------------
    # Dynamic timetable
    # --------------------------------------------------------

    remaining = target_minutes.copy()
    task_index = {
        subject: 0
        for subject in subjects
    }

    rows = []

    for current_date in dates:

        available = daily_minutes
        current_time = start_minutes

        while (
            available >= 15
            and any(
                value >= 15
                for value in remaining.values()
            )
        ):

            active = [
                subject
                for subject in subjects
                if remaining[subject] >= 15
            ]

            selected = max(
                active,
                key=lambda subject:
                    weights[subject]
                    * remaining[subject]
                    / max(
                        15,
                        target_minutes[subject]
                    )
            )

            session = min(
                predicted_session_minutes(selected),
                remaining[selected],
                available
            )

            session = max(
                15,
                (int(session) // 15) * 15
            )

            session = min(
                session,
                remaining[selected],
                available
            )

            if session < 15:
                break

            start_display = clock_time(current_time)
            end_display = clock_time(
                current_time + session
            )

            rows.append({
                "Date": current_date.strftime("%d %b %Y"),
                "Day": current_date.strftime("%A"),
                "Time": (
                    f"{start_display} – "
                    f"{end_display}"
                ),
                "Start Time": start_display,
                "End Time": end_display,
                "Subject": selected,
                "What to Study": get_task(
                    selected,
                    task_index[selected]
                ),
                "Duration": duration_text(session),
                "Duration Minutes": session,
                "Priority": priority_level(selected),
                "Category": subject_category(selected),
                "AI Reason": (
                    f"Importance {weights[selected]:.1f} | "
                    f"Exam urgency {(exam_urgency() if exam_urgency() is not None else 1.0):.1f}/5"
                ),
                "_date": current_date,
                "_start": current_time
            })

            remaining[selected] -= session
            task_index[selected] += 1

            current_time += session
            available -= session

    # --------------------------------------------------------
    # Coverage recovery
    # --------------------------------------------------------

    scheduled_subjects = {
        row["Subject"]
        for row in rows
    }

    for subject in subjects:

        if subject in scheduled_subjects:
            continue

        placed = False

        for current_date in dates:

            used = sum(
                row["Duration Minutes"]
                for row in rows
                if row["_date"] == current_date
            )

            if used + 15 <= daily_minutes:

                start = start_minutes + used

                rows.append({
                    "Date": current_date.strftime("%d %b %Y"),
                    "Day": current_date.strftime("%A"),
                    "Time": (
                        f"{clock_time(start)} – "
                        f"{clock_time(start + 15)}"
                    ),
                    "Start Time": clock_time(start),
                    "End Time": clock_time(start + 15),
                    "Subject": subject,
                    "What to Study": get_task(subject, 0),
                    "Duration": "15 min",
                    "Duration Minutes": 15,
                    "Priority": priority_level(subject),
                    "Category": subject_category(subject),
                    "AI Reason": "100% coverage recovery slot",
                    "_date": current_date,
                    "_start": start
                })

                placed = True
                break

        if not placed:
            return None, [
                "100% subject coverage is not possible "
                "with the current time constraints."
            ]

    # --------------------------------------------------------
    # Revision / Mock Test
    # --------------------------------------------------------

    if revision_enabled and dates:

        last_date = dates[-1]

        used = sum(
            row["Duration Minutes"]
            for row in rows
            if row["_date"] == last_date
        )

        free = daily_minutes - used

        if free >= 30:

            revision_minutes = min(
                45,
                (free // 15) * 15
            )

            start = start_minutes + used

            rows.append({
                "Date": last_date.strftime("%d %b %Y"),
                "Day": last_date.strftime("%A"),
                "Time": (
                    f"{clock_time(start)} – "
                    f"{clock_time(start + revision_minutes)}"
                ),
                "Start Time": clock_time(start),
                "End Time": clock_time(
                    start + revision_minutes
                ),
                "Subject": "Revision / Mock Test",
                "What to Study": (
                    f"{goal} - Final Revision / Mock Test"
                ),
                "Duration": duration_text(
                    revision_minutes
                ),
                "Duration Minutes": revision_minutes,
                "Priority": "High",
                "Category": "Revision",
                "AI Reason": "Adaptive revision using spare time",
                "_date": last_date,
                "_start": start
            })

    # --------------------------------------------------------
    # Final dataframe
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["_date", "_start"]
    ).reset_index(drop=True)

    df["Hours"] = (
        df["Duration Minutes"] / 60
    )

    return df, []


# ============================================================
# PLAN EVALUATION
# ============================================================

def evaluate_plan(df):

    if df is None or df.empty:
        return {
            "coverage": False,
            "time": False,
            "exam": False,
            "priority": False,
            "decomposition": False,
            "all": False,
            "missing": subjects
        }

    scheduled = set(df["Subject"])

    missing = [
        subject
        for subject in subjects
        if subject not in scheduled
    ]

    coverage = len(missing) == 0

    daily_hours = df.groupby("Date")["Hours"].sum()

    time_check = bool(
        (daily_hours <= study_hours + 0.001).all()
    )

    normal_rows = df[
        df["Subject"] != "Revision / Mock Test"
    ]

    if normal_rows.empty:
        exam_check = True
    else:
        actual_dates = normal_rows["Date"].apply(
            lambda x: datetime.strptime(
                x, "%d %b %Y"
            ).date()
        )

        exam_check = bool(
            (actual_dates < exam_date).all()
        )

    priority_check = all(
        any(
            is_match(subject, [x])
            for x in scheduled
        )
        for subject in priority_subjects
        if any(
            is_match(subject, [x])
            for x in subjects
        )
    )

    decomposition_check = bool(
        df["What to Study"]
        .astype(str)
        .str.len()
        .gt(5)
        .all()
    )

    all_pass = (
        coverage
        and time_check
        and exam_check
        and priority_check
        and decomposition_check
    )

    return {
        "coverage": coverage,
        "time": time_check,
        "exam": exam_check,
        "priority": priority_check,
        "decomposition": decomposition_check,
        "all": all_pass,
        "missing": missing
    }



# ============================================================
# CLEAR UI HELPERS
# ============================================================

def render_metric_cards(items):
    """Render readable pastel metric cards without theme-dependent metric text."""
    cols = st.columns(len(items))
    for col, (label, value, icon) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="clear-metric">
                    <div class="clear-metric-label">{icon} {label}</div>
                    <div class="clear-metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_clear_table(dataframe):
    """Render a light, readable HTML table."""
    if dataframe is None or dataframe.empty:
        st.info("No table data available.")
        return

    html = dataframe.to_html(
        index=False,
        escape=False,
        classes="clear-table",
        border=0
    )

    st.markdown(
        f"""
        <div class="table-scroll">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SESSION STATE
# ============================================================

if "plan" not in st.session_state:
    st.session_state.plan = None


# ============================================================
# GENERATE / REGENERATE
# ============================================================

if generate_button or regenerate_button:

    plan, errors = generate_plan()

    if errors:

        for error in errors:
            st.error(error)

    else:

        st.session_state.plan = plan

        st.success(
            "✅ AI/ML study plan generated successfully!"
        )


# ============================================================
# AI/ML ANALYSIS
# ============================================================

st.markdown(
    '<div class="section">🧠 Adaptive AI/ML Analysis</div>',
    unsafe_allow_html=True
)

render_metric_cards([
    ("Learner Level", level, "🎓"),
    ("Study Goal", goal, "🎯"),
    ("Exam Urgency", f"{exam_urgency():.1f}/5" if exam_urgency() is not None else "Not set", "⚡"),
    ("AI Workload", predict_workload(), "🧩")
])

st.markdown(
    f"""
    <div class="info">
    <b>AI decision factors:</b>
    learner level, study goal, available hours,
    {len(subjects)} subjects, priority, weak subjects,
    difficulty, exam countdown and study days.
    <br><br>
    <b>Hard rule:</b> Every entered subject must be included
    in the timetable.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# REQUIRED PROJECT CONCEPTS
# ============================================================

st.markdown(
    '<div class="section">⚙️ Intelligent Planning Engine</div>',
    unsafe_allow_html=True
)

features = [
    (
        "🧩 Task Decomposition",
        "Breaks each subject into smaller learning tasks."
    ),
    (
        "🎯 Priority Handling",
        "Priority and weak subjects receive more planning weight."
    ),
    (
        "⏰ Time Constraint Handling",
        "Never intentionally schedules more than the daily limit."
    ),
    (
        "📚 100% Subject Coverage",
        "Every entered subject must appear in the timetable."
    ),
    (
        "🤖 AIML Models",
        "KNN-based ML predicts workload and session time."
    ),
    (
        "🔤 NLP",
        "Token-based cosine similarity supports task classification."
    ),
    (
        "🔎 Evaluation",
        "Checks coverage, time, exam, priority and decomposition."
    ),
    (
        "🔄 Intelligent Regeneration",
        "Changing inputs creates a fresh timetable."
    ),
    (
        "🔔 Smart Alarms",
        "Shows the next session and provides a browser notification test."
    )
]

left, right = st.columns(2)

for i, (title, description) in enumerate(features):

    with (left if i % 2 == 0 else right):

        st.markdown(
            f"""
            <div class="card">
            <b>{title}</b><br>
            {description}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# GENERATED TIMETABLE
# ============================================================

st.markdown(
    '<div class="section">📅 AI Generated Dynamic Timetable</div>',
    unsafe_allow_html=True
)

if st.session_state.plan is None:

    st.markdown(
        """
        <div class="info">
        👈 Enter your details in the sidebar and click
        <b>🚀 Generate AI Plan</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    df = st.session_state.plan

    evaluation = evaluate_plan(df)

    scheduled_subjects = [
        subject
        for subject in subjects
        if subject in set(df["Subject"])
    ]

    coverage_percent = round(
        100 * len(scheduled_subjects)
        / max(1, len(subjects))
    )

    total_hours = round(
        df["Hours"].sum(),
        2
    )

    priority_sessions = int(
        df["Priority"].isin(
            ["High", "Critical"]
        ).sum()
    )

    render_metric_cards([
        ("Entered Subjects", len(subjects), "📚"),
        ("Coverage", f"{coverage_percent}%", "✅"),
        ("Planned Hours", total_hours, "⏰"),
        ("Priority Sessions", priority_sessions, "🎯")
    ])

    if coverage_percent == 100:

        st.markdown(
            """
            <div class="success">
            ✅ <b>100% Subject Coverage Achieved.</b>
            Every entered subject is scheduled.
            </div>
            """,
            unsafe_allow_html=True
        )

    display_df = df.drop(
        columns=["_date", "_start"],
        errors="ignore"
    )

    render_clear_table(display_df)


    # ========================================================
    # SUBJECT COVERAGE
    # ========================================================

    st.markdown(
        "### 📚 Subject Coverage & AI Allocation"
    )

    coverage_data = []

    for subject in subjects:

        rows = df[
            df["Subject"] == subject
        ]

        minutes = int(
            rows["Duration Minutes"].sum()
        )

        coverage_data.append({
            "Subject": subject,
            "Scheduled": (
                "✅ Yes"
                if not rows.empty
                else "❌ No"
            ),
            "Time": duration_text(minutes),
            "Priority": priority_level(subject),
            "Category": subject_category(subject),
            "AI Importance": importance_score(subject)
        })

    coverage_df = pd.DataFrame(
        coverage_data
    )

    render_clear_table(coverage_df)


    # ========================================================
    # EVALUATION
    # ========================================================

    st.markdown(
        '<div class="section">✅ Plan Evaluation</div>',
        unsafe_allow_html=True
    )

    checks = [
        ("Coverage", "coverage"),
        ("Daily Time", "time"),
        ("Exam Date", "exam"),
        ("Priority", "priority"),
        ("Task Decomposition", "decomposition")
    ]

    eval_cols = st.columns(5)

    for column, (name, key) in zip(eval_cols, checks):
        status = "PASS ✅" if evaluation[key] else "CHECK ❌"
        status_class = "" if evaluation[key] else "fail"

        with column:
            st.markdown(
                f"""
                <div class="eval-card">
                    <div class="eval-name">{name}</div>
                    <div class="eval-status {status_class}">{status}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    if evaluation["all"]:

        st.markdown(
            """
            <div class="success">
            🎉 <b>Evaluation Passed.</b><br>
            All major scheduling constraints are satisfied.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        missing_text = (
            ", ".join(evaluation["missing"])
            if evaluation["missing"]
            else "None"
        )

        st.markdown(
            f"""
            <div class="warning">
            ⚠️ Some constraints need adjustment.<br>
            Missing subjects: {missing_text}
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # SMART ALARMS
    # ========================================================

    st.markdown(
        '<div class="section">🔔 Smart Alarm Center</div>',
        unsafe_allow_html=True
    )

    if alarm_enabled:

        first = df.iloc[0]

        reminder_text = (
            "at session start"
            if alarm_before == 0
            else f"{alarm_before} minutes before"
        )

        st.markdown(
            f"""
            <div class="alarm">
            🔔 <b>Next Study Session</b><br><br>
            📅 {first["Date"]}<br>
            ⏰ {first["Time"]}<br>
            📚 {first["Subject"]}<br>
            📝 {first["What to Study"]}<br>
            🔔 Reminder: {reminder_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Browser notification test
        import streamlit.components.v1 as components

        subject_for_js = str(
            first["Subject"]
        ).replace("'", "\\'")

        start_for_js = str(
            first["Start Time"]
        ).replace("'", "\\'")

        components.html(
            f"""
            <button
                onclick="notifyMe()"
                style="
                padding:10px 18px;
                border-radius:12px;
                border:1px solid #C7B5E8;
                background:#F1E8FF;
                color:#5A3D79;
                font-weight:700;
                cursor:pointer;">
                🔔 Test Browser Alarm
            </button>

            <script>
            function notifyMe() {{
                if (!("Notification" in window)) {{
                    alert(
                        "Browser notifications are not supported."
                    );
                    return;
                }}

                Notification.requestPermission().then(
                    function(permission) {{
                        if (permission === "granted") {{
                            new Notification(
                                "AI Study Planner Pro",
                                {{
                                    body:
                                    "Next session: {subject_for_js}"
                                    + " at {start_for_js}"
                                }}
                            );
                        }}
                    }}
                );
            }}
            </script>
            """,
            height=55
        )

        st.caption(
            "Browser alarms need notification permission and "
            "the app/browser to remain active."
        )

    else:

        st.info(
            "Smart alarms are disabled. "
            "Enable them from the sidebar."
        )


    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    csv_data = display_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download AI Study Plan CSV",
        data=csv_data,
        file_name="AI_Study_Planner_Pro_Weekly_Plan.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# AI/ML ARCHITECTURE
# ============================================================

st.markdown(
    '<div class="section">🏗️ AI/ML Architecture</div>',
    unsafe_allow_html=True
)

architecture = [
    (
        "📥 INPUT",
        "Level • Goal • Subjects • Hours • Exam • Priority • Weak Subjects"
    ),
    (
        "🧠 AIML ANALYSIS",
        "KNN workload/time prediction + NLP cosine-similarity task classification"
    ),
    (
        "🧩 TASK DECOMPOSITION",
        "Concept • Practice • Application • Revision • Mock Test"
    ),
    (
        "⏱️ DYNAMIC ALLOCATION",
        "Urgency + priority + weakness + difficulty + available time"
    ),
    (
        "🔎 EVALUATION",
        "Coverage • Time • Exam • Priority • Task Decomposition"
    ),
    (
        "🔄 OUTPUT",
        "Timetable • Smart Alarm • CSV • Regeneration"
    )
]

architecture_columns = st.columns(3)

for i, (title, text) in enumerate(architecture):

    with architecture_columns[i % 3]:

        st.markdown(
            f"""
            <div class="card" style="min-height:125px">
            <b>{title}</b><br><br>
            {text}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# ITERATIVE REFINEMENT
# ============================================================

st.markdown(
    '<div class="section">🔄 Iterative Refinement</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info">
    <b>Change → Regenerate → Evaluate → Improve</b><br><br>
    Change subjects, hours, exam date, priority,
    weak subjects, learner level or study days.
    Then click <b>🔄 Intelligent Regeneration</b>.
    The planner creates a new timetable.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FINAL OUTCOME
# ============================================================

st.markdown(
    '<div class="section">🏆 Final Outcome</div>',
    unsafe_allow_html=True
)

outcomes = [
    ("📅", "Personalized Study Plan"),
    ("🎯", "Priority & Weak Subject Focus"),
    ("⏰", "Dynamic Time Management"),
    ("🤖", "AI/ML Based Planning"),
    ("🔎", "Automatic Evaluation"),
    ("🔄", "Intelligent Regeneration")
]

outcome_columns = st.columns(3)

for i, (icon, text) in enumerate(outcomes):

    with outcome_columns[i % 3]:

        st.markdown(
            f"""
            <div class="card" style="text-align:center">
            <div style="font-size:32px">{icon}</div>
            <b>{text}</b>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    🌿 <b>AI Study Planner Pro</b> 🌿<br><br>
    A simple, adaptive AI/ML study planner
    for learners from primary school to postgraduate level.
    </div>
    """,
    unsafe_allow_html=True
)
