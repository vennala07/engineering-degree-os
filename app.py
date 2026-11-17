import json
import os
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="My Engineering Degree OS",
    page_icon="🎓",
    layout="wide",
)

DATA_DIR = Path("data")

STATUS_OPTIONS = ["Not Started", "In Progress", "Completed"]
ASSESSMENT_TYPES = [
    "IA1",
    "IA2",
    "Assignment",
    "Lab Evaluation",
    "Project Review",
    "Exam",
]

CSV_SCHEMAS = {
    "semesters.csv": [
        "semester",
        "academic_year",
        "status",
        "is_current",
        "target_credits",
        "notes",
    ],
    "courses.csv": [
        "semester",
        "course_code",
        "course_title",
        "credits",
        "course_type",
        "syllabus_topics",
        "status",
    ],
    "university_syllabus.csv": [
        "programme",
        "university",
        "regulation",
        "total_semesters",
        "total_credits_required",
        "syllabus_source",
        "overview_notes",
    ],
    "calendar.csv": [
        "date",
        "academic_year",
        "semester",
        "event_type",
        "title",
        "description",
    ],
    "assessments.csv": [
        "semester",
        "course_code",
        "assessment_type",
        "title",
        "due_date",
        "max_marks",
        "obtained_marks",
        "status",
    ],
    "scores.csv": [
        "semester",
        "course_code",
        "course_title",
        "credits",
        "internal_marks",
        "external_marks",
        "total_marks",
        "max_marks",
        "grade",
        "status",
    ],
    "plan_checklist.csv": [
        "id",
        "semester",
        "category",
        "item",
        "target_date",
        "status",
        "notes",
    ],
}

NAV_ITEMS = [
    "Dashboard",
    "Semesters 1-8",
    "Courses & Syllabus",
    "University Syllabus",
    "Academic / Leave Calendar",
    "Internal Assessments",
    "Status Tracker",
    "Scorecard",
    "Plan & Checklist",
    "Ask Degree AI",
    "Data Admin",
]


# -----------------------------------------------------------------------------
# Minimal Clean Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .sem-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
        background-color: rgba(128, 128, 128, 0.04);
    }
    .sem-card-current {
        border: 1.5px solid #2b6cb0;
        background-color: rgba(43, 108, 176, 0.07);
    }
    .sem-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .sem-meta {
        font-size: 0.88rem;
        opacity: 0.85;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data Helpers
# -----------------------------------------------------------------------------
def load_csv(filename: str, warn_if_missing: bool = True) -> pd.DataFrame | None:
    """Load a CSV from /data safely. Show a friendly warning instead of crashing."""
    path = DATA_DIR / filename
    if not path.exists():
        if warn_if_missing:
            st.warning(
                f"Missing data file: `{path}`. Add `{filename}` inside the `/data` folder "
                "or create it in **Data Admin**."
            )
        return None
    try:
        df = pd.read_csv(path)
        return df
    except pd.errors.EmptyDataError:
        cols = CSV_SCHEMAS.get(filename, [])
        return pd.DataFrame(columns=cols)
    except Exception as exc:
        st.warning(f"Could not read `{path}`: {exc}")
        return None


def save_csv(filename: str, df: pd.DataFrame) -> bool:
    """Save a DataFrame back to /data/<filename>."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_DIR / filename, index=False)
        return True
    except Exception as exc:
        st.error(f"Failed to save `{filename}`: {exc}")
        return False


def to_numeric_col(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def parse_dates_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def get_current_semester(sem_df: pd.DataFrame | None) -> int | None:
    if sem_df is None or sem_df.empty or "semester" not in sem_df.columns:
        return None
    df = sem_df.copy()
    df["semester_num"] = pd.to_numeric(df["semester"], errors="coerce")
    if "is_current" in df.columns:
        current_rows = df[parse_bool_series(df["is_current"])]
        if not current_rows.empty and pd.notna(current_rows.iloc[0]["semester_num"]):
            return int(current_rows.iloc[0]["semester_num"])
    if "status" in df.columns:
        in_prog = df[df["status"].astype(str).str.strip() == "In Progress"]
        if not in_prog.empty and pd.notna(in_prog.iloc[0]["semester_num"]):
            return int(in_prog.iloc[0]["semester_num"])
    return None


def get_gemini_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if isinstance(secret_key, str) and secret_key.strip():
            return secret_key.strip()
    except Exception:
        pass
    return None


def call_gemini_rest(api_key: str, prompt: str) -> str:
    """Call Gemini REST API using Python standard library (no extra dependencies)."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 600},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    candidates = data.get("candidates", [])
    if not candidates:
        return "No response returned by Gemini."
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip() or "Empty response."


# -----------------------------------------------------------------------------
# Page Views
# -----------------------------------------------------------------------------
def render_dashboard():
    st.title("Dashboard")
    st.caption("High-level snapshot of your engineering degree progress.")

    sem_df = load_csv("semesters.csv")
    courses_df = load_csv("courses.csv")
    uni_df = load_csv("university_syllabus.csv")
    cal_df = load_csv("calendar.csv")
    assess_df = load_csv("assessments.csv")
    scores_df = load_csv("scores.csv")

    today = date.today()

    # 1. Current Semester
    current_sem = get_current_semester(sem_df)
    current_sem_label = f"Semester {current_sem}" if current_sem else "Not set"

    # 2. Credits & Degree Completion %
    completed_credits = 0.0
    total_course_credits = 0.0
    if courses_df is not None and not courses_df.empty and "credits" in courses_df.columns:
        c_credits = to_numeric_col(courses_df["credits"])
        total_course_credits = float(c_credits.sum())
        if "status" in courses_df.columns:
            completed_mask = courses_df["status"].astype(str).str.strip() == "Completed"
            completed_credits = float(c_credits[completed_mask].sum())

    required_credits = 0.0
    if (
        uni_df is not None
        and not uni_df.empty
        and "total_credits_required" in uni_df.columns
    ):
        required_credits = float(to_numeric_col(uni_df["total_credits_required"]).iloc[0])

    credit_denom = required_credits if required_credits > 0 else total_course_credits
    degree_pct = (completed_credits / credit_denom * 100) if credit_denom > 0 else 0.0

    # 3. Next Assessment
    next_assess_str = "None upcoming"
    next_assess_sub = ""
    if assess_df is not None and not assess_df.empty and "due_date" in assess_df.columns:
        adf = assess_df.copy()
        adf["parsed_date"] = parse_dates_col(adf["due_date"])
        adf = adf[adf["parsed_date"].notna() & (adf["parsed_date"] >= today)]
        if "status" in adf.columns:
            adf = adf[adf["status"].astype(str).str.strip() != "Completed"]
        if not adf.empty:
            nxt = adf.sort_values("parsed_date").iloc[0]
            next_assess_str = f"{nxt.get('course_code', '')} - {nxt.get('assessment_type', '')}".strip(" -")
            next_assess_sub = f"{nxt.get('title', '')} ({nxt['parsed_date']})"

    # 4. Next Holiday / Leave
    next_leave_str = "None upcoming"
    next_leave_sub = ""
    if cal_df is not None and not cal_df.empty and "date" in cal_df.columns:
        cdf = cal_df.copy()
        cdf["parsed_date"] = parse_dates_col(cdf["date"])
        cdf = cdf[cdf["parsed_date"].notna() & (cdf["parsed_date"] >= today)]
        if "event_type" in cdf.columns:
            leave_mask = cdf["event_type"].astype(str).str.lower().str.contains("holiday|leave")
            cdf = cdf[leave_mask]
        if not cdf.empty:
            nxt_l = cdf.sort_values("parsed_date").iloc[0]
            next_leave_str = str(nxt_l.get("title", "Holiday / Leave"))
            next_leave_sub = f"{nxt_l.get('event_type', '')} ({nxt_l['parsed_date']})"

    # Top Metrics (clean, minimal)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Semester", current_sem_label)
    m2.metric("Degree Completion", f"{degree_pct:.1f}%")
    denom_label = f"/ {credit_denom:.0f}" if credit_denom > 0 else ""
    m3.metric("Completed Credits", f"{completed_credits:.0f} {denom_label}".strip())

    # Score Summary Metric
    score_metric_val = "No scores yet"
    if (
        scores_df is not None
        and not scores_df.empty
        and {"total_marks", "max_marks"}.issubset(scores_df.columns)
    ):
        scored = scores_df.copy()
        scored["tot"] = to_numeric_col(scored["total_marks"])
        scored["max"] = to_numeric_col(scored["max_marks"])
        scored = scored[scored["max"] > 0]
        if not scored.empty:
            avg_pct = (scored["tot"].sum() / scored["max"].sum()) * 100
            score_metric_val = f"{avg_pct:.1f}%"
    m4.metric("Score Avg (Prototype)", score_metric_val)

    st.divider()

    # Upcoming Highlights + Score Summary
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Upcoming Schedule")
        st.markdown(f"**Next Assessment:** {next_assess_str}")
        if next_assess_sub:
            st.caption(next_assess_sub)
        st.markdown(f"**Next Holiday / Leave:** {next_leave_str}")
        if next_leave_sub:
            st.caption(next_leave_sub)

    with col_right:
        st.subheader("Score Summary")
        if (
            scores_df is not None
            and not scores_df.empty
            and {"semester", "total_marks", "max_marks"}.issubset(scores_df.columns)
        ):
            sdf = scores_df.copy()
            sdf["total_marks"] = to_numeric_col(sdf["total_marks"])
            sdf["max_marks"] = to_numeric_col(sdf["max_marks"])
            sdf = sdf[sdf["max_marks"] > 0]
            if not sdf.empty:
                sem_scores = (
                    sdf.groupby("semester", as_index=False)[["total_marks", "max_marks"]]
                    .sum()
                    .sort_values("semester")
                )
                sem_scores["Percentage (%)"] = (
                    sem_scores["total_marks"] / sem_scores["max_marks"] * 100
                ).round(2)
                st.dataframe(
                    sem_scores.rename(
                        columns={
                            "semester": "Semester",
                            "total_marks": "Total Marks",
                            "max_marks": "Max Marks",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Note: Overall average shown above is a simple percentage prototype metric.")
            else:
                st.info("No graded entries with max_marks > 0 in `scores.csv`.")
        else:
            st.info("Score summary will appear once `scores.csv` has scored courses.")

    st.divider()

    # Semester Progress Overview
    st.subheader("Semester Progress (1–8)")
    if sem_df is not None and not sem_df.empty:
        rows = []
        for s in range(1, 9):
            s_row = sem_df[pd.to_numeric(sem_df["semester"], errors="coerce") == s]
            status = str(s_row.iloc[0]["status"]) if not s_row.empty else "Not Listed"
            acad_yr = str(s_row.iloc[0].get("academic_year", "-")) if not s_row.empty else "-"

            c_done, c_total, cred_done, cred_total = 0, 0, 0.0, 0.0
            if courses_df is not None and not courses_df.empty:
                sc = courses_df[pd.to_numeric(courses_df["semester"], errors="coerce") == s]
                c_total = len(sc)
                if c_total > 0:
                    cred_total = float(to_numeric_col(sc["credits"]).sum())
                    done_mask = sc["status"].astype(str).str.strip() == "Completed"
                    c_done = int(done_mask.sum())
                    cred_done = float(to_numeric_col(sc.loc[done_mask, "credits"]).sum())

            pct = (c_done / c_total * 100) if c_total > 0 else 0.0
            rows.append(
                {
                    "Semester": f"Sem {s}",
                    "Academic Year": acad_yr,
                    "Status": status,
                    "Courses Completed": f"{c_done} / {c_total}",
                    "Credits Completed": f"{cred_done:.0f} / {cred_total:.0f}",
                    "Completion (%)": round(pct, 1),
                }
            )
        prog_df = pd.DataFrame(rows)
        st.dataframe(
            prog_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Completion (%)": st.column_config.ProgressColumn(
                    "Completion (%)", min_value=0, max_value=100, format="%.1f%%"
                )
            },
        )


def render_semesters():
    st.title("Semesters 1–8")
    st.caption("Overview cards for all eight semesters and detailed semester drill-down.")

    sem_df = load_csv("semesters.csv")
    courses_df = load_csv("courses.csv", warn_if_missing=False)
    assess_df = load_csv("assessments.csv", warn_if_missing=False)
    cal_df = load_csv("calendar.csv", warn_if_missing=False)
    scores_df = load_csv("scores.csv", warn_if_missing=False)

    # Render 8 Semester Cards (2 rows x 4 cols)
    for row_start in (1, 5):
        cols = st.columns(4)
        for idx, sem_num in enumerate(range(row_start, row_start + 4)):
            status = "Unknown"
            acad_yr = "-"
            is_curr = False
            notes = ""

            if sem_df is not None and not sem_df.empty and "semester" in sem_df.columns:
                match = sem_df[pd.to_numeric(sem_df["semester"], errors="coerce") == sem_num]
                if not match.empty:
                    rec = match.iloc[0]
                    status = str(rec.get("status", "Not Started"))
                    acad_yr = str(rec.get("academic_year", "-"))
                    is_curr = bool(parse_bool_series(pd.Series([rec.get("is_current", False)])).iloc[0])
                    notes = str(rec.get("notes", "")) if pd.notna(rec.get("notes")) else ""

            c_count, c_done, cr_done, cr_tot = 0, 0, 0.0, 0.0
            if courses_df is not None and not courses_df.empty:
                sc = courses_df[pd.to_numeric(courses_df["semester"], errors="coerce") == sem_num]
                c_count = len(sc)
                if c_count > 0:
                    cr_tot = float(to_numeric_col(sc["credits"]).sum())
                    done_m = sc["status"].astype(str).str.strip() == "Completed"
                    c_done = int(done_m.sum())
                    cr_done = float(to_numeric_col(sc.loc[done_m, "credits"]).sum())

            card_cls = "sem-card sem-card-current" if is_curr else "sem-card"
            curr_tag = " · Current" if is_curr else ""
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="{card_cls}">
                        <div class="sem-title">Semester {sem_num}{curr_tag}</div>
                        <div class="sem-meta">
                            <b>Year:</b> {acad_yr}<br>
                            <b>Status:</b> {status}<br>
                            <b>Courses:</b> {c_done}/{c_count} completed<br>
                            <b>Credits:</b> {cr_done:.0f}/{cr_tot:.0f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if notes:
                    st.caption(notes)

    st.divider()
    st.subheader("Semester Details")

    default_sem = get_current_semester(sem_df) or 1
    selected_sem = st.selectbox(
        "Select a semester to inspect",
        options=list(range(1, 9)),
        index=default_sem - 1,
        format_func=lambda x: f"Semester {x}",
    )

    tab_courses, tab_assess, tab_cal, tab_scores = st.tabs(
        ["Courses", "Assessments", "Calendar Events", "Scores"]
    )

    with tab_courses:
        if courses_df is None:
            st.info("Load `courses.csv` to view courses for this semester.")
        else:
            filtered = courses_df[
                pd.to_numeric(courses_df["semester"], errors="coerce") == selected_sem
            ]
            if filtered.empty:
                st.info(f"No courses recorded for Semester {selected_sem}.")
            else:
                st.dataframe(filtered, use_container_width=True, hide_index=True)

    with tab_assess:
        if assess_df is None:
            st.info("Load `assessments.csv` to view assessments for this semester.")
        else:
            filtered = assess_df[
                pd.to_numeric(assess_df["semester"], errors="coerce") == selected_sem
            ]
            if filtered.empty:
                st.info(f"No assessments recorded for Semester {selected_sem}.")
            else:
                st.dataframe(filtered, use_container_width=True, hide_index=True)

    with tab_cal:
        if cal_df is None:
            st.info("Load `calendar.csv` to view events for this semester.")
        else:
            filtered = cal_df[
                pd.to_numeric(cal_df["semester"], errors="coerce") == selected_sem
            ]
            if filtered.empty:
                st.info(f"No calendar events recorded for Semester {selected_sem}.")
            else:
                st.dataframe(filtered, use_container_width=True, hide_index=True)

    with tab_scores:
        if scores_df is None:
            st.info("Load `scores.csv` to view scores for this semester.")
        else:
            filtered = scores_df[
                pd.to_numeric(scores_df["semester"], errors="coerce") == selected_sem
            ]
            if filtered.empty:
                st.info(f"No scores recorded for Semester {selected_sem}.")
            else:
                st.dataframe(filtered, use_container_width=True, hide_index=True)


def render_courses_syllabus():
    st.title("Courses & Syllabus")
    st.caption("Browse courses and syllabus topics filtered by semester.")

    courses_df = load_csv("courses.csv")
    if courses_df is None or courses_df.empty:
        return

    df = courses_df.copy()
    df["sem_num"] = pd.to_numeric(df["semester"], errors="coerce")

    col1, col2 = st.columns([1, 1])
    with col1:
        sem_options = ["All Semesters"] + [f"Semester {s}" for s in range(1, 9)]
        sem_choice = st.selectbox("Filter by Semester", sem_options)
    with col2:
        status_filter = st.multiselect(
            "Filter by Status",
            options=STATUS_OPTIONS,
            default=STATUS_OPTIONS,
        )

    if sem_choice != "All Semesters":
        target_sem = int(sem_choice.replace("Semester ", ""))
        df = df[df["sem_num"] == target_sem]

    if status_filter and "status" in df.columns:
        df = df[df["status"].astype(str).str.strip().isin(status_filter)]

    display_cols = [
        c
        for c in [
            "semester",
            "course_code",
            "course_title",
            "credits",
            "course_type",
            "syllabus_topics",
            "status",
        ]
        if c in df.columns
    ]

    total_credits = float(to_numeric_col(df["credits"]).sum()) if "credits" in df.columns else 0
    st.caption(f"Showing **{len(df)}** courses | Total Credits: **{total_credits:.0f}**")

    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


def render_university_syllabus():
    st.title("University Syllabus")
    st.caption("Programme-level overview and official syllabus reference.")

    uni_df = load_csv("university_syllabus.csv")
    if uni_df is None or uni_df.empty:
        return

    row = uni_df.iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Programme", str(row.get("programme", "-")))
    c2.metric("Total Semesters", str(row.get("total_semesters", "8")))
    c3.metric("Required Credits", str(row.get("total_credits_required", "-")))

    st.subheader("Programme Overview")
    st.write(f"**University:** {row.get('university', '-')}")
    st.write(f"**Regulation / Scheme:** {row.get('regulation', '-')}")
    if pd.notna(row.get("overview_notes")):
        st.info(str(row.get("overview_notes")))

    st.divider()
    st.subheader("Syllabus Source / Reference")
    current_source = "" if pd.isna(row.get("syllabus_source")) else str(row.get("syllabus_source"))
    current_notes = "" if pd.isna(row.get("overview_notes")) else str(row.get("overview_notes"))

    with st.form("syllabus_ref_form"):
        new_source = st.text_input(
            "Syllabus Source / Reference (URL, Handbook Edition, or File Reference)",
            value=current_source,
            placeholder="e.g., https://university.edu/syllabus-2023.pdf or Academic Handbook 2023-27",
        )
        new_notes = st.text_area("Programme Overview Notes", value=current_notes, height=100)
        submitted = st.form_submit_button("Save Syllabus Reference")
        if submitted:
            updated = uni_df.copy()
            updated.loc[updated.index[0], "syllabus_source"] = new_source.strip()
            updated.loc[updated.index[0], "overview_notes"] = new_notes.strip()
            if save_csv("university_syllabus.csv", updated):
                st.success("Saved syllabus source/reference to `data/university_syllabus.csv`.")
                st.rerun()


def render_calendar():
    st.title("Academic / Leave Calendar")
    st.caption("Track academic milestones, exam windows, holidays, and personal leaves.")

    cal_df = load_csv("calendar.csv")
    if cal_df is None or cal_df.empty:
        return

    df = cal_df.copy()
    df["parsed_date"] = parse_dates_col(df["date"])

    col1, col2, col3 = st.columns(3)
    with col1:
        years = ["All Years"] + sorted(
            [str(y) for y in df["academic_year"].dropna().unique()]
        )
        sel_year = st.selectbox("Academic Year", years)
    with col2:
        sems = ["All Semesters"] + sorted(
            [str(s) for s in df["semester"].dropna().unique()],
            key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
        )
        sel_sem = st.selectbox("Semester", sems)
    with col3:
        types = ["All Types"] + sorted(
            [str(t) for t in df["event_type"].dropna().unique()]
        )
        sel_type = st.selectbox("Event Type", types)

    if sel_year != "All Years":
        df = df[df["academic_year"].astype(str) == sel_year]
    if sel_sem != "All Semesters":
        df = df[df["semester"].astype(str) == sel_sem]
    if sel_type != "All Types":
        df = df[df["event_type"].astype(str) == sel_type]

    df = df.sort_values("parsed_date", na_position="last")

    # Summary counts
    holiday_count = int(
        df["event_type"].astype(str).str.lower().str.contains("holiday|leave").sum()
    )
    acad_count = len(df) - holiday_count
    m1, m2, m3 = st.columns(3)
    m1.metric("Filtered Events", len(df))
    m2.metric("Academic / Exam Events", acad_count)
    m3.metric("Holidays / Leaves", holiday_count)

    show_cols = [
        c
        for c in ["date", "academic_year", "semester", "event_type", "title", "description"]
        if c in df.columns
    ]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)


def render_assessments():
    st.title("Internal Assessments")
    st.caption("Manage IA1, IA2, assignments, lab evaluations, project reviews, and exams.")

    assess_df = load_csv("assessments.csv")
    if assess_df is None or assess_df.empty:
        return

    df = assess_df.copy()
    df["parsed_date"] = parse_dates_col(df["due_date"])

    c1, c2, c3 = st.columns(3)
    with c1:
        sem_opts = ["All Semesters"] + [f"Semester {s}" for s in range(1, 9)]
        sel_sem = st.selectbox("Semester", sem_opts)
    with c2:
        existing_types = sorted(df["assessment_type"].dropna().astype(str).unique().tolist())
        all_types = list(dict.fromkeys(ASSESSMENT_TYPES + existing_types))
        sel_types = st.multiselect("Assessment Type", all_types, default=all_types)
    with c3:
        sel_status = st.multiselect("Status", STATUS_OPTIONS, default=STATUS_OPTIONS)

    if sel_sem != "All Semesters":
        target_s = int(sel_sem.replace("Semester ", ""))
        df = df[pd.to_numeric(df["semester"], errors="coerce") == target_s]
    if sel_types:
        df = df[df["assessment_type"].astype(str).isin(sel_types)]
    if sel_status:
        df = df[df["status"].astype(str).isin(sel_status)]

    df = df.sort_values("parsed_date", na_position="last")

    completed_cnt = int((df["status"].astype(str) == "Completed").sum())
    pending_cnt = len(df) - completed_cnt

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Shown", len(df))
    m2.metric("Completed", completed_cnt)
    m3.metric("Pending / In Progress", pending_cnt)

    show_cols = [
        c
        for c in [
            "semester",
            "course_code",
            "assessment_type",
            "title",
            "due_date",
            "max_marks",
            "obtained_marks",
            "status",
        ]
        if c in df.columns
    ]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)


def render_status_tracker():
    st.title("Status Tracker")
    st.caption("Unified completion tracking across Courses, Assessments, and Plan Checklist.")

    courses_df = load_csv("courses.csv", warn_if_missing=False)
    assess_df = load_csv("assessments.csv", warn_if_missing=False)
    plan_df = load_csv("plan_checklist.csv", warn_if_missing=False)

    if courses_df is None and assess_df is None and plan_df is None:
        st.warning("No CSV files found for tracking. Add data files under `/data`.")
        return

    sem_filter = st.selectbox(
        "Filter by Semester",
        ["All Semesters"] + [f"Semester {s}" for s in range(1, 9)],
    )

    def summarize_status(df: pd.DataFrame | None, label: str) -> dict:
        if df is None or df.empty or "status" not in df.columns:
            return {
                "Area": label,
                "Not Started": 0,
                "In Progress": 0,
                "Completed": 0,
                "Total": 0,
            }
        sub = df.copy()
        if sem_filter != "All Semesters" and "semester" in sub.columns:
            target = int(sem_filter.replace("Semester ", ""))
            sub = sub[pd.to_numeric(sub["semester"], errors="coerce") == target]
        s_col = sub["status"].astype(str).str.strip()
        return {
            "Area": label,
            "Not Started": int((s_col == "Not Started").sum()),
            "In Progress": int((s_col == "In Progress").sum()),
            "Completed": int((s_col == "Completed").sum()),
            "Total": len(sub),
        }

    summary_rows = [
        summarize_status(courses_df, "Courses"),
        summarize_status(assess_df, "Internal Assessments"),
        summarize_status(plan_df, "Plan & Checklist"),
    ]
    summary_df = pd.DataFrame(summary_rows)

    tot_ns = int(summary_df["Not Started"].sum())
    tot_ip = int(summary_df["In Progress"].sum())
    tot_co = int(summary_df["Completed"].sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Not Started", tot_ns)
    m2.metric("In Progress", tot_ip)
    m3.metric("Completed", tot_co)

    st.subheader("Completion Breakdown")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Detailed Items by Status")
    selected_status = st.radio(
        "Show items with status:",
        STATUS_OPTIONS,
        horizontal=True,
    )

    combined_items = []
    for area, df, title_col in [
        ("Course", courses_df, "course_title"),
        ("Assessment", assess_df, "title"),
        ("Checklist", plan_df, "item"),
    ]:
        if df is not None and not df.empty and "status" in df.columns:
            sub = df[df["status"].astype(str).str.strip() == selected_status].copy()
            if sem_filter != "All Semesters" and "semester" in sub.columns:
                target = int(sem_filter.replace("Semester ", ""))
                sub = sub[pd.to_numeric(sub["semester"], errors="coerce") == target]
            for _, r in sub.iterrows():
                combined_items.append(
                    {
                        "Area": area,
                        "Semester": r.get("semester", "-"),
                        "Code / Category": r.get("course_code", r.get("category", "-")),
                        "Title / Item": r.get(title_col, "-"),
                        "Status": r.get("status", "-"),
                    }
                )

    if combined_items:
        st.dataframe(pd.DataFrame(combined_items), use_container_width=True, hide_index=True)
    else:
        st.info(f"No items marked as **{selected_status}** for the current filter.")


def render_scorecard():
    st.title("Scorecard")
    st.caption("Semester-wise marks, percentages, and grades from `data/scores.csv`.")

    scores_df = load_csv("scores.csv")
    if scores_df is None or scores_df.empty:
        return

    df = scores_df.copy()
    for col in ["internal_marks", "external_marks", "total_marks", "max_marks", "credits"]:
        if col in df.columns:
            df[col] = to_numeric_col(df[col])

    df["Course %"] = df.apply(
        lambda r: round((r["total_marks"] / r["max_marks"]) * 100, 2)
        if r.get("max_marks", 0) > 0
        else 0.0,
        axis=1,
    )

    valid_df = df[df["max_marks"] > 0]

    # Prototype overall metric (clearly labeled)
    if not valid_df.empty:
        overall_pct = (valid_df["total_marks"].sum() / valid_df["max_marks"].sum()) * 100
        sem_summary = (
            valid_df.groupby("semester", as_index=False)[["total_marks", "max_marks", "credits"]]
            .sum()
            .sort_values("semester")
        )
        sem_summary["Semester Percentage (%)"] = (
            sem_summary["total_marks"] / sem_summary["max_marks"] * 100
        ).round(2)

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Simple Average % (Prototype Metric)",
            f"{overall_pct:.2f}%",
        )
        m2.metric("Scored Courses", len(valid_df))
        m3.metric("Scored Semesters", len(sem_summary))

        st.info(
            "**Prototype Metric Notice:** Percentages shown here are simple marks-based averages "
            "from `scores.csv` (Total Obtained / Total Max Marks). No official university CGPA "
            "or credit-weighted grading formula is assumed."
        )

        st.subheader("Semester Percentage Summary")
        st.dataframe(
            sem_summary.rename(
                columns={
                    "semester": "Semester",
                    "total_marks": "Obtained Marks",
                    "max_marks": "Max Marks",
                    "credits": "Credits",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No courses with `max_marks > 0` found in `scores.csv`.")

    st.divider()
    st.subheader("Course-Level Scorecard")
    sem_opts = ["All Semesters"] + sorted(
        [str(s) for s in df["semester"].dropna().unique()],
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )
    sel_sem = st.selectbox("Filter Scorecard by Semester", sem_opts)
    filtered = df if sel_sem == "All Semesters" else df[df["semester"].astype(str) == sel_sem]

    show_cols = [
        c
        for c in [
            "semester",
            "course_code",
            "course_title",
            "credits",
            "internal_marks",
            "external_marks",
            "total_marks",
            "max_marks",
            "Course %",
            "grade",
            "status",
        ]
        if c in filtered.columns
    ]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)


def render_plan_checklist():
    st.title("Plan & Checklist")
    st.caption("Create and edit action items stored in `data/plan_checklist.csv`.")

    plan_df = load_csv("plan_checklist.csv")
    if plan_df is None:
        plan_df = pd.DataFrame(columns=CSV_SCHEMAS["plan_checklist.csv"])

    # Quick Add Form
    with st.expander("Add New Checklist Item", expanded=False):
        with st.form("add_checklist_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                sem_val = st.selectbox("Semester", list(range(1, 9)), index=0)
            with c2:
                cat_val = st.text_input("Category", placeholder="e.g., Academics, Internship, Exam Prep")
            with c3:
                date_val = st.date_input("Target Date", value=date.today())

            item_val = st.text_input("Checklist Item", placeholder="e.g., Complete Lab Record")
            c4, c5 = st.columns([1, 2])
            with c4:
                status_val = st.selectbox("Status", STATUS_OPTIONS, index=0)
            with c5:
                notes_val = st.text_input("Notes", placeholder="Optional notes")

            if st.form_submit_button("Add Item"):
                if not item_val.strip():
                    st.warning("Please enter a checklist item description.")
                else:
                    next_id = 1
                    if not plan_df.empty and "id" in plan_df.columns:
                        max_id = pd.to_numeric(plan_df["id"], errors="coerce").max()
                        if pd.notna(max_id):
                            next_id = int(max_id) + 1
                    new_row = pd.DataFrame(
                        [
                            {
                                "id": next_id,
                                "semester": sem_val,
                                "category": cat_val.strip() or "General",
                                "item": item_val.strip(),
                                "target_date": str(date_val),
                                "status": status_val,
                                "notes": notes_val.strip(),
                            }
                        ]
                    )
                    updated = pd.concat([plan_df, new_row], ignore_index=True)
                    if save_csv("plan_checklist.csv", updated):
                        st.success("Checklist item added and saved to `data/plan_checklist.csv`.")
                        st.rerun()

    st.subheader("Editable Checklist")
    st.caption("Edit cells directly, add or delete rows, then click **Save Checklist**.")

    editable_df = plan_df.copy()
    for col in CSV_SCHEMAS["plan_checklist.csv"]:
        if col not in editable_df.columns:
            editable_df[col] = ""

    edited = st.data_editor(
        editable_df[CSV_SCHEMAS["plan_checklist.csv"]],
        num_rows="dynamic",
        use_container_width=True,
        key="checklist_editor",
        column_config={
            "semester": st.column_config.NumberColumn("Semester", min_value=1, max_value=8, step=1),
            "status": st.column_config.SelectboxColumn("Status", options=STATUS_OPTIONS, required=True),
        },
    )

    if st.button("Save Checklist", type="primary"):
        if save_csv("plan_checklist.csv", edited):
            st.success("Saved changes to `data/plan_checklist.csv`.")
            st.rerun()


def answer_from_csv_locally(question: str) -> str:
    """Simple deterministic Q&A over loaded CSV files when Gemini API is unavailable."""
    q = question.lower().strip()
    today = date.today()

    sem_df = load_csv("semesters.csv", warn_if_missing=False)
    courses_df = load_csv("courses.csv", warn_if_missing=False)
    assess_df = load_csv("assessments.csv", warn_if_missing=False)
    cal_df = load_csv("calendar.csv", warn_if_missing=False)
    scores_df = load_csv("scores.csv", warn_if_missing=False)
    plan_df = load_csv("plan_checklist.csv", warn_if_missing=False)

    # Check if user mentions a specific semester number
    mentioned_sem = None
    for s in range(1, 9):
        if f"sem {s}" in q or f"semester {s}" in q:
            mentioned_sem = s
            break

    if "current semester" in q or "which semester" in q:
        curr = get_current_semester(sem_df)
        return f"Your current semester is **Semester {curr}**." if curr else "Current semester is not marked in `semesters.csv`."

    if "credit" in q or "completion" in q or "progress" in q:
        if courses_df is not None and not courses_df.empty:
            tot = float(to_numeric_col(courses_df["credits"]).sum())
            done_mask = courses_df["status"].astype(str).str.strip() == "Completed"
            done = float(to_numeric_col(courses_df.loc[done_mask, "credits"]).sum())
            return f"You have completed **{done:.0f}** out of **{tot:.0f}** listed course credits across your CSV."
        return "Course credit data is not available (`courses.csv` missing or empty)."

    if any(k in q for k in ["assessment", "exam", "ia1", "ia2", "assignment", "lab", "due"]):
        if assess_df is not None and not assess_df.empty:
            adf = assess_df.copy()
            adf["parsed_date"] = parse_dates_col(adf["due_date"])
            if mentioned_sem:
                adf = adf[pd.to_numeric(adf["semester"], errors="coerce") == mentioned_sem]
            upcoming = adf[
                (adf["parsed_date"].notna())
                & (adf["parsed_date"] >= today)
                & (adf["status"].astype(str).str.strip() != "Completed")
            ].sort_values("parsed_date")
            if not upcoming.empty:
                lines = [
                    f"- **{r['course_code']} ({r['assessment_type']})**: {r['title']} — due {r['due_date']} [{r['status']}]"
                    for _, r in upcoming.head(5).iterrows()
                ]
                return "Upcoming assessments from `assessments.csv`:\n" + "\n".join(lines)
            return "No upcoming pending assessments found for that filter."
        return "Assessment data is not available (`assessments.csv` missing or empty)."

    if any(k in q for k in ["holiday", "leave", "calendar", "break", "vacation"]):
        if cal_df is not None and not cal_df.empty:
            cdf = cal_df.copy()
            cdf["parsed_date"] = parse_dates_col(cdf["date"])
            upcoming = cdf[cdf["parsed_date"].notna() & (cdf["parsed_date"] >= today)]
            if "holiday" in q or "leave" in q:
                upcoming = upcoming[
                    upcoming["event_type"].astype(str).str.lower().str.contains("holiday|leave")
                ]
            upcoming = upcoming.sort_values("parsed_date")
            if not upcoming.empty:
                lines = [
                    f"- **{r['date']}** ({r['event_type']}): {r['title']}"
                    for _, r in upcoming.head(5).iterrows()
                ]
                return "Upcoming calendar events:\n" + "\n".join(lines)
            return "No upcoming matching events found in `calendar.csv`."
        return "Calendar data is not available (`calendar.csv` missing or empty)."

    if any(k in q for k in ["score", "marks", "percentage", "grade", "cgpa", "gpa"]):
        if scores_df is not None and not scores_df.empty:
            sdf = scores_df.copy()
            sdf["total_marks"] = to_numeric_col(sdf["total_marks"])
            sdf["max_marks"] = to_numeric_col(sdf["max_marks"])
            if mentioned_sem:
                sdf = sdf[pd.to_numeric(sdf["semester"], errors="coerce") == mentioned_sem]
            sdf = sdf[sdf["max_marks"] > 0]
            if not sdf.empty:
                pct = (sdf["total_marks"].sum() / sdf["max_marks"].sum()) * 100
                scope = f"Semester {mentioned_sem}" if mentioned_sem else "all scored courses"
                return (
                    f"Simple average percentage for {scope} (prototype metric): **{pct:.2f}%** "
                    f"across {len(sdf)} courses."
                )
            return "No scored courses with `max_marks > 0` found."
        return "Score data is not available (`scores.csv` missing or empty)."

    if any(k in q for k in ["course", "subject", "syllabus", "topics"]):
        if courses_df is not None and not courses_df.empty:
            target = mentioned_sem or get_current_semester(sem_df)
            sub = (
                courses_df[pd.to_numeric(courses_df["semester"], errors="coerce") == target]
                if target
                else courses_df
            )
            if not sub.empty:
                lines = [
                    f"- **{r['course_code']} — {r['course_title']}** ({r['credits']} cr, {r['status']}): {r['syllabus_topics']}"
                    for _, r in sub.head(8).iterrows()
                ]
                header = f"Courses for Semester {target}:" if target else "Courses:"
                return header + "\n" + "\n".join(lines)
            return "No matching courses found in `courses.csv`."
        return "Course data is not available (`courses.csv` missing or empty)."

    if any(k in q for k in ["checklist", "plan", "todo", "to-do", "task"]):
        if plan_df is not None and not plan_df.empty:
            pending = plan_df[plan_df["status"].astype(str).str.strip() != "Completed"]
            if not pending.empty:
                lines = [
                    f"- [{r['status']}] **{r['item']}** (Sem {r['semester']}, target: {r['target_date']})"
                    for _, r in pending.head(5).iterrows()
                ]
                return "Open items from `plan_checklist.csv`:\n" + "\n".join(lines)
            return "All items in `plan_checklist.csv` are marked Completed."
        return "Checklist data is not available (`plan_checklist.csv` missing or empty)."

    return (
        "**AI API not connected.** Set `GEMINI_API_KEY` to enable generative answers for open-ended questions.\n\n"
        "Without the API, I can answer CSV-based questions such as:\n"
        "- *What is my current semester?*\n"
        "- *How many credits have I completed?*\n"
        "- *What are my upcoming assessments?*\n"
        "- *When is the next holiday or leave?*\n"
        "- *What is my score percentage in Semester 4?*\n"
        "- *Show courses and syllabus for Semester 5*\n"
        "- *What is on my checklist?*"
    )


def build_csv_context_summary() -> str:
    """Build a compact text context from loaded CSVs for Gemini."""
    parts = []
    for fname in CSV_SCHEMAS:
        df = load_csv(fname, warn_if_missing=False)
        if df is not None and not df.empty:
            parts.append(f"=== {fname} (first 30 rows) ===\n{df.head(30).to_csv(index=False)}")
    return "\n".join(parts)


def render_ask_ai():
    st.title("Ask Degree AI")
    st.caption("Ask questions about your courses, schedule, assessments, scores, and checklist.")

    api_key = get_gemini_api_key()
    if api_key:
        st.success("Gemini API connected (`GEMINI_API_KEY` detected).")
    else:
        st.info(
            "AI API not connected (`GEMINI_API_KEY` not set). "
            "Common questions will be answered from your local CSV data; "
            "questions needing AI generation will show **AI API not connected**."
        )

    if "ai_messages" not in st.session_state:
        st.session_state["ai_messages"] = []

    # Quick sample prompts
    st.caption("Quick questions:")
    q_cols = st.columns(4)
    quick_prompts = [
        "What is my current semester?",
        "What are my upcoming assessments?",
        "When is the next holiday or leave?",
        "How many credits have I completed?",
    ]
    clicked_prompt = None
    for idx, qp in enumerate(quick_prompts):
        if q_cols[idx].button(qp, use_container_width=True):
            clicked_prompt = qp

    for msg in st.session_state["ai_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about your degree, syllabus, scores, or plan...")
    prompt = clicked_prompt or user_input

    if prompt:
        st.session_state["ai_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if api_key:
            try:
                context = build_csv_context_summary()
                full_prompt = (
                    "You are a helpful academic assistant inside 'My Engineering Degree OS'. "
                    "Answer the student's question accurately using the CSV data below. "
                    "Do not invent university CGPA formulas.\n\n"
                    f"CSV DATA:\n{context}\n\n"
                    f"STUDENT QUESTION: {prompt}"
                )
                reply = call_gemini_rest(api_key, full_prompt)
            except urllib.error.URLError as exc:
                reply = f"Could not reach Gemini API ({exc}). Falling back to local CSV lookup:\n\n" + answer_from_csv_locally(prompt)
            except Exception as exc:
                reply = f"Gemini API error ({exc}). Falling back to local CSV lookup:\n\n" + answer_from_csv_locally(prompt)
        else:
            reply = answer_from_csv_locally(prompt)

        st.session_state["ai_messages"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


def render_data_admin():
    st.title("Data Admin")
    st.caption("Inspect, edit, and save any CSV dataset under the `/data` folder.")

    selected_file = st.selectbox("Select CSV File to Manage", list(CSV_SCHEMAS.keys()))
    file_path = DATA_DIR / selected_file

    df = load_csv(selected_file, warn_if_missing=False)
    if df is None:
        st.warning(f"`{file_path}` does not exist yet.")
        if st.button(f"Create Empty `{selected_file}` Template"):
            empty_df = pd.DataFrame(columns=CSV_SCHEMAS[selected_file])
            if save_csv(selected_file, empty_df):
                st.success(f"Created empty template at `{file_path}`.")
                st.rerun()
        return

    st.caption(f"Editing `{file_path}` — {len(df)} rows. You can add, edit, or delete rows below.")
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key=f"admin_editor_{selected_file}",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Save Changes to CSV", type="primary"):
            if save_csv(selected_file, edited_df):
                st.success(f"Saved `{selected_file}` successfully.")
                st.rerun()
    with col2:
        csv_bytes = edited_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download `{selected_file}`",
            data=csv_bytes,
            file_name=selected_file,
            mime="text/csv",
        )


# -----------------------------------------------------------------------------
# Main Navigation Router
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("🎓 My Engineering Degree OS")
    st.sidebar.caption("Semester 1–8 Academic Workspace")

    page = st.sidebar.radio("Navigation", NAV_ITEMS, index=0)

    st.sidebar.divider()
    sem_df = load_csv("semesters.csv", warn_if_missing=False)
    curr_sem = get_current_semester(sem_df)
    if curr_sem:
        st.sidebar.caption(f"Active: **Semester {curr_sem}**")
    st.sidebar.caption("Data folder: `/data/*.csv`")

    if page == "Dashboard":
        render_dashboard()
    elif page == "Semesters 1-8":
        render_semesters()
    elif page == "Courses & Syllabus":
        render_courses_syllabus()
    elif page == "University Syllabus":
        render_university_syllabus()
    elif page == "Academic / Leave Calendar":
        render_calendar()
    elif page == "Internal Assessments":
        render_assessments()
    elif page == "Status Tracker":
        render_status_tracker()
    elif page == "Scorecard":
        render_scorecard()
    elif page == "Plan & Checklist":
        render_plan_checklist()
    elif page == "Ask Degree AI":
        render_ask_ai()
    elif page == "Data Admin":
        render_data_admin()


if __name__ == "__main__":
    main()