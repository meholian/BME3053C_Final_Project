"""Streamlit symptom triage prototype for BME3053C Final Project."""
from __future__ import annotations

import json
import math
import re
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import pgeocode

DATA_PATH = Path(__file__).resolve().parent / "data" / "symptom_knowledge.json"
EMERGENCY_TOKENS = ("emergency", "emergent", "911", "call 911")

HOSPITAL_DIRECTORY: List[Dict[str, str | float]] = [
    {
        "name": "UF Health Shands Hospital",
        "city": "Gainesville, FL",
        "phone": "352-265-0111",
        "lat": 29.6516,
        "lon": -82.341,
    },
    {
        "name": "Mayo Clinic Hospital",
        "city": "Jacksonville, FL",
        "phone": "904-953-2000",
        "lat": 30.2666,
        "lon": -81.3899,
    },
    {
        "name": "Tampa General Hospital",
        "city": "Tampa, FL",
        "phone": "813-844-7000",
        "lat": 27.9392,
        "lon": -82.4571,
    },
    {
        "name": "AdventHealth Orlando",
        "city": "Orlando, FL",
        "phone": "407-303-5600",
        "lat": 28.5417,
        "lon": -81.3727,
    },
    {
        "name": "Cleveland Clinic Florida",
        "city": "Weston, FL",
        "phone": "954-659-5000",
        "lat": 26.1479,
        "lon": -80.3659,
    },
    {
        "name": "Emory University Hospital",
        "city": "Atlanta, GA",
        "phone": "404-712-2000",
        "lat": 33.7938,
        "lon": -84.3201,
    },
]

ZIP_COORDINATES: Dict[str, Tuple[float, float]] = {
    "32608": (29.6113, -82.3940),  # Gainesville, FL
    "32224": (30.2704, -81.4699),  # Jacksonville, FL
    "33606": (27.9360, -82.4560),  # Tampa, FL
    "32803": (28.5573, -81.3521),  # Orlando, FL
    "33331": (26.0635, -80.3663),  # Weston, FL
    "30322": (33.7982, -84.3220),  # Atlanta, GA
}

_ZIP_GEOCODER = pgeocode.Nominatim("us")


@st.cache_data(show_spinner=False)
def load_knowledge(base_path: str | Path = DATA_PATH) -> Dict:
    """Load symptom and diagnosis knowledge base from disk."""
    with Path(base_path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _feature_index(symptom_groups: Sequence[Dict]) -> Dict[str, int]:
    ids: Dict[str, int] = {}
    for group in symptom_groups:
        for item in group["symptoms"]:
            ids[item["id"]] = len(ids)
    return ids


def _synthesize_training_matrix(knowledge: Dict, feature_ids: Dict[str, int]) -> Tuple[pd.DataFrame, pd.Series]:
    """Create a lightweight synthetic dataset for the decision tree."""
    rng = np.random.default_rng(17)
    columns = list(feature_ids.keys()) + ["pain_score"]
    rows: List[List[float]] = []
    labels: List[str] = []

    for diag in knowledge["diagnoses"]:
        base_vector = np.zeros(len(columns), dtype=float)
        for symptom_id in diag["supporting_symptoms"]:
            if symptom_id in feature_ids:
                base_vector[feature_ids[symptom_id]] = 1.0
        pain_center = np.clip(diag.get("pain_weight", 0.5) * 10, 0, 10)
        for _ in range(48):
            sample = base_vector.copy()
            # Randomly drop or add symptoms to mimic noisy patient reports.
            for idx in range(len(feature_ids)):
                if sample[idx] == 1.0 and rng.random() < 0.18:
                    sample[idx] = 0.0
                elif sample[idx] == 0.0 and rng.random() < 0.04:
                    sample[idx] = 1.0
            sample[-1] = float(np.clip(rng.normal(pain_center, 1.6), 0, 10))
            rows.append(sample.tolist())
            labels.append(diag["name"])

    # Add nonspecific cases so the tree can choose the fallback node.
    for _ in range(160):
        sample = rng.binomial(1, 0.12, len(columns)).astype(float)
        sample[-1] = float(rng.uniform(0, 6))
        rows.append(sample.tolist())
        labels.append("General Check Needed")

    frame = pd.DataFrame(rows, columns=columns)
    label_series = pd.Series(labels, name="diagnosis")
    return frame, label_series


@st.cache_resource(show_spinner=False)
def build_model() -> Tuple[Dict, DecisionTreeClassifier, List[str]]:
    knowledge = load_knowledge()
    feature_ids = _feature_index(knowledge["symptom_groups"])
    X, y = _synthesize_training_matrix(knowledge, feature_ids)
    tree = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=10,
        random_state=11,
        class_weight="balanced",
    )
    tree.fit(X, y)
    return knowledge, tree, list(X.columns)


def _prediction_payload(
    selected_ids: Set[str],
    pain_score: int,
    model: DecisionTreeClassifier,
    feature_columns: Sequence[str],
) -> List[Tuple[str, float]]:
    vector = np.zeros(len(feature_columns), dtype=float)
    for idx, column in enumerate(feature_columns):
        if column == "pain_score":
            continue
        if column in selected_ids:
            vector[idx] = 1.0
    vector[-1] = float(pain_score)
    proba = model.predict_proba(vector.reshape(1, -1))[0]
    classes = model.classes_
    ranking = sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)
    return ranking


def _risk_level(probability: float, pain_score: int, pain_weight: float) -> str:
    score = probability
    score += (pain_score / 10) * (pain_weight * 0.4)
    if pain_score >= 8 and pain_weight >= 0.7:
        score += 0.1
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Moderate"
    return "Low"


def _format_symptom_hits(
    selected: Set[str],
    reference: Sequence[str],
    lookup: Dict[str, str],
    durations: Dict[str, int] | None = None,
) -> str:
    matches: List[str] = []
    for sid in reference:
        if sid in selected:
            label = lookup[sid]
            if durations and sid in durations:
                label = f"{label} ({durations[sid]} d)"
            matches.append(label)
    return ", ".join(matches) if matches else "None of the key symptoms selected"


def _render_sidebar(metadata: Dict[str, str]) -> None:
    st.sidebar.header("About This Prototype")
    st.sidebar.write(
        "This classroom demo uses a lightweight decision tree trained on curated symptom "
        "patterns derived from Mayo Clinic public summaries."
    )
    st.sidebar.info(metadata.get("disclaimer", "Educational use only."))
    st.sidebar.caption(f"Knowledge base source: {metadata.get('source', 'N/A')}")
    st.sidebar.markdown("---")
    st.sidebar.subheader("How to Use")
    st.sidebar.write(
        "Select symptoms within each organ system and rate overall pain. Submit to view "
        "probable diagnoses, their confidence, and suggested next steps."
    )


def _advice_requires_emergency(advice: str | None) -> bool:
    if not advice:
        return False
    text = advice.lower()
    return any(token in text for token in EMERGENCY_TOKENS)


def _duration_weight(diag_meta: Dict, durations: Dict[str, int]) -> float:
    if not durations:
        return 1.0
    windowed_days: List[int] = []
    for sym in diag_meta.get("supporting_symptoms", []):
        if sym in durations:
            windowed_days.append(min(durations[sym], 60))
    if not windowed_days:
        return 1.0
    avg_days = sum(windowed_days) / len(windowed_days)
    boost = min(avg_days / 30.0, 0.4)
    return 1.0 + boost

def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_miles * c


def _nearest_hospitals(lat: float, lon: float, limit: int = 3) -> List[Dict[str, str]]:
    distances: List[Tuple[float, Dict[str, str]]] = []
    for facility in HOSPITAL_DIRECTORY:
        facility_lat = float(facility["lat"])
        facility_lon = float(facility["lon"])
        distance = _haversine_distance(lat, lon, facility_lat, facility_lon)
        distances.append(
            (
                distance,
                {
                    "Hospital": facility["name"],
                    "City": facility["city"],
                    "Phone": facility["phone"],
                    "Distance (mi)": f"{distance:.1f}",
                },
            )
        )
    distances.sort(key=lambda item: item[0])
    return [entry for _, entry in distances[:limit]]


@lru_cache(maxsize=256)
def _geocode_zip_online(zip_code: str) -> Tuple[float, float] | None:
    try:
        result = _ZIP_GEOCODER.query_postal_code(zip_code)
    except Exception:
        return None
    if result is None:
        return None
    latitude = result.latitude
    longitude = result.longitude
    if latitude is None or longitude is None or pd.isna(latitude) or pd.isna(longitude):
        return None
    return float(latitude), float(longitude)


def _coords_for_zip(zip_code: str) -> Tuple[float, float] | None:
    if not zip_code:
        return None
    normalized = zip_code.strip()
    if not re.fullmatch(r"\d{5}", normalized):
        return None
    if normalized in ZIP_COORDINATES:
        return ZIP_COORDINATES[normalized]
    return _geocode_zip_online(normalized)

def main() -> None:
    st.set_page_config(page_title="Biomedical Symptom Triage", layout="wide")
    knowledge, model, feature_columns = build_model()
    _render_sidebar(knowledge.get("metadata", {}))

    st.title("Biomedical Symptom Decision Support")
    st.caption(
        "Prototype triage aid for BME3053C. Outputs are informational and do not replace "
        "professional evaluation."
    )

    system_groups = knowledge["symptom_groups"]
    symptom_label_lookup = {
        symptom["id"]: symptom["label"]
        for group in system_groups
        for symptom in group["symptoms"]
    }

    # Dynamic symptom selection with instant durations and an Analyze button
    st.subheader("Organ-System Symptom Checklist")

    # Track selections in session_state so checkboxes update instantly
    if "selected_ids" not in st.session_state:
        st.session_state["selected_ids"] = set()
    if "analysis_ready" not in st.session_state:
        st.session_state["analysis_ready"] = False

    selected: Set[str] = set(st.session_state["selected_ids"])

    for group in system_groups:
        with st.expander(group["system"], expanded=False):
            cols = st.columns(2)
            for idx, symptom in enumerate(group["symptoms"]):
                cb_key = f"cb_{group['system']}_{symptom['id']}"
                initial_val = symptom["id"] in selected
                checked = cols[idx % 2].checkbox(symptom["label"], key=cb_key, value=initial_val)
                if checked:
                    selected.add(symptom["id"])
                else:
                    selected.discard(symptom["id"])

    # Persist current selection for subsequent rerenders
    st.session_state["selected_ids"] = set(selected)

    # Duration inputs appear immediately for selected symptoms (same appearance)
    st.subheader("Symptom Durations (days experienced)")
    symptom_durations: Dict[str, int] = {}
    if selected:
        duration_columns = st.columns(min(3, len(selected)))
        for idx, symptom_id in enumerate(sorted(selected)):
            label = symptom_label_lookup.get(symptom_id, symptom_id)
            column = duration_columns[idx % len(duration_columns)]
            duration_key = f"dur_{symptom_id}"
            if duration_key not in st.session_state:
                st.session_state[duration_key] = 2
            duration = column.number_input(
                f"{label}",
                min_value=0,
                max_value=365,
                value=st.session_state[duration_key],
                step=1,
                key=duration_key,
            )
            symptom_durations[symptom_id] = int(duration)
    else:
        st.caption("Select symptoms above to log how long they have been present.")

    # Pain slider remains outside a form to update instantly
    pain_score = st.slider(
        "Overall pain intensity (0 = none, 10 = extreme)",
        min_value=0,
        max_value=10,
        value=4,
    )

    # Analyze button to trigger computation (replaces form_submit_button)
    submitted = st.button("Analyze Symptom Profile", type="primary")

    if submitted:
        st.session_state["analysis_ready"] = True

    if not st.session_state["analysis_ready"]:
        st.info("Use the controls above to select symptoms and press Analyze.")
        st.stop()

    if not selected and pain_score == 0:
        st.session_state["analysis_ready"] = False
        st.warning("Select at least one symptom or report a non-zero pain score to proceed.")
        return

    ranking = _prediction_payload(selected, pain_score, model, feature_columns)
    diagnosis_lookup = {diag["name"]: diag for diag in knowledge["diagnoses"]}

    # Re-weight predictions to account for symptom duration emphasis.
    adjusted_entries: List[Tuple[str, float]] = []
    total_weighted = 0.0
    for name, probability in ranking:
        diag_meta = diagnosis_lookup.get(name, {})
        factor = _duration_weight(diag_meta, symptom_durations)
        weighted_prob = probability * factor
        adjusted_entries.append((name, weighted_prob))
        total_weighted += weighted_prob
    if total_weighted > 0:
        ranking = sorted(
            [(name, prob / total_weighted) for name, prob in adjusted_entries],
            key=lambda item: item[1],
            reverse=True,
        )

    st.subheader("Model Output")
    top_entries = ranking[:3]
    table_rows = []
    emergency_flag = False
    for name, probability in top_entries:
        diag_meta = diagnosis_lookup.get(name, {})
        pain_weight = float(diag_meta.get("pain_weight", 0.5))
        risk_bucket = _risk_level(probability, pain_score, pain_weight)
        advice_text = diag_meta.get("advice", "Follow up with a clinician.")
        if _advice_requires_emergency(advice_text):
            emergency_flag = True
        table_rows.append(
            {
                "Diagnosis": name,
                "Confidence": f"{probability:.0%}",
                "Risk Level": risk_bucket,
                "Matched Symptoms": _format_symptom_hits(
                    selected,
                    diag_meta.get("supporting_symptoms", []),
                    symptom_label_lookup,
                    symptom_durations,
                ),
                "Suggested Action": advice_text,
            }
        )

    results_df = pd.DataFrame(table_rows)
    for column, width in ("Suggested Action", 60), ("Matched Symptoms", 46):
        if column in results_df.columns:
            results_df[column] = results_df[column].map(
                lambda value, wrap_width=width: textwrap.fill(value, width=wrap_width)
                if isinstance(value, str)
                else value
            )

    st.table(results_df)

    should_show_locator = emergency_flag
    if should_show_locator:
        st.markdown("---")
        st.subheader("Emergency Resources")
        st.info(
            "Emergency-level guidance detected. Use the locator below to find nearby hospitals."
        )
        with st.expander("Locate Nearby Hospitals", expanded=False):
            st.caption(
                "Enter your ZIP code to approximate the closest hospitals in our reference list."
            )
            zip_code = st.text_input(
                "ZIP code",
                value="32608",
                max_chars=5,
                help="Enter any US ZIP code; we'll approximate the nearest hospitals from the reference list.",
                key="hospital_locator_zip_copy",
            )
            if st.button("Find Hospitals", key="hospital_locator_btn_copy"):
                coords = _coords_for_zip(zip_code)
                if not coords:
                    st.error(
                        "Unable to locate that ZIP code. Confirm the 5-digit code or try a nearby ZIP before using map apps for exact directions."
                    )
                else:
                    nearest = _nearest_hospitals(coords[0], coords[1], limit=3)
                    st.table(pd.DataFrame(nearest))
                    st.caption(
                        "Distances are approximate great-circle estimates based on ZIP centroids. Call ahead to confirm availability."
                    )

    st.markdown("---")
    st.subheader("Why these diagnoses?")
    for name, probability in top_entries:
        diag_meta = diagnosis_lookup.get(name, {})
        matched = _format_symptom_hits(
            selected,
            diag_meta.get("supporting_symptoms", []),
            symptom_label_lookup,
            symptom_durations,
        )
        st.markdown(
            f"**{name}** — model confidence {probability:.1%}.\n"
            f"- Key systems: {', '.join(diag_meta.get('systems', [])) or 'n/a'}\n"
            f"- Matched hallmark symptoms: {matched}\n"
            f"- Pain sensitivity weighting: {diag_meta.get('pain_weight', 0.5):.1f}\n"
            f"- Guidance: {diag_meta.get('advice', 'Consult a licensed clinician.')}"
        )

    if symptom_durations:
        st.markdown("---")
        st.subheader("Symptom Duration Log")
        duration_rows = [
            {
                "Symptom": symptom_label_lookup[sid],
                "Days": days,
            }
            for sid, days in symptom_durations.items()
            if sid in selected
        ]
        if duration_rows:
            st.table(pd.DataFrame(duration_rows))

    st.markdown("---")
    if emergency_flag:
        st.markdown(
            "**Emergency Notice:** One or more suggested actions recommend emergency or emergent "
            "evaluation. Seek in-person care or call local emergency services without delay."
        )
        st.markdown("---")
    st.caption(
        "This decision support tool is not FDA cleared and should not guide emergency care. "
        "Dial emergency services for life-threatening symptoms."
    )


if __name__ == "__main__":
    main()