"""
ClinIQ — Streamlit Frontend
Full-featured clinical intelligence dashboard with:
  • Risk gauge (Plotly)
  • Per-category risk breakdown bar chart
  • Drug interaction cards
  • Tabbed analysis results
  • Sample cases for demo
  • Knowledge base management
  • Download report as JSON
"""
from __future__ import annotations
import json
import os
import time
import uuid
from typing import Any, Dict, Optional

import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinIQ — Clinical Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL","https://cliniq-yjib.onrender.com")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0d1626; }
  .risk-badge-CRITICAL { background:#dc2626;color:#fff;padding:3px 10px;border-radius:12px;font-weight:700;font-size:13px; }
  .risk-badge-HIGH     { background:#ea580c;color:#fff;padding:3px 10px;border-radius:12px;font-weight:700;font-size:13px; }
  .risk-badge-MODERATE { background:#d97706;color:#000;padding:3px 10px;border-radius:12px;font-weight:700;font-size:13px; }
  .risk-badge-LOW      { background:#16a34a;color:#fff;padding:3px 10px;border-radius:12px;font-weight:700;font-size:13px; }
  .risk-badge-UNKNOWN  { background:#6b7280;color:#fff;padding:3px 10px;border-radius:12px;font-weight:700;font-size:13px; }
  .interaction-severe   { border-left:4px solid #dc2626; }
  .interaction-moderate { border-left:4px solid #ea580c; }
  .interaction-mild     { border-left:4px solid #d97706; }
  .card { background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px;margin:6px 0; }
  .step-ok   { color:#22c55e;font-family:monospace;font-size:13px;margin:2px 0; }
  .step-warn { color:#f59e0b;font-family:monospace;font-size:13px;margin:2px 0; }
  .step-info { color:#60a5fa;font-family:monospace;font-size:13px;margin:2px 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

RISK_COLORS = {
    "LOW": "#16a34a",
    "MODERATE": "#d97706",
    "HIGH": "#ea580c",
    "CRITICAL": "#dc2626",
    "UNKNOWN": "#6b7280",
}

RISK_VALUES = {"LOW": 15, "MODERATE": 45, "HIGH": 72, "CRITICAL": 92, "UNKNOWN": 50}


def risk_badge(level: str) -> str:
    return f"<span class='risk-badge-{level}'>{level}</span>"


def gauge_chart(level: str) -> go.Figure:
    val = RISK_VALUES.get(level, 50)
    color = RISK_COLORS.get(level, "#6b7280")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Overall Risk", "font": {"size": 17, "color": "#94a3b8"}},
        number={"suffix": f"  {level}", "font": {"size": 22, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickwidth": 1, "tickfont": {"color": "#94a3b8"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#1e293b",
            "borderwidth": 1,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 30],  "color": "#14532d"},
                {"range": [30, 60], "color": "#713f12"},
                {"range": [60, 80], "color": "#7c2d12"},
                {"range": [80, 100], "color": "#450a0a"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=230,
        margin=dict(t=40, b=5, l=20, r=20),
        font={"color": "#94a3b8"},
    )
    return fig


def risk_bar_chart(risk_scores: Dict) -> Optional[go.Figure]:
    if not risk_scores:
        return None
    cats, vals, colors = [], [], []
    for cat, data in risk_scores.items():
        if not isinstance(data, dict):
            continue
        score = data.get("score", 0) * 100
        level = data.get("level", "UNKNOWN")
        cats.append(cat.replace("_", " ").title())
        vals.append(score)
        colors.append(RISK_COLORS.get(level, "#6b7280"))

    fig = go.Figure(go.Bar(
        x=vals, y=cats, orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in vals],
        textposition="outside",
        textfont={"color": "#cbd5e1", "size": 12},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,26,47,0.9)",
        height=max(180, len(cats) * 48),
        margin=dict(t=5, b=5, l=10, r=60),
        xaxis={"range": [0, 120], "gridcolor": "#1e3a5f", "tickfont": {"color": "#94a3b8"}, "title": {"text": "Risk Score (%)", "font": {"color": "#94a3b8"}}},
        yaxis={"gridcolor": "#1e3a5f", "tickfont": {"color": "#e2e8f0"}},
    )
    return fig


def api_post(endpoint: str, payload: Dict) -> Optional[Dict]:
    try:
        resp = requests.post(f"{API_BASE_URL}/{endpoint}", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("❌ Cannot reach the backend API. Start it with: `make run-api`")
    except requests.Timeout:
        st.error("⏱ Request timed out (180s). The LLM may be slow — try again.")
    except Exception as exc:
        st.error(f"API error: {exc}")
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 ClinIQ")
    st.markdown("**Multi-Agent Clinical Intelligence**")
    st.divider()

    # Health check
    try:
        h = requests.get(f"{API_BASE_URL}/health", timeout=3).json()
        st.success("✅ API Online")
        c1, c2 = st.columns(2)
        c1.metric("KB Docs", h.get("knowledge_base_docs", 0))
        c2.metric("Provider", h.get("llm_provider", "?").upper())
        st.caption(f"Domain: {h.get('domain','?').upper()}")
    except Exception:
        st.error("❌ API Offline")

    st.divider()
    st.markdown("### 🏗️ Pipeline")
    st.markdown("""
```
📋 Intake Agent
    ↓
📚 Knowledge Agent
   (Hybrid RAG)
    ↓
⚠️ Risk Agent
   (Rules + LLM)
    ↓
📊 Report Agent
    ↓
  ✅ Done
```
    """)
    st.divider()
    st.markdown("### ⚙️ Stack")
    st.caption("LangGraph · Groq · ChromaDB")
    st.caption("BM25+Semantic RAG · Langfuse")
    st.caption("FastAPI · Streamlit · Docker")


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown("# 🏥 ClinIQ — Clinical Intelligence Platform")
st.markdown("Drop in any clinical document. Get AI-powered risk analysis, drug checks, and a full report.")
st.divider()

tab_analysis, tab_kb, tab_about = st.tabs(["🔬 Analysis", "📚 Knowledge Base", "ℹ️ About"])

# ╔══════════════════════════════════════════════════════╗
# ║                 TAB 1 — ANALYSIS                     ║
# ╚══════════════════════════════════════════════════════╝
with tab_analysis:
    left, right = st.columns([1.05, 0.95], gap="large")

    # ── Input column ──────────────────────────────────────────────────────────
    with left:
        st.markdown("### 📋 Patient Document")
        method = st.radio("Input method:", ["📝 Paste Text", "📤 Upload File", "🔬 Sample Case"], horizontal=True, label_visibility="collapsed")
        doc_text = ""

        # Paste
        if method == "📝 Paste Text":
            doc_text = st.text_area(
                "Paste clinical document:",
                height=290,
                placeholder="Paste any clinical document:\n• Discharge summary\n• Lab report\n• Clinical note\n• Prescription\n• Outpatient visit note",
            )

        # Upload
        elif method == "📤 Upload File":
            uploaded = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
            if uploaded:
                try:
                    r = requests.post(
                        f"{API_BASE_URL}/upload-document",
                        files={"file": (uploaded.name, uploaded, uploaded.type)},
                        timeout=30,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        doc_text = data["extracted_text"]
                        st.success(f"✅ Extracted {data['char_count']:,} chars / {data['word_count']:,} words")
                        st.text_area("Preview:", value=doc_text[:500] + ("..." if len(doc_text) > 500 else ""), height=120, disabled=True)
                    else:
                        st.error(f"Upload failed: {r.text}")
                except Exception as exc:
                    st.error(f"Upload error: {exc}")

        # Sample cases
        else:
            sample = st.selectbox("Select sample case:", [
                "Complex HF + DM + CKD (High Risk)",
                "Septic Shock — ICU (Critical)",
                "Post-Op Orthopedic (Moderate Risk)",
                "Hypertension + CKD Progression (High Risk)",
            ])
            SAMPLES = {
                "Complex HF + DM + CKD (High Risk)": """DISCHARGE SUMMARY — INPATIENT

Patient: John Smith | DOB: 15-Mar-1957 | MRN: JS-78234 | Blood Type: A+
Admission: 10-Dec-2024 | Discharge: 15-Dec-2024 | LOS: 5 days

ADMITTING PHYSICIAN: Dr. Sarah Johnson, MD (Internal Medicine)
ATTENDING: Dr. Michael Chen, MD (Cardiology Consult)

CHIEF COMPLAINT:
Progressive dyspnea at rest, bilateral ankle swelling, orthopnea (3-pillow) — 2 weeks duration.

HISTORY OF PRESENT ILLNESS:
Mr. Smith is a 67-year-old male with longstanding Type 2 Diabetes Mellitus (HbA1c 9.2% three months prior), 
poorly controlled Hypertension, and Chronic Kidney Disease Stage 3b presenting with a 2-week history of 
worsening shortness of breath, inability to lie flat, bilateral lower extremity pitting oedema to the knee, 
and a 7 kg weight gain over 2 weeks. He reports 3-pillow orthopnea and two episodes of PND this week.
Noted reduction in urinary output over 3 days. No fever, no cough, no chest pain.

PAST MEDICAL HISTORY:
1. Type 2 Diabetes Mellitus — Duration 12 years, poor glycaemic control
2. Hypertension — Poorly controlled, 15 years
3. Chronic Kidney Disease Stage 3b (eGFR 32 mL/min baseline — 6 months ago)
4. Atrial Fibrillation — On anticoagulation
5. Hyperlipidaemia
6. Osteoarthritis — Lower back (takes ibuprofen regularly)

CURRENT MEDICATIONS (Home):
1. Metformin 1000 mg twice daily (with meals)
2. Lisinopril 20 mg once daily
3. Warfarin 5 mg once daily (INR target 2.0-3.0 for AF)
4. Ibuprofen 400 mg three times daily (PRN back pain — takes most days)
5. Atorvastatin 40 mg once nightly
6. Furosemide 40 mg once daily
7. Carvedilol 12.5 mg twice daily
8. Aspirin 81 mg once daily

ALLERGIES: Penicillin (rash)

VITAL SIGNS ON ADMISSION:
BP: 168/96 mmHg (sitting) | HR: 102 bpm (irregularly irregular) | RR: 24 breaths/min
O2 Saturation: 90% on room air | Temperature: 37.1°C | Weight: 96.2 kg (dry weight ~89 kg)
BMI: 31.2 | JVP: 5 cm above sternal angle

PHYSICAL EXAMINATION:
Respiratory: Bibasal crepitations to mid-zones, dullness at right base (pleural effusion)
Cardiovascular: Irregular rhythm, S3 gallop, no murmurs
Extremities: 3+ pitting oedema bilateral legs to knees
Abdomen: Mildly distended, hepatomegaly

LABORATORY RESULTS:
CBC:
  Haemoglobin: 10.2 g/dL (baseline ~11.5)
  WBC: 8,900 /μL | Neutrophils: 75%
  Platelets: 138,000 /μL

METABOLIC:
  Sodium: 130 mEq/L | Potassium: 5.3 mEq/L | Chloride: 97 mEq/L
  Bicarbonate: 22 mEq/L | BUN: 52 mg/dL | Creatinine: 3.1 mg/dL (baseline 2.0)
  eGFR: 19 mL/min (calculated — CKD-EPI) | Glucose: 294 mg/dL | HbA1c: 9.8%
  Albumin: 3.1 g/dL | Total Protein: 5.9 g/dL

LIVER FUNCTION: ALT 31 | AST 38 | Total Bili 1.2 | ALP 88 — all normal

CARDIAC:
  BNP: 1,847 pg/mL | Troponin I: 0.06 ng/mL (mildly elevated — repeat in 6h: 0.07 — stable)
  INR: 4.1 | PT: 38.2 sec

LIPIDS (3 months ago): Total cholesterol 198 | LDL 112 | HDL 38 | TG 210

URINALYSIS: 2+ protein, no leucocytes, no nitrites. Urine ACR: 520 mg/g

IMAGING:
Chest X-Ray: Cardiomegaly, bilateral pulmonary oedema, right pleural effusion
ECG: Atrial fibrillation, rate 102, LBBB (new vs. 2 years ago — widened QRS 148ms)
Echocardiogram (Day 2): LVEF 28% (severely reduced), LV dilation, Grade 3 diastolic dysfunction, 
  Moderate functional mitral regurgitation. RV mildly dilated. IVC dilated, no collapse.

HOSPITAL COURSE:
Day 1: IV furosemide 80mg bolus → continuous infusion 10mg/hr. Warfarin held (INR 4.1). 
  Ibuprofen discontinued immediately. Metformin held (eGFR<30).
Day 2: Diuresis 3.2L negative. Echo performed. Cardiology consult.
Day 3: eGFR improved to 24. BNP 1,120. Weight 93.1 kg. Continued diuresis.
Day 4: O2 off. RR normalised. eGFR 28. INR 2.6 (warfarin restarted 3mg).
Day 5: Discharge ready. Weight 89.8 kg. O2 sat 96% RA. Furosemide PO resumed 80mg.

DISCHARGE DIAGNOSES:
1. Acute Decompensated Heart Failure (HFrEF, EF 28%) — Stage C, NYHA Class IV (adm), Class II (disch)
2. Type 2 Diabetes Mellitus — Poorly Controlled (HbA1c 9.8%)
3. Acute Kidney Injury Stage 2 (Cr 3.1, baseline 2.0) superimposed on CKD Stage 4
4. Atrial Fibrillation with RVR — Supratherapeutic anticoagulation (INR 4.1)
5. Hyponatremia (Na 130) — Dilutional/HF-related
6. Anaemia of Chronic Disease (Hgb 10.2)
7. New LBBB on background AF — Possible CRT candidacy
8. NSAID-induced renal toxicity (Ibuprofen)

DISCHARGE MEDICATIONS:
1. Furosemide 80mg PO daily (increased from 40mg)
2. Carvedilol 12.5mg twice daily (continue, reassess)
3. Warfarin 3mg daily (INR check in 5 days — target 2-3)
4. Atorvastatin 40mg nightly
5. Lisinopril 10mg daily (reduced from 20mg — AKI recovery)
6. SGLT2 inhibitor — dapagliflozin 10mg to START once eGFR >20 (check in 2 weeks)
7. INSULIN: Glargine 10 units SC nightly (metformin PERMANENTLY STOPPED — eGFR)
8. Aspirin 81mg daily

MEDICATIONS STOPPED: Metformin (contraindicated — eGFR<30), Ibuprofen (contraindicated — renal/cardiac)

FOLLOW-UP:
- Cardiology: 1 week | Primary Care: 2 weeks | Nephrology: 3 weeks (CKD stage 4)
- INR check: 5 days | BMP: 1 week | CRT evaluation: as outpatient

Attending: Dr. Sarah Johnson, MD | Date: 15-Dec-2024 | Signature: [Electronic]""",

                "Septic Shock — ICU (Critical)": """EMERGENCY DEPARTMENT / ICU TRANSFER NOTE

Patient: Maria Garcia | DOB: 12-Jun-1966 | MRN: MG-55129 | Wt: 68kg
Date/Time: 15-Dec-2024 / 14:35
Transferred FROM: ED to Medical ICU

CHIEF COMPLAINT: Fever, confusion, hypotension — community onset

TRIAGE VITALS:
Temp: 39.9°C | HR: 126 bpm (sinus tachycardia) | BP: 78/46 mmHg
RR: 28 breaths/min | O2 Sat: 91% (4L NC) → 94%
GCS: 12/15 (E3 V4 M5) | Weight: 68 kg

ED HISTORY:
Mrs. Garcia is a 58-year-old female with Type 2 DM and COPD (GOLD 2) who arrived by EMS 
from a nursing home with a 36-hour history of worsening confusion, fever, rigors, and reduced 
urine output. Nursing home staff noted she was unable to stand this morning.

PAST MEDICAL HISTORY:
1. Type 2 Diabetes Mellitus — poorly controlled
2. COPD GOLD Stage 2 (FEV1 65% predicted)
3. Recurrent UTIs — 3 episodes this year, last treated with TMP-SMX 6 months ago
4. Hypertension
5. CKD Stage 2 (baseline Cr 1.1)

HOME MEDICATIONS:
- Lisinopril 10mg daily
- Metformin 500mg twice daily
- Tiotropium 18mcg inhaler once daily
- Albuterol MDI 2 puffs QID PRN
- No recent antibiotics documented

ALLERGIES: Sulfa (TMP-SMX) — hives/rash

LABORATORY (ED):
CBC: WBC 28,900/μL (Bands 22%, Neutrophils 69%), Hgb 11.2, Plt 89,000
METABOLIC: Na 134 | K 4.8 | Cr 3.4 (baseline 1.1) | BUN 78 | Glucose 389 | HCO3 14 | AG 22
COAGULATION: INR 1.8 | PT 21 | aPTT 44 | Fibrinogen 480
SEPSIS MARKERS: Lactate 5.1 mmol/L | Procalcitonin 68 ng/mL | CRP 298 mg/L
CULTURES: Blood cultures x2 (drawn before abx) | Urine culture STAT | Sputum pending
URINALYSIS: WBC >100/hpf, RBC 15, Bacteria 4+, Nitrites POSITIVE, Leukocyte esterase 3+

ABG (on 4L NC): pH 7.21 | PaCO2 28 | PaO2 72 | HCO3 14 (metabolic acidosis + resp compensation)

IMAGING:
CXR: No focal consolidation. No pneumothorax.
CT Abdomen (ordered, pending)

ASSESSMENT AND MANAGEMENT IN ED:
Primary diagnosis: SEPTIC SHOCK secondary to UROSEPSIS (E. coli likely given nitrite positive)
Secondary: AKI Stage 3 (Cr 3.4 on baseline 1.1), Severe metabolic acidosis, Hyperglycaemia

ED MANAGEMENT:
14:40 — Blood cultures x2 collected
14:42 — IV Piperacillin-Tazobactam 4.5g IV over 30 min (avoiding TMP-SMX — allergy; Cipro — resistance concern)
14:48 — IV fluid resuscitation: 2,040 mL NS over 45 min (30 mL/kg) ✅
15:05 — Norepinephrine via peripheral IV (central access being placed): started 0.08 mcg/kg/min
         MAP current: 58 mmHg — target ≥65
15:10 — Foley catheter placed: UO 18 mL in first hour (0.26 mL/kg/h — oliguria)
15:15 — Insulin infusion started: Glucose 389 → target 140-180 mg/dL
15:20 — Metformin HELD immediately (AKI + contrast possible)
15:30 — MICU bed confirmed. Transfer in progress.

CURRENT STATUS AT TRANSFER:
BP: 92/58 (MAP 69) on norepinephrine 0.10 mcg/kg/min | HR: 118 | RR: 22 | SpO2: 95% on 6L NRB | Temp: 39.2°C
Lactate re-check (60 min): 4.2 mmol/L (trending down from 5.1)
GCS: 13/15 (improving with resuscitation)

ICU PLAN:
- Norepinephrine titrate to MAP ≥65 (add vasopressin if >0.25 mcg/kg/min)
- Targeted fluid resuscitation — avoid over-resuscitation (FACTT trial)
- Consider hydrocortisone 200mg/day if refractory shock (vasopressor-dependent >24h)
- Serial lactate Q4h until normalised (<2 mmol/L)
- CT abdomen/pelvis to r/o abscess when haemodynamically stable
- Renal: AKI management — hold nephrotoxins, fluid optimisation; dialysis if anuria/acidosis worsens
- Repeat cultures: 48-72h de-escalation based on sensitivities
- Glycaemic control: Insulin infusion protocol

ICU Attending: Dr. Ahmad Raza, MD (Critical Care) | Bedside RN: S. Thomas RN
Date: 15-Dec-2024 | Transfer time: 15:45""",

                "Post-Op Orthopedic (Moderate Risk)": """POST-OPERATIVE DAY 2 — CLINICAL NOTE

Patient: Robert Chen | DOB: 22-Sep-1951 | MRN: RC-33012 | Age: 72M
Ward: Orthopaedic 4B | Bed 12 | Date: 14-Dec-2024

PROCEDURE: Right Total Hip Arthroplasty (THA)
Surgery date: 12-Dec-2024 | Surgeon: Mr. David Williams, FRCS
Anaesthesia: Spinal (isobaric bupivacaine) + sedation | Duration: 2 hrs 10 min | EBL: 350 mL

PAST MEDICAL HISTORY:
1. Hypertension (well-controlled on 2 agents)
2. Hyperlipidaemia
3. Osteoarthritis — bilateral hips (R>L), both knees
4. Type 2 DM — well-controlled (last HbA1c 6.9%)
5. No history of VTE, bleeding disorder, cardiac disease

CURRENT MEDICATIONS (post-op):
1. Enoxaparin 40 mg SQ once daily (VTE prophylaxis — day 1 post-op)
2. Ketorolac 15 mg IV Q6H (scheduled x 3 days — NSAID for pain)
3. Oxycodone 5 mg PO Q4H PRN (breakthrough)
4. Paracetamol 1g PO Q6H (scheduled)
5. Lisinopril 10 mg PO daily (HELD post-op — resume when PO stable)
6. Amlodipine 5 mg PO daily (CONTINUED)
7. Atorvastatin 20 mg PO nightly
8. Metformin 500 mg PO twice daily (HELD peri-op — resume after 48h if Cr stable)
9. Ondansetron 4 mg IV PRN nausea

ALLERGIES: Codeine (nausea/vomiting)

VITAL SIGNS TODAY:
BP: 142/88 mmHg | HR: 78 bpm (regular) | RR: 16 | Temp: 37.9°C (low-grade — post-op)
O2 Sat: 97% on RA | Pain score: 4/10 (improved from 7 yesterday) | Urine output: adequate

POST-OP DAY 2 LABS:
Haemoglobin: 9.1 g/dL (pre-op: 13.8 g/dL — expected post-THA drop)
WBC: 13,400 /μL (post-operative reactive leukocytosis, expected)
Platelets: 178,000 /μL (pre-op 245,000)
Sodium: 138 mEq/L | Potassium: 3.1 mEq/L | Creatinine: 1.4 mg/dL (baseline 1.0)
BUN: 22 mg/dL | Glucose: 128 mg/dL | eGFR: 52 mL/min

COAGULATION: PT/INR 1.1 (no anticoagulation — enoxaparin only)
Iron studies: Serum ferritin 28 ng/mL (low) | TIBC elevated — iron deficiency anaemia

CLINICAL ASSESSMENT:
1. Post-operative anaemia: Expected after THA. Hgb 9.1 — monitor; iron supplementation initiated (IV iron sucrose 200mg given).
   No indication for transfusion at this level (haemodynamically stable, asymptomatic).
2. Acute Kidney Injury (Stage 1): Cr 1.4 vs baseline 1.0. Likely multifactorial — NSAIDs (ketorolac), 
   perioperative hypotension, reduced oral intake. Lisinopril held appropriately. Hydration optimised.
   Consider NSAID dose reduction or switch to paracetamol-only regimen.
3. Hypokalemia (K 3.1): Oral KCl supplementation 40 mEq daily x 3 days. Recheck tomorrow.
4. Low-grade fever: Post-operative days 1-3 — expected (tissue inflammation + atelectasis).
   No source identified. Wound clean and dry. Continue monitoring.
5. VTE Prophylaxis: Enoxaparin continuing. Note: enoxaparin + ketorolac combination — monitor for bleeding.
   Compression stockings + early mobilisation.

PHYSIOTHERAPY: PT visit today — patient stood with zimmer frame. Mobilising with supervision.
  Target: Independent ambulation by POD 4.

PLAN:
- Reduce ketorolac to 10mg Q8H (renal precaution — AKI)
- Continue enoxaparin (but monitor HFT given ketorolac combination)
- Recheck BMP and Hgb tomorrow
- Resume metformin once Cr returns to ≤1.2 and patient fully PO
- Resume lisinopril at 50% dose (5mg) when PO stable — hold if Cr worsens
- Oral potassium supplementation — recheck K tomorrow
- Target discharge POD 4-5 with community physio arranged

Signed: Dr. Priya Mehta, MBBS (Junior Registrar — Orthopaedics)
Reviewed by: Mr. David Williams, FRCS | Date: 14-Dec-2024 09:15""",

                "Hypertension + CKD Progression (High Risk)": """NEPHROLOGY OUTPATIENT CLINIC NOTE

Patient: Linda Thompson | DOB: 08-Mar-1959 | MRN: LT-88823 | Age: 65F
Clinic: Renal and Hypertension Clinic | Date: 15-Dec-2024
Nephrologist: Dr. Fatima Al-Rashid, MD, PhD (Nephrology)

REASON FOR VISIT: 6-monthly CKD review — worsening renal function and uncontrolled HTN

HISTORY:
Mrs. Thompson is a 65-year-old woman with a 15-year history of Type 2 Diabetes, 
Hypertension, and progressive Chronic Kidney Disease. She was seen 6 months ago with eGFR 38 mL/min.
Today she reports increased fatigue, leg swelling, and poor blood pressure control despite medication.
She admits to dietary non-compliance (high-sodium, high-potassium foods) and missed medications.

PAST MEDICAL HISTORY:
1. Type 2 Diabetes Mellitus — Duration 18 years
2. Hypertension — Duration 15 years, difficult to control
3. CKD — Progressive diabetic nephropathy (DKD)
4. Anaemia of CKD — On ESA therapy
5. Secondary hyperparathyroidism — CKD-MBD
6. Peripheral neuropathy (diabetic)

CURRENT MEDICATIONS:
1. Lisinopril 40 mg once daily (maximum dose)
2. Losartan 100 mg once daily (added 4 months ago — dual RAAS blockade)
3. Hydrochlorothiazide 25 mg once daily
4. Metformin 500 mg twice daily (patient self-continuing despite CKD)
5. Insulin Glargine 24 units SC nightly
6. Insulin Aspart: Correction scale with meals
7. Calcium Carbonate 1g three times daily with meals (phosphate binder)
8. Calcitriol 0.25 mcg once daily (active vitamin D)
9. Epoetin alfa 4000 IU SQ weekly (anaemia management)
10. Atorvastatin 20 mg nightly
11. Aspirin 75 mg once daily

ALLERGIES: NSAIDs (prescribed avoidance — CKD)

VITAL SIGNS:
BP: 162/98 mmHg (average of 3 readings; home BP log average 160/96 past 2 weeks)
HR: 74 bpm (regular) | Weight: 79 kg (was 75 kg 6 months ago — 4 kg gain) | BMI: 29.1
Oedema: 2+ bilateral ankles | No JVP elevation

TODAY'S LABORATORY RESULTS:
Creatinine: 2.9 mg/dL (was 1.8 mg/dL six months ago — RAPID DECLINE)
eGFR: 19 mL/min (was 38 six months ago — DECLINED 19 UNITS IN 6 MONTHS)
BUN: 68 mg/dL | Uric acid: 9.1 mg/dL

ELECTROLYTES: Na 138 | K 6.1 mEq/L | CO2 17 mEq/L (mild metabolic acidosis)

BONE/MINERAL: Phosphorus 6.4 mg/dL (HIGH) | Calcium 8.8 mg/dL (normal) | PTH 312 pg/mL (HIGH — secondary HPT)
Vitamin D (25-OH): 14 ng/mL (deficient)

HAEMATOLOGY: Hgb 9.8 g/dL | MCV 86 (normocytic) | Iron studies: TSAT 16%, Ferritin 95 ng/mL

URINE: ACR 892 mg/g (severely elevated — was 450 mg/g six months ago)
24h urine protein: 3.1 g/day (nephrotic range approaching)
HbA1c: 8.4% (suboptimal — was 7.9% six months ago)

ASSESSMENT:
1. CKD Stage G4 A3 — RAPIDLY PROGRESSIVE (eGFR decline 19 units/6 months — exceptional rate)
   URGENT: DUAL RAAS BLOCKADE (Lisinopril + Losartan) — major safety concern (hyperkalemia, AKI)
   URGENT: Metformin CONTRAINDICATED (eGFR 19) — lactic acidosis risk
   Referral to transplant team today.
   
2. Hypertension — Severely uncontrolled (162/98) despite 3 agents
   Possible causes: dual RAAS (paradoxical effect), non-compliance, volume overload
   
3. Hyperkalemia (K 6.1) — HIGH RISK for arrhythmia
   Likely worsened by dual RAAS. Immediate dietary counselling + potassium binder.

4. CKD-Mineral Bone Disorder:
   Hyperphosphatemia (Phos 6.4) — increase phosphate binder dose
   Secondary HPT (PTH 312) — escalate calcitriol, check for tertiary HPT
   
5. Anaemia — Suboptimally treated (Hgb 9.8, iron deficient TSAT 16%)
   Add IV iron before increasing EPO dose (iron deficiency limits EPO response)
   
6. Metabolic acidosis (HCO3 17) — Initiate sodium bicarbonate 650 mg TDS
   Target HCO3 ≥22 (proven to slow CKD progression — BASE trial)

PLAN (URGENT CHANGES TODAY):
□ STOP Losartan IMMEDIATELY — remove dual RAAS, reduce hyperkalemia risk
□ STOP Metformin IMMEDIATELY — replace with insulin adjustment
□ ADD Patiromer 8.4g daily (K+ binder) — urgent given K 6.1
□ ADD Sodium Bicarbonate 650mg three times daily
□ INCREASE calcium carbonate to 2g three times daily with meals
□ IV Iron Sucrose 200mg — administer in clinic today
□ REFER: Transplant clinic TODAY given rapid progression
□ REFER: Dietitian for low-K, low-phosphorus, low-sodium renal diet TODAY
□ ECG today — K 6.1 without prior ECG unacceptable

FOLLOW-UP: 4 weeks (not 6 months — too rapid decline)
EMERGENCY: Patient instructed to present to ED if K >6.5, chest pain, palpitations, severe weakness

Dr. Fatima Al-Rashid, MD, PhD | Consultant Nephrologist | 15-Dec-2024"""
            }
            doc_text = SAMPLES[sample]
            st.text_area("Document preview:", value=doc_text[:450] + "...", height=130, disabled=True, label_visibility="collapsed")

        st.markdown("")
        run_btn = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)

    # ── Results column ────────────────────────────────────────────────────────
    with right:
        st.markdown("### 📊 Results")

        if run_btn and doc_text:
            with st.spinner("Running 4-agent pipeline…"):
                progress = st.empty()
                for step in ["📋 Intake Agent parsing…", "📚 Knowledge Agent retrieving…",
                             "⚠️ Risk Agent scoring…", "📊 Report Agent writing…"]:
                    progress.caption(step)
                    time.sleep(0.25)

                result = api_post("analyze", {
                    "document": doc_text,
                    "document_type": "medical_report",
                    "enable_drug_check": True,
                    "session_id": str(uuid.uuid4()),
                })
                progress.empty()

            if result:
                st.session_state["result"] = result

        elif run_btn and not doc_text:
            st.warning("⚠️ Please enter a document first.")

        if "result" not in st.session_state:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;color:#64748b'>
              <div style='font-size:52px'>🏥</div>
              <p style='font-size:16px;margin-top:12px'>
                Select a sample case or paste a clinical document,<br>then click <b>Run Analysis</b>
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            r = st.session_state["result"]
            pt = r.get("processing_time_seconds", 0)
            st.success(f"✅ Done in {pt:.1f}s · trace `{r.get('trace_id','')[:8]}…`")

            # Gauge
            rl = r.get("overall_risk_level", "UNKNOWN")
            st.plotly_chart(gauge_chart(rl), use_container_width=True)

            # Quick stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Conditions", len(r.get("conditions", [])))
            c2.metric("Medications", len(r.get("medications", [])))
            c3.metric("Drug Alerts", len(r.get("drug_interactions", [])))

    # ── Full-width detail tabs ────────────────────────────────────────────────
    if "result" in st.session_state:
        r = st.session_state["result"]
        st.divider()
        st.markdown("## 📑 Full Analysis")

        d_tabs = st.tabs(["👤 Patient", "⚠️ Risk Scores", "💊 Drug Alerts", "📚 Guidelines", "📋 Report", "🔄 Pipeline"])

        # Patient tab
        with d_tabs[0]:
            ca, cb = st.columns(2)
            with ca:
                st.markdown("#### Demographics")
                pi = r.get("patient_info", {})
                for k, v in pi.items():
                    if k != "vital_signs" and v:
                        st.markdown(f"**{k.replace('_',' ').title()}:** {v}")
                if pi.get("vital_signs"):
                    st.markdown("**Vital Signs:**")
                    for vk, vv in pi["vital_signs"].items():
                        if vv:
                            st.caption(f"  {vk.replace('_',' ').title()}: {vv}")
            with cb:
                st.markdown("#### Conditions")
                for c in r.get("conditions", []):
                    st.markdown(f"• {c}")
                st.markdown("#### Medications")
                for m in r.get("medications", []):
                    st.markdown(f"💊 {m}")
                if r.get("lab_values"):
                    st.markdown("#### Key Labs")
                    for k, v in r["lab_values"].items():
                        st.markdown(f"• **{k}**: {v}")

        # Risk tab
        with d_tabs[1]:
            r1, r2 = st.columns([1.1, 0.9])
            with r1:
                fig_bar = risk_bar_chart(r.get("risk_scores", {}))
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
            with r2:
                st.markdown(f"**Overall:** {risk_badge(r.get('overall_risk_level','UNKNOWN'))}", unsafe_allow_html=True)
                st.markdown("")
                for cat, data in r.get("risk_scores", {}).items():
                    if not isinstance(data, dict):
                        continue
                    lv = data.get("level", "UNKNOWN")
                    sc = data.get("score", 0)
                    jt = data.get("justification", "")
                    st.markdown(f"**{cat.replace('_',' ').title()}** — {risk_badge(lv)} `{sc:.0%}`", unsafe_allow_html=True)
                    if jt:
                        st.caption(jt)
                    st.markdown("")

            st.markdown("#### Priority Concerns")
            for rf in r.get("risk_factors", [])[:8]:
                lvl = "⚠️" if any(x in rf.upper() for x in ["HIGH", "CRITICAL", "CONTRAINDICATION"]) else "🔸"
                st.markdown(f"{lvl} {rf}")

        # Drug alerts tab
        with d_tabs[2]:
            di_list = r.get("drug_interactions", [])
            if di_list:
                for di in di_list:
                    sev = di.get("severity", "mild").lower()
                    color = {"severe": "#dc2626", "moderate": "#ea580c", "mild": "#d97706"}.get(sev, "#6b7280")
                    st.markdown(f"""
                    <div class='card interaction-{sev}' style='border-left-color:{color}'>
                      <b>💊 {di.get('drug_1','?')} &nbsp;+&nbsp; {di.get('drug_2','?')}</b>
                      <span style='float:right;color:{color};font-weight:700'>{sev.upper()}</span><br>
                      <em style='color:#94a3b8'>{di.get('description','')}</em><br>
                      <b>→</b> {di.get('recommendation','')}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No significant drug interactions detected")

        # Guidelines tab
        with d_tabs[3]:
            gl = r.get("guidelines_summary", "")
            if gl:
                st.markdown("#### Evidence-Based Guidelines (AI Synthesis)")
                st.markdown(gl)
            else:
                st.info("No guidelines summary available")
            st.metric("Knowledge Chunks Retrieved", r.get("retrieved_knowledge_count", 0))

        # Report tab
        with d_tabs[4]:
            fr = r.get("final_report", {})
            if fr:
                if fr.get("executive_summary"):
                    st.info(f"**Executive Summary:** {fr['executive_summary']}")

                if fr.get("clinical_narrative"):
                    st.markdown("#### Clinical Narrative")
                    st.markdown(fr["clinical_narrative"])

                ra, rb = st.columns(2)
                with ra:
                    st.markdown("#### ✅ Recommendations")
                    for rec in fr.get("recommendations", []):
                        st.markdown(f"• {rec}")
                with rb:
                    st.markdown("#### 🔜 Follow-Up")
                    for act in fr.get("follow_up_actions", []):
                        st.markdown(f"→ {act}")

                if fr.get("clinical_findings"):
                    st.markdown("#### Clinical Findings")
                    for f in fr["clinical_findings"]:
                        sig = f.get("significance", "normal")
                        icon = {"critical": "🔴", "abnormal": "🟡", "normal": "🟢"}.get(sig.lower(), "⚪")
                        st.markdown(f"{icon} **{f.get('category','')}**: {f.get('finding','')}")
                        if f.get("recommendation"):
                            st.caption(f"  ↳ {f['recommendation']}")

                st.divider()
                dl = json.dumps(r, indent=2, default=str)
                st.download_button(
                    "📥 Download Full Report (JSON)",
                    data=dl,
                    file_name=f"cliniq_{r.get('trace_id','x')[:8]}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        # Pipeline tab
        with d_tabs[5]:
            st.markdown("#### 🔄 Agent Execution Log")
            for step in r.get("processing_steps", []):
                cls = "step-ok" if "✅" in step else ("step-warn" if "⚠️" in step else "step-info")
                st.markdown(f"<p class='{cls}'>{step}</p>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Trace ID", r.get("trace_id", "N/A")[:8] + "…")
            m2.metric("Total Time", f"{r.get('processing_time_seconds',0):.2f}s")
            m3.metric("Agents Run", 4)

# ╔══════════════════════════════════════════════════════╗
# ║               TAB 2 — KNOWLEDGE BASE                 ║
# ╚══════════════════════════════════════════════════════╝
with tab_kb:
    st.markdown("### 📚 Knowledge Base Management")
    try:
        stats = requests.get(f"{API_BASE_URL}/knowledge-base/stats", timeout=5).json()
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Docs", stats.get("total_documents", 0))
        k2.metric("Embedding", stats.get("embedding_model", "?"))
        k3.metric("Domain", stats.get("domain", "?").upper())
    except Exception:
        st.warning("Could not fetch knowledge base stats")

    st.divider()
    st.markdown("#### ➕ Add Document to Knowledge Base")
    with st.form("kb_form"):
        kb_text = st.text_area("Content:", height=160, placeholder="Paste clinical guideline, drug reference, protocol…")
        c_src, c_cat = st.columns(2)
        kb_src = c_src.text_input("Source:", placeholder="e.g. AHA Guidelines 2024")
        kb_cat = c_cat.selectbox("Category:", ["guideline", "drug_info", "condition", "procedure", "reference"])
        kb_tags = st.text_input("Tags (comma-separated):", placeholder="cardiology, hypertension, ACE inhibitor")
        if st.form_submit_button("Add to Knowledge Base", type="primary") and kb_text and kb_src:
            res = api_post("knowledge-base/ingest", {
                "text": kb_text, "source": kb_src, "category": kb_cat,
                "tags": [t.strip() for t in kb_tags.split(",") if t.strip()],
            })
            if res:
                st.success(f"✅ Added! Total: {res.get('total_documents', '?')} documents")

# ╔══════════════════════════════════════════════════════╗
# ║                  TAB 3 — ABOUT                       ║
# ╚══════════════════════════════════════════════════════╝
with tab_about:
    st.markdown("""
    ## ClinIQ — Autonomous Multi-Agent Clinical Intelligence Platform

    ### 🔧 Architecture

    | Layer | Technology | Role |
    |-------|-----------|------|
    | **Agent Orchestration** | LangGraph (StateGraph) | Stateful 4-agent pipeline with conditional routing |
    | **LLM** | Groq — Llama 3.3 70B | Fast, free inference (14k TPM free tier) |
    | **Vector Store** | ChromaDB (persistent) | Medical knowledge base |
    | **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` | Local, free embeddings |
    | **Retrieval** | Hybrid BM25 + Semantic + RRF | Best-of-both retrieval |
    | **Observability** | Langfuse (free cloud tier) | LLM tracing + token tracking |
    | **API** | FastAPI (async) | REST backend |
    | **UI** | Streamlit + Plotly | Interactive frontend |
    | **Container** | Docker + Compose | Production deployment |

    ### 🤖 Agent Pipeline

    ```
    START
      ↓
    [1] Intake Agent        →  Parse document, extract patient_info / conditions / medications / labs
      ↓  (error → skip to report)
    [2] Knowledge Agent     →  Multi-query Hybrid RAG → Guideline synthesis (LLM)
      ↓
    [3] Risk Agent          →  Rule engine (anticoagulant checks, critical labs)
                               + LLM risk scoring (30-day readmit, CV, renal, medication)
                               + Drug interaction detection
      ↓
    [4] Report Agent        →  Full clinical narrative + recommendations + follow-up actions
      ↓
    END (structured JSON response)
    ```

    ### 🔄 Domain Adaptation
    Change `DOMAIN=finance` in `.env` + seed finance knowledge to analyse:
    - **Finance**: Earnings calls, financial statements, credit risk reports, regulatory filings
    - **Legal**: Contracts, case summaries, regulatory documents
    - **Insurance**: Claims documents, underwriting reports

    ### 🆓 All Free Resources
    - **Groq API**: Free tier (register at console.groq.com)
    - **ChromaDB**: Local, no account needed
    - **sentence-transformers**: Local, runs on CPU
    - **Langfuse**: Free cloud tier (cloud.langfuse.com)
    - **Ollama**: Local fallback (100% free, runs on any machine)

    ---
    *Built by Atharva Kate | github.com/AtharvaKate2001*
    """)
