"""
ClinIQ — Knowledge Seeder
Pre-loads the ChromaDB vector store with 8 high-quality medical knowledge chunks.
All content is derived from publicly available clinical guidelines (ACC/AHA, ADA, KDIGO, etc.)

To adapt for Finance: replace KNOWLEDGE_BASE entries with financial guidelines,
regulatory requirements, risk frameworks (Basel III, IFRS 9, etc.)
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List

from core.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

# ── Medical Knowledge Base ────────────────────────────────────────────────────
# 8 chunks covering the most common high-acuity inpatient conditions

KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "hf_acc_aha_001",
        "text": """ACC/AHA 2022 Heart Failure Guidelines — Staging and Management

Stage Classification (replaced NYHA in 2022 update):
• Stage A (At Risk): Risk factors only — HTN, DM, obesity, CAD. No structural disease. Treat: lifestyle, ACE-i if high-risk, SGLT2i if DM.
• Stage B (Pre-HF): Structural disease without symptoms (reduced/mildly reduced EF). Treat: ACE-i or ARB + beta-blocker + SGLT2i.
• Stage C (Symptomatic HF): Current/prior symptoms. HFrEF (EF≤40%): GDMT = ACE-i/ARB/ARNI + beta-blocker + MRA + SGLT2i + loop diuretic for decongestion.
• Stage D (Advanced): Refractory. Consider LVAD, transplant, or palliative care.

Key Thresholds:
• BNP >100 pg/mL: HF likely | NT-proBNP >300 pg/mL (acute) or >125 pg/mL (chronic)
• EF ≤35%: Refer for ICD evaluation; consider CRT if QRS ≥150ms
• Na <135 mEq/L at admission: independent predictor of 90-day mortality
• Cr rise >0.3 mg/dL during decongestion: cardiorenal syndrome — reassess diuretic dose

SGLT2 Inhibitors in HFrEF: Dapagliflozin or empagliflozin reduce HF hospitalisation by 25-30% regardless of DM status (Class I, Level A).
Sacubitril/valsartan (ARNI): Preferred over ACE-i in chronic HFrEF when tolerated — reduces CV death by 20%.

Acute Decompensated HF — Immediate Actions:
1. IV diuresis (furosemide ≥2.5x oral dose IV, or continuous infusion)
2. Supplemental O2/NIV for SpO2 <90%
3. Hold or reduce ACEI/ARB if Cr rising or K+ >5.0
4. Weight daily; target -0.5 to 1.0 kg/day fluid removal""",
        "source": "ACC/AHA/HFSA Heart Failure Guideline 2022 (Circulation 2022;145:e895)",
        "category": "guideline",
        "tags": ["heart failure", "HFrEF", "SGLT2", "BNP", "cardiology", "decompensated HF"],
    },
    {
        "id": "dm_ada_2024_001",
        "text": """ADA Standards of Medical Care in Diabetes 2024 — Key Recommendations

Glycaemic Targets:
• HbA1c <7.0% for most non-pregnant adults
• HbA1c <8.0% acceptable: elderly, limited life expectancy, severe hypoglycaemia unawareness, long disease duration with comorbidities
• Fasting plasma glucose: 80–130 mg/dL | Post-prandial <180 mg/dL

Pharmacotherapy Hierarchy:
1st line: Metformin (if eGFR ≥30 and no contraindications)
Add-on by comorbidity:
• Established CVD or high CV risk → GLP-1 RA (semaglutide, dulaglutide) OR SGLT2i
• HFrEF or CKD (eGFR 20-60 or albuminuria) → SGLT2i preferred (empagliflozin/dapagliflozin)
• Weight loss priority → GLP-1 RA or tirzepatide (GLP-1/GIP dual agonist)
• Cost-sensitive → Sulfonylurea (glipizide preferred; avoid glibenclamide in elderly) or NPH insulin

Metformin Contraindications:
• eGFR <30 mL/min: CONTRAINDICATED (lactic acidosis)
• eGFR 30-45: Use with caution, monitor closely
• IV iodinated contrast: Hold 48h before and after
• Active hepatic disease, excessive alcohol use

Hypoglycaemia Management:
• Level 1 (BG <70): Rule of 15 — 15g fast carbs, recheck in 15 min
• Level 2 (BG <54): Clinically significant — IV dextrose if unable to swallow
• Level 3 (severe): Altered cognition/consciousness — glucagon IM/intranasal

Annual Monitoring: HbA1c, urine ACR, eGFR, lipid panel, eye exam, foot exam, BP""",
        "source": "ADA Standards of Medical Care in Diabetes 2024 (Diabetes Care 2024;47:S1-S321)",
        "category": "guideline",
        "tags": ["diabetes", "HbA1c", "metformin", "SGLT2", "GLP-1", "hypoglycaemia", "endocrinology"],
    },
    {
        "id": "htn_aha_2017_001",
        "text": """AHA/ACC 2017 Hypertension Guidelines — Classification and Treatment

BP Classification:
• Normal: <120/<80 | Elevated: 120-129/<80 | Stage 1: 130-139 or 80-89 | Stage 2: ≥140 or ≥90
• Hypertensive Urgency: >180/>120 without organ damage | Crisis: >180/>120 with end-organ damage

Treatment Thresholds:
• CVD or 10-yr ASCVD risk ≥10%: Initiate pharmacotherapy at BP ≥130/80
• CVD risk <10%: Initiate at BP ≥140/90 after lifestyle failure

First-Line Agents (evidence-based):
• ACE inhibitors (lisinopril, ramipril) — first choice in DM, CKD, post-MI
• ARBs (losartan, valsartan) — if ACE-i intolerant
• Thiazide/thiazide-like diuretics (chlorthalidone preferred over HCTZ — longer half-life)
• Dihydropyridine CCBs (amlodipine) — first-line in Black patients, elderly, isolated systolic HTN

Combination Strategy:
• Stage 2 HTN: Initiate 2-drug combination (ACE-i/ARB + CCB or diuretic)
• Note: ACE-i + ARB COMBINATION IS CONTRAINDICATED (dual RAAS blockade → AKI, hyperkalaemia)
• Beta-blockers: NOT recommended as first-line (worse outcomes vs. CCB/diuretic for primary HTN)

Special Populations:
• CKD: ACE-i or ARB (renoprotective); avoid NSAIDs
• Post-MI: Beta-blocker + ACE-i/ARB (evidence-based)
• Black patients: CCB or thiazide diuretic preferred (ACE-i less effective as monotherapy)
• Elderly (≥65): Target <130 systolic; start low, go slow; watch orthostasis

Target: <130/80 mmHg for most adults; <140/90 in selected elderly""",
        "source": "AHA/ACC High Blood Pressure Guideline 2017 (Hypertension 2018;71:e13-e115)",
        "category": "guideline",
        "tags": ["hypertension", "blood pressure", "ACE inhibitor", "ARB", "CCB", "cardiology"],
    },
    {
        "id": "drug_interactions_001",
        "text": """High-Risk Drug Interactions in Clinical Practice — Reference Guide

ANTICOAGULANTS:
Warfarin + NSAIDs (ibuprofen, naproxen, diclofenac):
• Risk: 3-4x increased GI and intracranial bleeding (additive antiplatelet + reduced mucosal protection)
• Management: Avoid combination; if necessary, add PPI (omeprazole 20mg) and monitor INR closely

Warfarin + Antibiotics (fluoroquinolones, metronidazole, azithromycin):
• Risk: Inhibit CYP2C9 → ↑ warfarin levels → supratherapeutic INR
• Management: Reduce warfarin dose 25-50% empirically; recheck INR in 3-5 days

Warfarin + Statins (simvastatin, lovastatin, atorvastatin):
• Risk: Mild ↑ warfarin effect; simvastatin/lovastatin also carry myopathy risk via CYP3A4
• Management: Monitor INR at dose changes; prefer rosuvastatin or pravastatin with warfarin

RAAS DUAL BLOCKADE (ACE-i + ARB):
• Risk: Significantly ↑ hyperkalaemia, hypotension, and acute kidney injury
• Evidence: ONTARGET trial showed harm with combination vs. monotherapy
• Management: DO NOT combine; choose one agent; monitor K+ and creatinine if combination used in error

STATINS — Myopathy Risk:
Simvastatin/lovastatin + Azole antifungals (fluconazole, itraconazole):
• Risk: CYP3A4 inhibition → 3-10x statin levels → rhabdomyolysis
• Management: Use fluconazole sparingly; switch to pravastatin or rosuvastatin (not CYP3A4 substrates)

Simvastatin >20mg + Amiodarone:
• Risk: CYP3A4 inhibition → myopathy; max simvastatin dose 20mg with amiodarone
• Management: Limit dose or switch statin

METFORMIN INTERACTIONS:
IV iodinated contrast: Hold metformin 48h before and restart 48h after IF renal function stable
Alcohol (heavy use): ↑ lactic acidosis risk
Topiramate: May ↑ metformin levels — monitor for GI toxicity

OPIOIDS + BENZODIAZEPINES:
• Risk: Synergistic CNS and respiratory depression; 3.86x higher overdose death risk
• Management: Avoid combination; if prescribed, lowest effective doses; naloxone co-prescription recommended""",
        "source": "Clinical Pharmacology Drug Interaction Database / Lexicomp / Micromedex 2024",
        "category": "drug_info",
        "tags": ["drug interactions", "warfarin", "NSAIDs", "ACE inhibitor", "ARB", "statins", "metformin"],
    },
    {
        "id": "sepsis_ssc_2021_001",
        "text": """Surviving Sepsis Campaign (SSC) 2021 Guidelines — Hour-1 Bundle and Management

Definitions (Sepsis-3):
• Sepsis: Life-threatening organ dysfunction from dysregulated host response to infection; SOFA score ≥2
• Septic Shock: Sepsis + vasopressor requirement to maintain MAP ≥65 + lactate >2 mmol/L despite fluid resuscitation

Diagnostic SOFA Components:
Respiratory (PaO2/FiO2) | Coagulation (Plt <150) | Liver (Bili >1.2) | Cardiovascular (MAP or vasopressors) | CNS (GCS) | Renal (Cr >1.2)

HOUR-1 BUNDLE (Mandatory, all elements within 60 minutes):
1. Measure lactate; re-measure if initial >2 mmol/L
2. Obtain blood cultures (minimum 2 sets) BEFORE antibiotics
3. Administer broad-spectrum antibiotics immediately
4. Administer 30 mL/kg crystalloid for hypotension or lactate ≥4 mmol/L
5. Apply vasopressors for persistent hypotension (MAP <65 despite fluids) — norepinephrine first-line

Antibiotic Selection:
• Community-onset: Ceftriaxone 2g IV + azithromycin 500mg (CAP); pip/tazo 4.5g IV Q6h (abdominal/urinary)
• Hospital-acquired/VAP/immunocompromised: Meropenem 1-2g IV + vancomycin 25-30mg/kg load
• Add antifungal (micafungin or fluconazole) if risk factors: immunosuppression, TPN, prolonged antibiotics

Vasopressor Guidance:
• Norepinephrine: 0.01-3 mcg/kg/min — first-line; MAP target ≥65 mmHg
• Vasopressin 0.03 units/min: Add as second agent; reduces norepinephrine dose
• Hydrocortisone 200mg/day: Add if refractory shock (vasopressor requirements increasing after 24h)

Targets: MAP ≥65, lactate normalisation (<2 within 6h), UO >0.5 mL/kg/h
Fluid balance: Avoid liberal fluids after initial resuscitation — FACTT trial and SMART data support balanced crystalloids""",
        "source": "Surviving Sepsis Campaign International Guidelines 2021 (Intensive Care Med 2021;47:1181-1247)",
        "category": "guideline",
        "tags": ["sepsis", "septic shock", "lactate", "antibiotics", "vasopressors", "critical care", "ICU"],
    },
    {
        "id": "lab_reference_001",
        "text": """Critical Laboratory Values — Interpretation and Action Thresholds

COMPLETE BLOOD COUNT:
• Hgb <7 g/dL: Transfusion threshold in most inpatients (liberal target ≥8 in cardiac patients)
• Hgb <10 g/dL in CKD: Consider erythropoiesis-stimulating agent
• WBC >15,000: Significant leukocytosis — infection, inflammation, steroid effect; >30,000 consider haematology consult
• WBC <2,000 (neutrophils <500): Febrile neutropenia — empiric broad-spectrum antibiotics within 1 hour
• Platelets <50,000: Major bleeding risk; hold anticoagulation, avoid invasive procedures
• Platelets <20,000: Spontaneous bleeding risk — transfusion threshold

METABOLIC PANEL — CRITICAL VALUES:
• Na <125 mEq/L: Symptomatic hyponatraemia — urgent neurology consult, correct ≤8-10 mEq/24h (ODS risk)
• Na >155 mEq/L: Hypernatraemia — free water deficit calculation; correct slowly over 48-72h
• K <3.0 mEq/L: Hypokalemia — IV/oral repletion; cardiac monitoring if <2.5; also correct Mg
• K >5.5 mEq/L: Hyperkalemia — cardiac monitoring; K >6.5 or ECG changes: emergency (calcium, insulin-dextrose, bicarb)
• Creatinine: Acute ≥0.3 mg/dL rise above baseline within 48h = AKI Stage 1 (KDIGO)
• Lactate >2 mmol/L: Tissue hypoperfusion; >4 = septic shock criteria; repeat to trend

CARDIAC:
• Troponin I/T: Any value above 99th percentile ULN = myocardial injury; rise-and-fall pattern = acute MI
• BNP >500 / NT-proBNP >2000 (acute): High HF probability
• D-dimer >500 ng/mL FEU: Low pre-test probability PE ruled in for workup; not diagnostic alone

COAGULATION:
• INR >4.0 (non-anticoagulated): Severe coagulopathy — liver disease or vitamin K deficiency
• INR >3.5 on warfarin: Hold warfarin; vitamin K if urgent reversal needed
• Fibrinogen <150: DIC consideration — check FDP, D-dimer, clinical correlation""",
        "source": "ARUP Laboratory / Mayo Clinic Reference Values / AACC Critical Values Guidelines 2024",
        "category": "reference",
        "tags": ["lab values", "CBC", "metabolic panel", "troponin", "BNP", "creatinine", "potassium", "INR"],
    },
    {
        "id": "readmission_lace_001",
        "text": """30-Day Readmission Risk — LACE Score and Evidence-Based Prevention

LACE Index (validated readmission predictor):
L — Length of stay: 1 day=1pt, 2=2, 3=3, 4-6=4, 7-13=5, ≥14=7
A — Acuity of admission: Emergent/urgent=3, Elective=0
C — Charlson Comorbidity Index ≥4=5, 3=3, 2=2, 1=1, 0=0
E — ED visits in prior 6 months: 4+=4, 3=2, 1-2=1, 0=0
• LACE ≥10: HIGH risk (>20% 30-day readmission) — requires transitional care intervention

High-Risk Diagnoses (CMS penalty program):
• Heart Failure: LVEF <25%, BNP >700 at discharge, Na <135, Cr >1.5 increase mortality + readmit risk
• COPD: FEV1 <30%, >2 hospitalisations/year, O2-dependent
• Pneumonia: Bacteraemia, multilobar, CURB-65 ≥3
• AMI, CABG, TAVR: 30-day readmission rates tracked by CMS

Evidence-Based Prevention Interventions:
1. Medication reconciliation at discharge → 30% readmission reduction (Joint Commission mandate)
2. Follow-up within 7 days of discharge → 20-30% reduction (strong evidence)
3. Structured teach-back education: Patient demonstrates understanding, not just acknowledges
4. Remote monitoring: Daily weight in HF (>2 kg gain in 2 days = action threshold); pulse oximetry for COPD
5. Transitional care nurse calls: 24-48h post-discharge and again at 7 days
6. Social determinants screening (SDOH): Housing instability, food insecurity, transportation barriers add 2-3x readmission risk
7. Palliative care for Stage D HF, COPD GOLD 4, end-stage CKD: Reduces hospitalisation + improves QoL

Discharge Checklist Must-Haves:
□ Medication list reconciled + printed | □ Follow-up appointment scheduled | □ Red-flag symptoms explained
□ Weight scale at home (HF) | □ Primary care notified within 24h | □ Transition care coordinator assigned (LACE ≥10)""",
        "source": "CMS Hospital Readmissions Reduction Program / LACE Score (van Walraven 2010) / AHA Transitional Care",
        "category": "guideline",
        "tags": ["readmission", "LACE", "transitional care", "discharge planning", "heart failure", "COPD"],
    },
    {
        "id": "ckd_kdigo_2024_001",
        "text": """KDIGO 2024 CKD Guidelines — Staging, Management, and Medication Adjustments

CKD Staging (eGFR mL/min/1.73m²):
• G1: ≥90 | G2: 60-89 | G3a: 45-59 | G3b: 30-44 | G4: 15-29 | G5: <15 (kidney failure)
Albuminuria categories: A1 <30 | A2 30-300 | A3 >300 mg/g creatinine
Combined staging (e.g. G3b A3) guides intensity of management.

Disease Progression: eGFR decline >5 mL/min/year = rapidly progressive — urgent nephrology referral

Management by Stage:
• G1-G2: BP control <130/80 (ACE-i or ARB for proteinuric DKD), lifestyle, manage CVD risk
• G3a-G3b: Add SGLT2i (dapagliflozin 10mg) if DM or HF with eGFR ≥20; finerenone for DKD G3-G4
• G3b-G4: Anaemia management (Hgb target 10-11.5 with ESA if eGFR <30); CKD-MBD (phosphate binders, calcitriol/alfacalcidol, calcimimetics if on dialysis)
• G4: Nephrology referral mandatory; dialysis access planning; transplant evaluation
• G5: Dialysis initiation or kidney transplant; conservative management if appropriate

Critical Medication Adjustments in CKD:
• Metformin: eGFR 30-45 (reduce dose); <30 CONTRAINDICATED
• NSAIDs: Avoid in G3+; even short-term worsens GFR and raises BP
• Gabapentin/Pregabalin: eGFR 30-59: reduce 50%; <30: reduce 75%
• Low-molecular-weight heparins (enoxaparin): eGFR <30: use UFH or reduce LMWH dose with anti-Xa monitoring
• Antibiotics: Penicillins, cephalosporins, fluoroquinolones, carbapenems all require dose reduction in G4-G5
• Contrast nephropathy prevention: IV hydration (1 mL/kg/h NS), minimise contrast volume, hold nephrotoxins 24-48h

Hyperkalemia in CKD: K >5.0 in G3+ — consider patiromer or sodium zirconium cyclosilicate; low K diet
Anaemia targets: Hgb 10-11.5 g/dL with ESA; iron replete before ESA (TSAT >20%, ferritin >200)""",
        "source": "KDIGO 2024 CKD Clinical Practice Guideline Update (Kidney Int 2024)",
        "category": "guideline",
        "tags": ["CKD", "chronic kidney disease", "eGFR", "KDIGO", "dialysis", "metformin", "nephrology"],
    },
]


def seed_medical_knowledge() -> None:
    """
    Seed the ChromaDB vector store with curated medical knowledge.
    Idempotent: skips if collection already has documents (upsert via IDs).
    """
    vs = get_vectorstore()
    current_count = vs.count()

    if current_count >= len(KNOWLEDGE_BASE):
        logger.info(f"Knowledge base already has {current_count} documents — skipping seed")
        return

    logger.info(f"Seeding {len(KNOWLEDGE_BASE)} medical knowledge chunks into vector store...")

    documents = [item["text"] for item in KNOWLEDGE_BASE]
    metadatas = [
        {
            "source": item["source"],
            "category": item["category"],
            "tags": ", ".join(item["tags"]),
            "domain": "healthcare",
        }
        for item in KNOWLEDGE_BASE
    ]
    ids = [item["id"] for item in KNOWLEDGE_BASE]

    vs.add_documents(documents=documents, metadatas=metadatas, ids=ids)
    logger.info(f"✅ Knowledge base seeded — {vs.count()} total documents")


if __name__ == "__main__":
    # Run directly: python -m tools.knowledge_seeder
    logging.basicConfig(level=logging.INFO)
    seed_medical_knowledge()
