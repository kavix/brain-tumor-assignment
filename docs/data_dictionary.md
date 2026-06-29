# Brain Tumor Classification Dataset — Data Dictionary

**Synthetic data for ML coursework only — not real patient records.**

- **Rows:** 9,000
- **Columns:** 29 (1 identifier + 27 features + 1 target)
- **Task:** 3-class classification
- **Target:** `tumor_type` ∈ {Glioma, Meningioma, Pituitary} 

## Identifier
| Column | Type | Notes |
|---|---|---|
| patient_id | string | Unique ID, drop before modelling |

## Features (27)
| # | Column | Type | Values / Range |
|---|---|---|---|
| 1 | age | numeric | 5–95 |
| 2 | gender | categorical | Male, Female (a few lowercase to clean) |
| 3 | ethnicity | categorical | Asian, Caucasian, African, Hispanic, Other |
| 4 | region | categorical | Urban, Suburban, Rural |
| 5 | bmi | numeric | 14–50 |
| 6 | smoking_status | categorical | Never, Former, Current |
| 7 | alcohol_consumption | ordinal cat. | None, Moderate, Heavy |
| 8 | family_history | categorical | Yes, No |
| 9 | tumor_size_mm | numeric | 2–90 |
| 10 | tumor_location | categorical | Frontal, Temporal, Parietal, Occipital, Cerebellum, Sellar, Convexity |
| 11 | tumor_growth_rate | numeric | 0.1–20 |
| 12 | headache_severity | numeric | 0–10 |
| 13 | nausea | categorical | Yes, No |
| 14 | vision_problems | categorical | Yes, No |
| 15 | seizures | categorical | Yes, No |
| 16 | memory_loss | categorical | Yes, No |
| 17 | balance_issues | categorical | Yes, No |
| 18 | mri_intensity | numeric | 0–255 |
| 19 | ct_density | numeric | Hounsfield-like, -50–120 |
| 20 | edema_grade | ordinal | 0, 1, 2, 3 |
| 21 | contrast_enhancement | ordinal cat. | None, Mild, Moderate, Strong |
| 22 | ki67_index | numeric | proliferation marker, 0.1–60 |
| 23 | bp_systolic | numeric | 85–210 |
| 24 | bp_diastolic | numeric | 50–130 |
| 25 | wbc_count | numeric | 2–20 |
| 26 | crp_level | numeric | 0.1–60 |
| 27 | genetic_marker_status | categorical | Positive, Negative, Inconclusive |


