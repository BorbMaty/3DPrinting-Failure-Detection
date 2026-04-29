# Conference Presentation – Speaker Notes & Structure

**Title:** 3D nyomtatási folyamat monitorizálás és hibadetektálás mesterséges intelligenciával  
**Author:** Borbáth Mátyás-Levente  
**Institution:** Sapientia EMTE, Marosvásárhely  
**Format:** 14 slides · 15 minutes · LaTeX Beamer (Madrid theme, 16:9)  
**Source file:** `docs/latex/docs/presentation.tex`

---

## Platform decision

**LaTeX Beamer** was chosen because:
- All figures already exist in the correct paths (`figures/images/`)
- No re-import of assets needed
- Professional, sharp output suitable for conference projection
- Easy to maintain in sync with the thesis source

Compile with:
```bash
cd docs/latex/docs
pdflatex presentation.tex
```

---

## Slide-by-slide breakdown

### Slide 1 – Title (30 sec)
Just let it sit. Introduce yourself and the project topic in one sentence.

> "In this talk I'll present a real-time 3D printing failure detection system built with YOLOv8 on Google Cloud Platform, developed as my thesis at Sapientia."

---

### Slide 2 – Motiváció (1 min)
**Figure:** `literature/3dprintfarm.jpg`

Key points to hit:
- Printer farms with 10–100 machines make manual supervision impossible
- A missed failure can waste tens of grams of filament and damage the nozzle
- This is not a convenience feature — it's an economic necessity at scale

> "When you have 100 printers running overnight, you can't have someone watching each one. A spaghetti failure left undetected for 4 hours destroys an entire print and can clog the nozzle."

---

### Slide 3 – 9 hibaosztály (1 min)
**Figures:** 3×3 grid from `literature/` folder

Walk through briefly — don't read all 9 names. Group them:
- **High-severity** (top row + not-sticking): trigger email + FCM push
- **Low-severity** (others): Firestore log only

> "We detect 9 failure types. The top 4 — spaghetti, layer shift, warping, not sticking — are high-severity: they trigger an immediate email alert. The others are logged but don't page the operator."

---

### Slide 4 – Adathalmaz (1 min)
Key numbers to emphasize:
- **2422 annotated images** from 3 sources
- **397 own captures** on the same hardware the system runs on (reduces domain shift)
- 3-tier augmentation pipeline expands train set **~4×**

> "The hardest part of this project was the data. We combined three public datasets with 397 images we captured ourselves — on the exact same cameras and lighting the live system uses. This is crucial for minimizing domain shift."

---

### Slide 5 – YOLOv8x + kétfázisú tanítás (1.5 min)
Explain the two-phase logic:
- Phase 1 at 960px learns general features cheaply
- Phase 2 at 1280px fine-tunes on the best weights → picks up fine details like thin stringing

> "We chose YOLOv8x — the largest in the family — because accuracy matters more than speed here; inference runs in the cloud, not on the edge device. The two-phase approach let us start broad and then zoom into fine-grained details at higher resolution."

---

### Slide 6 – Tanítási görbék (1 min)
**Figure:** `results/y8x_phase2_12803/results.png`

Point out:
- All three loss curves converge cleanly
- The small bump at epoch ~86 is from `close_mosaic` — model recovered immediately
- mAP@0.5 peaked at **0.901** at epoch 60

> "All losses converge steadily. The slight bump around epoch 86 is from the mosaic augmentation shutting off — a known behavior in YOLO training — and the model recovered within 2 epochs."

---

### Slide 7 – PR görbe + Konfúziós mátrix (1 min)
**Figures:** `BoxPR_curve.png` + `confusion_matrix_normalized.png`

- PR curve shows most classes at high area under curve
- Confusion matrix: diagonal is strong; most errors are into "background" (missed detections, not false class swaps)
- Hardest classes: `under_extrusion` and `nozzle_clog` (small, visually similar)

> "The confusion matrix is clean — most errors are missed detections rather than misclassifications. The two hardest classes, under-extrusion and nozzle clog, look similar to each other and have small visual extent."

---

### Slide 8 – Validációs predikciók (1 min)
**Figure:** `results/y8x_phase2_12803/val_batch0_pred.jpg`

This is the "wow" slide — let the image speak. Point out:
- Bounding boxes align tightly with defect areas
- Multiple defects per image detected correctly
- Confidence values printed on boxes

> "This is a raw validation batch. The boxes are tight, the classes are correct, and the model handles multiple defects in a single frame without confusion."

---

### Slide 9 – Összesített teljesítmény (1 min)
Walk through the table:
- Precision **0.937** — very few false positives
- Recall **0.836** — catches most real failures
- Peak mAP@0.5 **0.901**

> "Precision of 0.937 means we almost never cry wolf. Recall of 0.836 means we catch 84% of real failures. For a safety-monitoring system, this tradeoff is intentional — false alarms erode operator trust faster than missed detections."

---

### Slide 10 – Rendszerarchitektúra (1.5 min)
**Figure:** `literature/systemarch.jpg`

Walk left-to-right through the diagram:
- Pi (edge) → Pub/Sub → Dispatcher → Vertex AI → detections-out → Alert Manager → Dashboard

> "The architecture is edge-cloud hybrid. The Raspberry Pi only captures and publishes — all the heavy lifting (GPU inference) happens in the cloud. This keeps the edge device cheap and the model upgradeable without touching the hardware."

---

### Slide 11 – Adatfolyam (1 min)
The 5-step pipeline. Walk through briskly:
1. Pi captures 2 fps per camera, JPEG-compresses, publishes to Pub/Sub
2. Dispatcher forwards to Vertex AI with OAuth2
3. Judge container runs YOLOv8x, publishes results
4. Alert-Manager writes to Firestore, sends email with 60s cooldown per camera
5. Dashboard reads Firestore in real time, shows live streams via Cloudflare tunnel

---

### Slide 12 – Dashboard (1 min)
**Figures:** `dash_normal.png` + `dash_high.png`

Side by side: normal state vs alerting state.

> "The left shows normal operation — three live camera streams and a scrolling detection log. On the right, a high-severity detection turns the border red. No email-checking needed — the operator sees it instantly."

---

### Slide 13 – Összefoglalás + Jövőbeli irányok (1 min)
Left: what was built. Right: honest list of what's next.

Highlight the two most impactful future items:
1. **Printer control integration** (OctoPrint) — currently we only alert, next step is auto-pause
2. **Scale-to-zero** — the current Vertex AI endpoint costs ~$37/day even idle

---

### Slide 14 – Köszönöm (30 sec)
Hold for questions. Dashboard URL is visible for anyone who wants to look it up.

---

## Figure inventory

| Slide | Figure path | Notes |
|---|---|---|
| 2 | `literature/3dprintfarm.jpg` | Printer farm photo |
| 3 | 9 images from `literature/` | 3×3 grid of failure types |
| 6 | `results/y8x_phase2_12803/results.png` | Training curves |
| 7 | `results/y8x_phase2_12803/BoxPR_curve.png` | PR curve |
| 7 | `results/y8x_phase2_12803/confusion_matrix_normalized.png` | Confusion matrix |
| 8 | `results/y8x_phase2_12803/val_batch0_pred.jpg` | Validation batch predictions |
| 10 | `literature/systemarch.jpg` | Full system architecture |
| 12 | `literature/dash_normal.png` | Dashboard normal state |
| 12 | `literature/dash_high.png` | Dashboard alert state |

## Figures NOT used (skip for 15-min talk)

| Figure | Reason |
|---|---|
| `literature/judge_cicd_pipeline.jpg` | Too implementation-specific |
| `literature/github_actions_cicd.jpg` | Same — mention verbally |
| `literature/mediamtx_pipeline.jpg` | Same |
| `literature/terraformstages.png` | Same |
| `results/y8x_phase2_12803/BoxF1_curve.png` | F1 number is in the table |
| `results/swarm_second_version.png` | Not referenced in active chapters |

---

## Key numbers to memorize

- **2422** annotated images
- **397** own captures
- **9** failure classes
- **0.901** peak mAP@0.5 (epoch 60)
- **0.937** precision / **0.836** recall
- **2 fps** per camera (3 cameras → 6 frames/sec total)
- **60 s** email cooldown per camera
- **35%** confidence threshold
- **~$37/day** Vertex AI T4 GPU cost
