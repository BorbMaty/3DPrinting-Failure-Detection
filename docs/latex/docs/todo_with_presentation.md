# Presentation TODO List

## Slide Changes

- [ ] **Slide 2** – Add a brief explanation of what FMD (Fused Metal Deposition / your specific term) is
- [x] **Slide 3** – Rework the goals section: clearly transition from "our current system" to "our goal"
- [x] **Slide 5** – Clarify the confidence threshold note:
  - Current text: *"Confidence threshold: 35% – peak F1 ≈ 0.38"*
  - Clarify that this refers to the **worst per-class F1**, not the macro/weighted average (where peak F1 = 0.88)
  - Jury members may not have read the thesis, so this distinction must be explicit on the slide
- [ ] **Slide 14** – Replace placeholder with a full-width dashboard screenshot (TODO)

## Structure

- [x] **Add an intro/summary slide at the beginning** – brief overview of what will be covered in the presentation

## Media

- [ ] **Add a demo video at the end** – must show the **entire system**, end to end (placeholder slide added at slide 16)

## Content & Clarifications to Prepare

- [ ] **Spaghetti class confusion matrix** – Be ready to explain why it underperforms; likely caused by rare, visually diverse class instances that are hard to distinguish
- [ ] **Operational cost slide** – Currently appears without context; add a concrete number: what is the **expected ROI period** for a printer farm? Important if a company jury member is present
- [ ] **Oversampling** – Be prepared to answer:
  - Was oversampling done **before or after** the train/val split?
  - Was the split **stratified**?

## Methodological Note (be ready to explain if asked)

With 2422 images split 80/20 randomly, rare classes may end up almost entirely in the training set, leaving only 3–4 validation samples. This makes validation metrics misleading — poor performance may reflect data scarcity, not model weakness.

**Stratified split** = the 80/20 ratio is maintained **per class**. E.g., 50 "foreign object" images → 40 train, 10 val — not all 50 randomly landing in train.

## General Notes

- [ ] Do a **rehearsal run** before the presentation
- [ ] Make sure all **diagrams are legible** when projected
