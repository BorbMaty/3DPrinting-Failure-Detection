# Review: `main.pdf` — Issues & Suggested Fixes

> Document: *3D nyomtatási folyamat monitorizálás és hibadetektálás mesterséges intelligenciával*
> Reviewed: 2026-04-15

---

## 🔴 Critical — Factual Inconsistencies

### 1. Abstract vs. Actual Results Mismatch (Kivonat)

The abstract states metrics that **do not match** the results reported in Chapter 7:

| Metric | Abstract claims | Actual results (Ch. 7) |
|---|---|---|
| mAP50 | **92.6%** | **89.2%** (peak 90.1%) |
| Recall | **87.3%** | **83.6%** |
| Frames/sec | **3 fps** | **6 fps** (3 cameras × 2 fps, confirmed in Ch. 9.1) |

All three numbers in the abstract are inconsistent with the actual experimental data presented later in the paper. The abstract must be updated to reflect the real results.

---

### 2. `nc: 8` in `data.yaml` Example (Section 4.2.5)

The YAML code example shows:
```yaml
nc: 8
```
But the document consistently mentions **9 classes**, and the `names:` list directly below contains exactly 9 entries. Should be:
```yaml
nc: 9
```

---

## 🟠 Spelling & Typos

### 3. "Rasberry Pi" → "Raspberry Pi"

Missing **'p'** — appears **multiple times** throughout:
- Table of contents: `Kamera Agent-Rasberry pi`
- Section 5.1.1 body: `egy rasberry pi 4b 8gb`, `A rasberry pi`, `A rasberry Os`
- Keywords (Kulcsszavak): `Rasberry Pi`

**Fix:** Replace all occurrences with `Raspberry Pi` / `Raspberry OS`.

---

### 4. "hibdetekciójának" → "hibadetekciójának" (Section 2.1)

Missing letter **'a'**:

> `A 3D nyomtatási folyamatok automatizált **hibdetekciójának** kutatása`

→ `hibadetekciójának`

---

### 5. "hibadetekcójára" / "hibadetekcójához" → "hibadetekciójára" / "hibadetekciójához" (Section 2.4)

Missing letter **'i'** — appears twice:

> `hanem acélfelületek **hibadetekcójára** fókuszál`  
> `a 3D nyomtatás **hibadetekcójához** hasonló körülmények`

---

### 6. "reláciiós" → "relációs" (Section 5.3.8)

Double **'ii'** typo:

> `a **reláciiós** adatbázisokban`

→ `relációs`

---

### 7. "szügség" → "szükség" (Section 5.3.5)

Wrong vowel:

> `szügség esetén`

→ `szükség esetén`

---

### 8. "rendkivül" → "rendkívül" (Section 5.3.1)

Missing accent on **'i'**:

> `maga az egész folyamat **rendkivül** könnyen automatizálható`

→ `rendkívül`

---

### 9. "robosztusabb" → "robusztusabb" (Section 2.5)

Transposed letters:

> `**robosztusabb** detekciós rendszer valósuljon meg`

→ `robusztusabb`

---

### 10. "verszatilis" → "versatilis" (Section 5.1.1)

Wrong letter order:

> `egy elég **verszatilis** fegyver a kezünkben`

→ `versatilis` (or preferably the Hungarian: `sokoldalú`)

---

### 11. "korrekciójaárat" — Broken/Garbled Word (Section 2.2.2)

> `a nyomtatási paraméterek **korrekciójaárat**`

This is not a valid word. Likely a broken compound from the source. The sentence should probably read:

> `lehetőséget nyújt a folyamat automatikus megállítására vagy a nyomtatási paraméterek **korrekciójára**`

---

### 12. "szerepete" — Non-existent Word (Section 5.3.5)

> `Az ő **szerepete** hogy leossza a streameket`

→ `Feladata, hogy elosztja a streameket` (or: `Szerepe az, hogy leossza...`)

---

## 🟡 Grammar & Style

### 13. First Person Inconsistency (Sections 5.3 and throughout)

Section 5.3 uses **first person singular**:
> `„igen széleskörű eszköztárat biztosított **számomra**"`

But the rest of the document uses **first person plural**:
> `„használtunk"`, `„egészítettük ki"`, `„állítottunk össze"`, `„alkalmaztunk"`, etc.

**Fix:** Choose one voice (preferably plural for academic writing) and apply it consistently throughout.

---

### 14. Space Before Comma (Section 5.3.8)

> `A fő különbség **,** hogy`

There must be no space between a word and the following comma. → `A fő különbség, hogy`

---

### 15. Missing Space in Keywords (Kulcsszavak)

> `3D nyomtatás,Konvoluciós Neuronháló`

Missing space after the comma: → `3D nyomtatás, Konvolúciós neuronháló`

---

### 16. "a under_extrusion" → "az under_extrusion" (Section 7.3.2)

Hungarian grammar rule: use **"az"** before words starting with a vowel sound:

> `például a **under_extrusion** és nozzle_clog`

→ `például **az** under_extrusion és a nozzle_clog`

---

### 17. Informal Register — "lelke" (Section 5.3)

> `„A rendszer lelke teljes valójában a google cloudban található"`

"Lelke" (soul/heart of the system) is too informal/poetic for academic writing. Suggested replacement:

> `„A rendszer felhőoldali komponensei a Google Cloud Platformon futnak"`

---

### 18. Informal Metaphor — "fegyver a kezünkben" (Section 5.1.1)

> `„egy elég verszatilis fegyver a kezünkben"`

"Weapon in our hands" is an informal idiom, inappropriate for an academic paper. Suggested replacement:

> `„rendkívül sokoldalú eszköz"`

---

## 🔵 Capitalisation & Formatting

### 19. "Github" → "GitHub"

The official product name uses a capital **H**. Appears incorrectly in:
- Section 5.5 heading: `Github Action Workflows`
- Section 5.5.2 body text

→ `GitHub` everywhere.

---

### 20. "vertexAI" → "Vertex AI" (Keywords)

In the keyword list: `vertexAI` — inconsistent with how it is spelled everywhere else in the document (`Vertex AI`).

---

### 21. "Konvoluciós Neuronháló" in Keywords

Two issues:
1. Missing accent: `Konvoluciós` → `Konvolúciós`
2. Unnecessary capitalisation of the second word: `Neuronháló` → `neuronháló`

Full correction: `Konvolúciós neuronháló`

---

### 22. "usb" → "USB" (Section 5.1.1)

> `megfelelő számú **usb** porttal is rendelkezik`

→ `USB`

---

### 23. "google cloud" → "Google Cloud" (Section 5.3)

> `a **google cloudban** található`

→ `a Google Cloudban`

---

## ⚪ Structural / Minor

### 24. "Irodalom kutatás" → "Irodalomkutatás"

The chapter title (both in the Table of Contents and the Chapter 2 heading) is written as two separate words. In Hungarian this is a compound word and should be written as one:

→ **Irodalomkutatás**

Note: the bibliography is correctly written as `Irodalomjegyzék` (one word) — apply the same rule here.

---

### 25. Footnote Numbers Merged Into Body Text

In several places, footnote/citation reference numbers appear glued to the preceding word without a space (a LaTeX source issue):

- `spagetti**5**` (Section 2.2.1) — appears as if "spagetti5" is a word
- `Terraform Lépések**1**` (Figure 5.3 caption)

**Fix in LaTeX source:** Ensure `\footnote{}` and `\cite{}` commands are placed immediately after the word with no space inserted manually — or add a `~` non-breaking space if needed.

---

### 26. Unlisted "Összefoglalás" Section (End of Chapter 5)

There is a short summary paragraph titled **"Összefoglalás"** at the end of Chapter 5 (p. 46) that does **not appear in the Table of Contents**. Either:
- Add it as a numbered subsection (`5.6. Összefoglalás`) and include it in the TOC, or
- Remove the standalone heading and keep it as plain body text.

---

## ✅ Priority Fix Summary

| # | Issue | Location | Severity |
|---|---|---|---|
| 1 | Abstract metrics don't match actual results | Kivonat | 🔴 Critical |
| 2 | `nc: 8` should be `nc: 9` | Section 4.2.5 | 🔴 Critical |
| 3 | "Rasberry Pi" misspelling (~6 occurrences) | Multiple | 🟠 High |
| 4 | "korrekciójaárat" — garbled word | Section 2.2.2 | 🟠 High |
| 5 | First/third person voice inconsistency | Multiple | 🟡 Medium |
| 6 | "reláciiós", "szügség", "rendkivül", "robosztusabb" | Multiple | 🟠 High |
| 7 | "Github" → "GitHub" | Section 5.5 | 🔵 Low |
| 8 | "vertexAI", "Konvoluciós", "usb", "google cloud" | Keywords / 5.x | 🔵 Low |
| 9 | "Irodalom kutatás" → "Irodalomkutatás" | TOC / Ch. 2 | ⚪ Minor |
| 10 | Unlisted "Összefoglalás" in Ch. 5 | p. 46 | ⚪ Minor |
