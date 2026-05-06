# Előadói jegyzetek – konferencia előadás

**Cím:** 3D nyomtatási folyamat monitorizálás és hibadetektálás mesterséges intelligenciával  
**Előadó:** Borbáth Mátyás-Levente  
**Intézmény:** Sapientia EMTE, Marosvásárhely  
**Célkeret:** 15 perc előadás + kérdések  
**Diák száma:** 17

---

## Időterv

| Dia | Cím | Idő |
|-----|-----|-----|
| 1 | Cím | 30 mp |
| 2 | Áttekintés | 30 mp |
| 3 | Motiváció | 1 perc |
| 4 | Két súlyossági szint | 1 perc |
| 5 | 9 hibaosztály – képrács | 45 mp |
| 6 | Adathalmaz | 1 perc |
| 7 | YOLOv8x + kétfázisú tanítás | 1:30 perc |
| 8 | Tanítási görbék | 1 perc |
| 9 | PR görbe + Konfúziós mátrix | 1 perc |
| 10 | Validációs predikciók | 45 mp |
| 11 | Összesített teljesítmény | 1 perc |
| 12 | Rendszerarchitektúra | 1:30 perc |
| 13 | Adatfolyam | 1 perc |
| 14 | Élő dashboard | 30 mp |
| 15 | Összefoglalás + Jövő | 1 perc |
| 16 | Élő bemutató (demó) | 1 perc |
| 17 | Köszönet | 15 mp |
| **Összesen** | | **~15 perc** |

---

## Dia 1 – Cím *(~30 mp)*

**Mit mutatsz:** A cím dia, a neved, témavezetők.

**Mit mondj:**

> „Jó napot kívánok! Borbáth Mátyás-Levente vagyok, a Sapientia Erdélyi Magyar Tudományegyetem végzős hallgatója. Diplomamunkámban egy valós idejű 3D nyomtatási hibadetektáló rendszert dolgoztam ki, amely mesterséges intelligenciát és felhőalapú infrastruktúrát ötvöz. Engedjék meg, hogy bemutassam a rendszert és az elért eredményeket."

---

## Dia 2 – Áttekintés *(~30 mp)*

**Mit mutatsz:** 6 pontos tartalomjegyzék.

**Mit mondj:**

> „Az előadás hat részből áll: először a problémát és a motivációt mutatom be, majd az adathalmazt és a tanítási stratégiát. Ezután a modell teljesítményét értékelem, bemutatom a teljes rendszerarchitektúrát, ha az idő engedi, egy demó videót is láthatnak, végül összefoglalom az eredményeket és a jövőbeli terveket."

---

## Dia 3 – Motiváció *(~1 perc)*

**Mit mutatsz:** Szöveges érvek + nyomtatófarm fotó.

**Mit mondj:**

> „Miért van szükség automatizált monitorozásra? Képzeljünk el egy FDM nyomtatófarmon 50–100 gépet, amelyek egyszerre üzemelnek — akár éjszaka is. Egy ember nem képes mindet egyszerre figyelni.
>
> Ha egy nyomtatási hiba észrevétlenül marad, az következményei súlyosak: anyagveszteség, esetleg gépsérülés, és ami még kritikusabb — kiesési idő. Ezek nem apró kellemetlenségek, hanem közvetlen üzleti veszteséget jelentenek.
>
> A korai, automatikus detekció tehát nem kényelmi funkció — gazdasági szükségszerűség."

---

## Dia 4 – 9 hibaosztály – két súlyossági szint *(~1 perc)*

**Mit mutatsz:** Vörös doboz (magas súlyosság) + zöld doboz (alacsony súlyosság).

**Mit mondj:**

> „A rendszer 9 különböző hibatípust ismer fel, amelyeket két csoportba osztottunk a beavatkozás sürgőssége szerint.
>
> A magas súlyosságú hibák — spagetti, nem tapadó réteg, réteg-eltolódás és vetemedés — azonnal e-mail értesítést és push üzenetet váltanak ki, mert ezek leállítandó nyomtatást jelentenek.
>
> Az alacsony súlyosságúak — például szálazás, alul- vagy felül-extrúdálás — csak az adatbázisba kerülnek naplózásra. Ezek fokozatos minőségromlást okoznak, az operátor dönt a beavatkozásról."

---

## Dia 5 – 9 detektált hibaosztály – képrács *(~45 mp)*

**Mit mutatsz:** 3×3 képrács valódi tanítóképekkel, minden hibatípusból egy-egy.

**Mit mondj:**

> „Ez a kilenc hibaosztály vizuálisan. Láthatják a spagettit — amikor a szál a levegőbe nyomódik —, a réteg-eltolódást, a vetemedést az aljánál, a szálazást, a fúvóka-dugulást és a többi kategóriát. Némelyik, mint például a felül- és alul-extrúdálás, vizuálisan nagyon hasonlít egymásra — ez az egyik legnagyobb kihívás a modell számára."

---

## Dia 6 – Adathalmaz összeállítása *(~1 perc)*

**Mit mutatsz:** Bal oldal: adatforrások és felosztás. Jobb oldal: háromszintű augmentációs pipeline.

**Mit mondj:**

> „Az adathalmaz összeállítása volt a projekt legidőigényesebb része. Összesen 2422 annotált képet használtunk 9 osztályhoz: Kaggle-ről és HuggingFace-ről nyilvános adathalmazokat töltöttünk le, de ami döntő volt — 397 képet mi magunk készítettünk, ugyanazon a kamerán és ugyanolyan megvilágítás mellett, amin a rendszer élesben fut. Ez minimalizálja a domain shift-et.
>
> Az augmentációs pipeline háromszintű: alapvetranszformációk, Smart MixUp kompatibilis osztálypárokkal, és mozaik augmentáció. Ezzel a tanítókészlet közel négyszeresére bővült."

---

## Dia 7 – YOLOv8x – Kétfázisú tanítási stratégia *(~1:30 perc)*

**Mit mutatsz:** Bal: két fázis leírása. Jobb: miért YOLOv8x + konfidenciaküszöb.

**Mit mondj:**

> „A modellválasztásnál a YOLOv8x-re esett a döntés — ez a YOLOv8 családon belül a legnagyobb és legpontosabb változat. Fontos érv: az inferencia a felhőben, T4 GPU-n fut, nem az edge eszközön, tehát a sebesség másodlagos — a pontosság az elsődleges szempont.
>
> A kétfázisú tanítás lényege a következő: az első fázisban 960 pixeles bemeneti felbontáson, COCO előtanított súlyokból indulva az általános képjellemzőket tanulja meg a modell. A második fázisban — 1280 pixeles felbontáson — az első fázis legjobb súlyaiból folytatjuk a tanítást: 100 epokig, AdamW optimalizálóval, koszinuszos tanulási ráta ütemezéssel.
>
> A konfidenciaküszöböt 35%-ra állítottuk be. Ezt az indokolja, hogy a leggyengébben teljesítő osztálynál a csúcs F1 érték 0.38 körül van — ha magasabb küszöböt használnánk, ezeket az eseteket teljesen kihagynánk. A makro F1 csúcs egyébként 0.88."

---

## Dia 8 – Tanítási görbék – 100 epok *(~1 perc)*

**Mit mutatsz:** A `results.png` grafikon — veszteség görbék és mAP görbe.

**Mit mondj:**

> „A tanítási görbék megmutatják, hogy a modell stabilan konvergált. A box, osztályozási és DFL veszteségek egyenletesen csökkennek mind a tanítón, mind a validáción.
>
> Figyeljék meg a kis uggrást a 86. epok körül: ez a mozaik augmentáció kikapcsolásának ismert mellékhatása YOLO tréningeknél — a modell két epokon belül visszatért a trendhez.
>
> A mAP@0.5 csúcsa 0.901 a 60. epokon, a mAP@0.5:0.95 csúcsa pedig 0.683 a 68. epokon."

---

## Dia 9 – Precision-Recall görbe & Konfúziós mátrix *(~1 perc)*

**Mit mutatsz:** PR görbe bal oldalon, normalizált konfúziós mátrix jobb oldalon.

**Mit mondj:**

> „A PR görbe mutatja, hogy a legtöbb osztálynál magas az AUC — a spagetti és a réteg-eltolódás szinte tökéletes. A nehezebb osztályok, mint az alul-extrúdálás és a fúvóka-dugulás, kissé alacsonyabb görbét mutatnak, ez várható vizuális hasonlóságuk miatt.
>
> A konfúziós mátrix átlója erős. A legfontosabb megfigyelés: a hibák döntő többsége háttérbe való téves sorolásként jelenik meg — vagyis kihagyott detekció —, nem keresztosztályos tévesztésként. Ez jobb helyzet, mint a fordított eset lenne."

---

## Dia 10 – Validációs predikciók *(~45 mp)*

**Mit mutatsz:** `val_batch0_pred.jpg` — egy validációs köteg bounding boxokkal.

**Mit mondj:**

> „Ez egy nyers validációs köteg. A bounding boxok szorosan illeszkednek a hibaterületekre, a modell egyszerre több hibát is detektál egyetlen képen belül, és a konfidenciaértékek is realisztikusak. Ezt a képet nem finomhangoltuk — ez a modell valós teljesítménye."

---

## Dia 11 – Összesített teljesítmény *(~1 perc)*

**Mit mutatsz:** Bal: metrika táblázat. Jobb: három kiemelt érték blokkban.

**Mit mondj:**

> „Foglaljuk össze a számokat. A pontosság 0.937 — ez azt jelenti, hogy szinte soha nem küldünk hamis riasztást. A visszahívás 0.836 — a valós hibák 84%-át detektáljuk. Az mAP@0.5 0.892, a csúcsérték 0.901 volt.
>
> Ez a pontossági arány szándékos választás: monitorozási rendszernél a hamis riasztás gyorsan aláássa az operátor bizalmát, ezért érdemes inkább a pontosságot előtérbe helyezni a visszahívással szemben."

---

## Dia 12 – Edge-Cloud hibrid architektúra *(~1:30 perc)*

**Mit mutatsz:** A rendszerarchitektúra diagram.

**Mit mondj:**

> „Térjünk rá a rendszerre. Az architektúra edge-cloud hibrid: az érzékelés az edge-en, a Raspberry Pi-n történik, az inferencia pedig a felhőben.
>
> A Raspberry Pi 4B-hez három USB webkamera csatlakozik. A frame-extractor szolgáltatás 2 képkockánként másodpercenként kameránként JPEG-et készít, és publikálja a Google Cloud Pub/Sub üzenetsorba.
>
> A felhőben egy Dispatcher Cloud Function veszi át az üzenetet, hitelesíti és továbbítja a Vertex AI-on futó Judge konténernek, amely YOLOv8x inferenciát futtat T4 GPU-n. Az eredmények egy másik Pub/Sub témába kerülnek.
>
> Az Alert-Manager Cloud Function kiolvassa ezeket: magas súlyosságú detekciónál e-mailt küld — kameránként 60 másodperces cooldown-nal —, és a Firestore adatbázisba ír. Az élő dashboard a Firestore onSnapshot API-val valós időben frissül, a kamerastream-ek WebRTC protokollon, Cloudflare alagúton keresztül érhetők el."

---

## Dia 13 – Az adatfolyam lépései *(~1 perc)*

**Mit mutatsz:** 5 lépéses sorszámozott lista + egy kis infrastruktúra blokk.

**Mit mondj:**

> „Az öt lépéses pipeline gyorsan: a Pi rögzíti és publikálja a képeket, a Dispatcher hitelesítve továbbítja a Vertex AI-nak, a Judge futtatja az inferenciát és közzéteszi az eredményeket, az Alert-Manager értesítéseket küld és naplóz, a Dashboard pedig valós időben megjeleníti az összes adatot.
>
> Az infrastruktúra teljes egészében Terraform kóddal van definiálva — ez újratelepíthető és auditálható. A CI/CD három GitHub Actions workflow-ból áll: Python tesztek, Terraform lint és tervfuttatás, valamint Docker image build a Judge számára."

---

## Dia 14 – Élő dashboard *(~30 mp)*

**Mit mutatsz:** Dashboard képernyőkép (vagy TODO placeholder).

> **Megjegyzés:** Ha a placeholder még nincs lecserélve, mondd el szóban, mit mutat a dashboard, és mutasd meg élőben ha lehetséges.

**Mit mondj:**

> „A dashboard három élő WebRTC kamerastream-et jelenít meg egyszerre, mellette egy valós idejű detekciós naplóval. Magas súlyosságú detekciónál a felület vizuálisan is jelzi a riasztást. Az operátor egy böngészőből, bárhonnan elérheti a rendszert."

---

## Dia 15 – Összefoglalás és jövőbeli irányok *(~1 perc)*

**Mit mutatsz:** Bal: elért eredmények. Jobb: jövőbeli irányok.

**Mit mondj:**

> „Összefoglalva: sikerült egy 9 osztályos hibadetektáló modellt betanítani 0.892-es mAP@0.5 értékkel, egy teljes edge-cloud pipeline-t felépíteni a Raspberry Pi-tól a GCP-ig, és valós idejű dashboard-ot és e-mail riasztást megvalósítani.
>
> A jövőbeli fejlesztések közül kettőt emelnék ki. Az egyik a nyomtatóvezérlő integráció — jelenleg csak riasztunk, a következő lépés az automatikus nyomtatás-szüneteltetés lenne OctoPrint-en keresztül. A másik a scale-to-zero megoldás: a jelenlegi Vertex AI endpoint naponta mintegy 37 dollárt költ még üresjáratban is — ezt el kell számolni egy éles telepítésnél."

---

## Dia 16 – Élő bemutató *(~1 perc)*

**Mit mutatsz:** Demó videó (vagy TODO placeholder).

> **Megjegyzés:** Ha a videó elkészült, hagyd lejátszani. Ha nem, mutasd meg élőben a dashboard-ot, vagy írj le egy konkrét esetet.

**Mit mondj (ha nincs videó):**

> „Egy rövid forgatókönyv a rendszer működéséről: a nyomtató szálazást kezd produkálni. A Raspberry Pi 0.5 másodpercen belül rögzíti a képkockát, a Judge 1–2 másodpercen belül detektálja és klasszifikálja a hibát, az Alert-Manager ír a Firestore-ba, a dashboard azonnal frissül. Ha a hiba magas súlyosságú, az operátor 3–5 másodpercen belül e-mailt kap."

---

## Dia 17 – Köszönet / Kérdések *(~15 mp)*

**Mit mondj:**

> „Köszönöm a figyelmet! Szívesen válaszolok a kérdéseikre."

---

## Várható kérdések – felkészülési anyag

### 1. Miért YOLOv8x és nem egy kisebb modell?
> Az inferencia a felhőben fut T4 GPU-n, ahol a számítási kapacitás nem korlát. A rendszer nem valós idejű videófolyamot dolgoz fel, hanem 2 fps-es képkockákat — tehát a latencia nem kritikus. Így a pontosság dominálta a választást.

### 2. A 35%-os konfidenciaküszöb miért ilyen alacsony?
> A leggyengébben teljesítő osztálynál — fúvóka-dugulás, alul-extrúdálás — a csúcs F1 értéke 0.38 körül van. Ha 50%-ra emeljük a küszöböt, ezeket az eseteket szinte sosem detektáljuk. A hamis pozitívak ellen a 60 másodperces cooldown véd, vagyis ugyanarról a kameráról percenként maximum egy riasztás megy.

### 3. Az oversampling a split előtt vagy után történt?
> Az oversampling a train/val split után, csak a tanítókészleten történt, hogy elkerüljük az adatszivárgást a validációs halmazba. A split véletlen 80/20 volt seed=42 értékkel.

### 4. A split stratifikált volt-e?
> Nem volt stratifikált split. Egyes ritka osztályoknál ez azt jelenti, hogy kevés validációs minta áll rendelkezésre — a metrikák ezért ezeken az osztályokon kevésbé megbízhatóak. Ez ismert korlátja az értékelésnek.

### 5. Miért nem fut az inferencia a Raspberry Pi-n?
> A YOLOv8x 68 millió paramétert tartalmaz. Raspberry Pi 4B-n egy képkocka feldolgozása 8–15 másodpercet venne igénybe — a rendszer gyakorlatilag használhatatlan lenne. A felhős T4 GPU-n ez 100–200 ms.

### 6. Mennyi az üzemelési költség?
> A Vertex AI T4 GPU endpoint napi kb. 37 dollárba kerül, ha folyamatosan fut. Ez éves szinten 13–14 ezer dollár. Egy skálázható megoldásnál scale-to-zero szükséges — ez az egyik legfontosabb jövőbeli fejlesztési irány.

### 7. Miért teljesít gyengébben a spagetti osztály a konfúziós mátrixon?
> A spagetti vizuálisan nagyon változatos: vékony szálak, vastag kusza tömegek, különböző szín és megvilágítás. Az adathalmazban ez az osztály viszonylag kis mintaszámmal és nagy vizuális varianciával rendelkezik. Az aktív tanulási ciklus segíthetne ezen.

---

## Kulcsszámok – memorizálandó

| Szám | Jelentés |
|------|----------|
| 2422 | Annotált képek száma |
| 397 | Saját felvételek száma |
| 9 | Hibaosztályok száma |
| 0.901 | Csúcs mAP@0.5 (60. epok) |
| 0.892 | Végső mAP@0.5 |
| 0.937 | Pontosság (Precision) |
| 0.836 | Visszahívás (Recall) |
| 0.88 | Csúcs makró F1 |
| 2 fps | Képkocka-kinyerés sebessége kameránként |
| 60 s | E-mail cooldown kameránként |
| 35% | Konfidenciaküszöb |
| ~$37/nap | Vertex AI T4 GPU endpoint költsége |
