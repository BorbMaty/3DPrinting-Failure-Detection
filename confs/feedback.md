---
title: Feedback — codebase & docs review
tags: [feedback, review, project/printermonitor]
aliases: [feedback]
type: feedback
---

# Feedback

Observations from the doc-sync pass on **2026-06-01**, after reconciling the [[00-index|knowledge base]] and `CLAUDE.md` against the current source. Ordered roughly by impact. None of this is blocking — the system clearly works — these are the things I'd tighten next.

## Security / correctness

1. **`system_state` is world-writable.** `firestore.rules` grants `read, write: if true` on the `system_state` collection so the unauthenticated dashboard can flip the capture kill-switch. That means *anyone* who loads `https://printermonitor-488112.web.app` can stop your Pi from capturing frames (or spam writes). Fine for a thesis demo; call it out explicitly in the thesis as a known limitation, or gate it behind Firebase Anonymous Auth + a rule like `request.auth != null`. See [[03-pi-edge#Remote capture kill-switch]], [[06-dashboard#Known issues / smells]].

2. **Public `*-frames` bucket with no lifecycle rule.** Every inference uploads a JPEG to a publicly-readable bucket and nothing ever deletes it. At 0.1 fps × 3 cams ≈ 26k objects/day, ~unbounded. Two asks:
   - Add a `lifecycle { rule { action = "Delete"; condition { age = 7 } } }` to `google_storage_bucket.frames` so old frames self-expire.
   - The frames are inference snapshots of your print bed — "non-sensitive" is true today, but public-read + guessable paths (`frames/cam1/<ts>.jpg`) means anyone can enumerate your prints. Consider signed URLs or a CDN-fronted private bucket if this ever leaves the thesis context.
   See [[05-infrastructure#Storage]], [[11-costs-and-monitoring]].

3. **`inferences` collection grows forever.** `flush_firestore.py` can clear it manually, but there's no TTL. Firestore TTL policies (a `created_at`-based TTL field) are free and would cap both cost and the page-2 query surface. See [[01-architecture#Failure modes]].

## Code smells / drift

4. **`alert-manager` `CONF_THRESHOLD` code default is `0.20`, not `0.35`.** Production is fine because Terraform injects `var.conf_threshold` (0.35) into the function env — but if anyone runs the module outside Terraform (tests, local), they get a different threshold than prod. The local-test harness already hard-codes 0.35. Suggest making the code default match prod (`0.35`) so there's one number to reason about. This was the #1 stale fact in the old docs. See [[12-glossary#Drift between docs and source]].

5. **HIGH_SEV is duplicated in three places** — `alert-manager/main.py`, `dashboard/index.html`, and `local_test/local_alert_handler.py`. A class added to one and not the others silently mis-routes. Hard to DRY across Python + JS without a shared config artifact, but at minimum a comment cross-link in each (some already exist) and a test asserting the Python set matches a checked-in JSON the dashboard imports.

6. **Dead code still shipped.** `services/budget-notifier/main.py` (FCM version) is never deployed but lives next to the real source and confuses every reader; `firebase-admin` is in `alert-manager/requirements.txt` but unused, inflating cold start. Either delete the FCM path or finish wiring it (the dashboard SW is already in place). See [[02-cloud-services#Budget Notifier]], [[06-dashboard#Service worker (`firebase-messaging-sw.js`)]].

7. **`judge-svc` is a manual, not-in-Terraform dependency.** Terraform references it by email in two IAM bindings, so a fresh `apply` fails until someone runs the `gcloud iam service-accounts create judge-svc` step. The cold-start runbook covers it ([[09-deployment-ops]]), but a `null_resource` with a `gcloud` `local-exec` create-if-missing would make `terraform apply` self-sufficient.

## Architecture / reliability (mostly good — nits)

8. **The reliability work is solid.** The dispatcher staleness filter (`MAX_FRAME_AGE_S`) + 404/503-drop + the capture kill-switch together make the old "purge the backlog" ritual largely obsolete. That's a real improvement; the docs now reflect it. One gap: the staleness drop relies on the Pi's clock being roughly in sync with GCP — if the Pi's NTP drifts, *every* frame could read as stale (or never). Worth a one-line note in the Pi setup that NTP must be running.

9. **No Pi heartbeat.** If `frame_extractor.py` dies, `frames-in` goes silent and nothing alerts — the dashboard tiles just go dark. The `pubsub_backlog` monitoring policy catches *too many* messages, not *zero*. A "publish rate < threshold for N minutes" alert would close the loop. Listed as a known gap in [[11-costs-and-monitoring#Monitoring gaps]].

10. **Judge streak state is in-memory and per-replica.** With `min=max=1` it's fine, but it silently resets on every redeploy/restart — a real failure mid-restart needs 2 fresh frames again. Acceptable; just don't be surprised by a missed first alert right after a deploy.

## Docs / repo hygiene

11. **The root markdown files have diverged** — `README.MD`, `documentation.md`, `main_review.md` still describe older behaviour (per-camera cooldown, `spaghetti` spelling, no inference log). The `confs/` vault is now the source of truth; consider either pointing those files at the vault or regenerating them, and deleting stray artifacts (`coomm`, `pre-commit`, `texput.log`, `.coverage` are committed and look accidental).

12. **The vault is the repo.** `confs/` *is* the live Obsidian vault (`.obsidian/` lives under it), so editing the vault edits the repo — convenient. But Obsidian's `.obsidian/workspace` churn will land in git diffs. A `.gitignore` entry for `confs/.obsidian/workspace*` (and `confs/.obsidian/*.json` cache) would keep commits clean.

---

*This file is a vault note; the `[[feedback]]` links in the other notes point here. Update or prune as items get addressed.*
