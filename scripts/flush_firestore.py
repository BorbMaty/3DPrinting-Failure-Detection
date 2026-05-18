"""
Delete all documents from Firestore collections used by the alert pipeline.
Run with: python scripts/flush_firestore.py [--collections alerts cooldowns budget_alerts]
Requires: google-cloud-firestore, GCP credentials (gcloud auth application-default login)
"""
import argparse
from google.cloud import firestore

PROJECT_ID = "printermonitor-488112"

COLLECTIONS = {
    "alerts":         "alerts",
    "cooldowns":      "alert_cooldowns",
    "budget_alerts":  "budget_alerts",
    "inferences":     "inferences",
}


def delete_collection(db: firestore.Client, col_name: str, batch_size: int = 100):
    col_ref = db.collection(col_name)
    deleted = 0
    while True:
        docs = list(col_ref.limit(batch_size).stream())
        if not docs:
            break
        batch = db.batch()
        for doc in docs:
            batch.delete(doc.reference)
        batch.commit()
        deleted += len(docs)
        print(f"  deleted {deleted} docs from '{col_name}'...")
    print(f"  '{col_name}' flushed — {deleted} total docs removed.")
    return deleted


def main():
    parser = argparse.ArgumentParser(description="Flush Firestore collections")
    parser.add_argument(
        "--collections",
        nargs="+",
        choices=list(COLLECTIONS.keys()),
        default=list(COLLECTIONS.keys()),
        help="Which collections to flush (default: all)",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    targets = {k: COLLECTIONS[k] for k in args.collections}
    print(f"Will delete ALL documents from: {list(targets.values())}")
    if not args.yes:
        confirm = input("Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    db = firestore.Client(project=PROJECT_ID)
    for label, col_name in targets.items():
        print(f"\nFlushing '{col_name}'...")
        delete_collection(db, col_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
