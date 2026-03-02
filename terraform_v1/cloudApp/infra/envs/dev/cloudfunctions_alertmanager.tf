# ── AlertManager – Cloud Function v2 ──────────────────────────────────────
# Triggered by Pub/Sub "detections-out".
# Writes alert documents to Firestore "alerts" collection.
#
# Prerequisites:
#   • Place your function source in ./functions/alertmanager/
#     (main.py + requirements.txt is enough for Python)
#   • Run:  cd functions/alertmanager && zip -r ../../alertmanager.zip .
#     OR set var.alertmanager_source_dir and let the archive data source do it.
# ──────────────────────────────────────────────────────────────────────────

# Upload function source zip to GCS
resource "google_storage_bucket_object" "alertmanager_zip" {
  name   = "functions/alertmanager-${filemd5(var.alertmanager_zip_path)}.zip"
  bucket = google_storage_bucket.frames.name
  source = var.alertmanager_zip_path
}

resource "google_cloudfunctions2_function" "alertmanager" {
  name     = "alertmanager"
  location = var.region

  build_config {
    runtime     = "python311"
    entry_point = "handle_detection"   # must match your function name

    source {
      storage_source {
        bucket = google_storage_bucket.frames.name
        object = google_storage_bucket_object.alertmanager_zip.name
      }
    }
  }

  service_config {
    service_account_email            = google_service_account.alertmanager.email
    min_instance_count               = 0
    max_instance_count               = 5
    available_memory                 = "256M"
    timeout_seconds                  = 60
    ingress_settings                 = "ALLOW_INTERNAL_ONLY"

    environment_variables = {
      GCP_PROJECT      = var.project_id
      FIRESTORE_DB     = "(default)"
      ALERTS_COLLECTION = "alerts"
    }
  }

  event_trigger {
    trigger_region        = var.region
    event_type            = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic          = google_pubsub_topic.detections_out.id
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.alertmanager.email
  }

  depends_on = [
    google_project_service.cloudfunctions,
    google_project_service.eventarc,
    google_project_service.cloudbuild,
    google_firestore_database.main,
  ]
}

# Allow Eventarc to invoke the function's underlying Cloud Run service
resource "google_cloud_run_v2_service_iam_member" "alertmanager_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloudfunctions2_function.alertmanager.name

  role   = "roles/run.invoker"
  member = "serviceAccount:${google_service_account.alertmanager.email}"
}
