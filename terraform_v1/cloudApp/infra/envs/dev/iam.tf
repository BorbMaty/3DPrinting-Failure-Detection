# ── Service Accounts ──────────────────────────────────────────────────────

resource "google_service_account" "edge_pi" {
  account_id   = "edge-pi"
  display_name = "Edge Raspberry Pi"
}

resource "google_service_account" "inference" {
  account_id   = "inference-svc"
  display_name = "Cloud Inference Service (GPU VM)"
}

resource "google_service_account" "mediamtx" {
  account_id   = "mediamtx-svc"
  display_name = "MediaMTX Cloud Run"
}

resource "google_pubsub_topic_iam_member" "mediamtx_pubsub_publisher" {
  topic  = google_pubsub_topic.frames_in.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.mediamtx.email}"
}

resource "google_service_account" "extractor" {
  account_id   = "extractor-svc"
  display_name = "Frame Extractor Cloud Run"
}

resource "google_service_account" "judge" {
  account_id   = "judge-svc"
  display_name = "Judge Service (GPU)"
}

resource "google_service_account" "alertmanager" {
  account_id   = "alertmanager-svc"
  display_name = "AlertManager Cloud Function"
}

# ── Edge Pi permissions ────────────────────────────────────────────────────

resource "google_storage_bucket_iam_member" "edge_pi_bucket_writer" {
  bucket = google_storage_bucket.frames.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.edge_pi.email}"
}

resource "google_pubsub_topic_iam_member" "edge_pi_pubsub_publisher" {
  topic  = google_pubsub_topic.frames_in.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.edge_pi.email}"
}

# ── Inference VM permissions ───────────────────────────────────────────────

resource "google_storage_bucket_iam_member" "inference_bucket_reader" {
  bucket = google_storage_bucket.frames.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.inference.email}"
}

resource "google_storage_bucket_iam_member" "inference_bucket_writer" {
  bucket = google_storage_bucket.frames.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.inference.email}"
}

resource "google_pubsub_topic_iam_member" "inference_pubsub_publisher" {
  topic  = google_pubsub_topic.detections_out.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.inference.email}"
}

resource "google_pubsub_topic_iam_member" "inference_pubsub_subscriber" {
  topic  = google_pubsub_topic.frames_in.name
  role   = "roles/pubsub.subscriber"
  member = "serviceAccount:${google_service_account.inference.email}"
}

# ── Extractor permissions ──────────────────────────────────────────────────

# Extractor publishes frames to frames-in
resource "google_pubsub_topic_iam_member" "extractor_pubsub_publisher_frames" {
  topic  = google_pubsub_topic.frames_in.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.extractor.email}"
}

# ── Judge permissions ──────────────────────────────────────────────────────

# Pub/Sub push subscription delivers to judge; judge needs subscriber role
# on the topic so it can ack messages via the push endpoint
resource "google_pubsub_topic_iam_member" "judge_pubsub_subscriber_frames" {
  topic  = google_pubsub_topic.frames_in.name
  role   = "roles/pubsub.subscriber"
  member = "serviceAccount:${google_service_account.judge.email}"
}

# Judge publishes detections
resource "google_pubsub_topic_iam_member" "judge_pubsub_publisher_detections" {
  topic  = google_pubsub_topic.detections_out.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.judge.email}"
}

# ── AlertManager permissions ───────────────────────────────────────────────

# AlertManager subscribes to detections (Eventarc handles sub creation,
# but the SA still needs subscriber rights)
resource "google_pubsub_topic_iam_member" "alert_pubsub_subscriber_detections" {
  topic  = google_pubsub_topic.detections_out.name
  role   = "roles/pubsub.subscriber"
  member = "serviceAccount:${google_service_account.alertmanager.email}"
}

# AlertManager writes to Firestore
resource "google_project_iam_member" "alert_firestore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.alertmanager.email}"
}

# Eventarc needs to invoke the Cloud Function's underlying Cloud Run service
# The alertmanager SA is used as the trigger's OIDC identity
resource "google_project_iam_member" "alertmanager_eventarc_receiver" {
  project = var.project_id
  role    = "roles/eventarc.eventReceiver"
  member  = "serviceAccount:${google_service_account.alertmanager.email}"
}

# Eventarc service agent needs to publish to Pub/Sub (for trigger plumbing)
data "google_project" "project" {
  project_id = var.project_id
}

resource "google_project_iam_member" "eventarc_sa_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-eventarc.iam.gserviceaccount.com"
}

resource "google_cloud_run_v2_service_iam_member" "judge_sa_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.judge.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.judge.email}"
}

resource "google_project_iam_member" "inference_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.inference.email}"
}

resource "google_project_iam_member" "inference_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.inference.email}"
}

resource "google_pubsub_topic_iam_member" "inference_frames_publisher" {
  topic  = google_pubsub_topic.frames_in.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.inference.email}"
}