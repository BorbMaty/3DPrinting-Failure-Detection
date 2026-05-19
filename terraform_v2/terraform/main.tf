terraform {
  required_version = ">= 1.6"

  backend "gcs" {
    bucket = "printermonitor-488112-functions-source"
    prefix = "terraform/state"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ── Enable all required GCP APIs ──────────────────────────────────────────────

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "pubsub.googleapis.com",
    "firestore.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "eventarc.googleapis.com",
    "storage.googleapis.com",
    "firebase.googleapis.com",
    "firebaseapphosting.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "billingbudgets.googleapis.com",
    # Required for FCM web push token registration (fixes messaging/token-subscribe-failed)
    "firebaseinstallations.googleapis.com",
    "fcm.googleapis.com",
    "fcmregistrations.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# ── GCS: Model Storage ────────────────────────────────────────────────────────

# ── GCS: Inference Frames ─────────────────────────────────────────────────────
# Stores one JPEG per inference. Public read lets the dashboard display frames
# directly. 7-day lifecycle keeps storage costs low (~few MB/day per camera).

#checkov:skip=CKV_GCP_28:Public read is intentional — frames are non-sensitive
#checkov:skip=CKV_GCP_78:Versioning not needed for transient frame storage
resource "google_storage_bucket" "frames" {
  #checkov:skip=CKV_GCP_28:Public read intentional — dashboard displays inference frames
  #checkov:skip=CKV_GCP_114:Public access prevention disabled intentionally for dashboard
  #checkov:skip=CKV_GCP_78:Versioning not needed for transient frame storage
  #checkov:skip=CKV_GCP_62:Access logs not needed for transient frames
  name                        = "${var.project_id}-frames"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "inherited"
  force_destroy               = true

  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 7 }
  }

  depends_on = [google_project_service.apis]
}

#checkov:skip=CKV_GCP_28:allUsers read is intentional — frames are public for dashboard display
resource "google_storage_bucket_iam_member" "frames_public_read" {
  #checkov:skip=CKV_GCP_28:allUsers read intentional for dashboard
  bucket = google_storage_bucket.frames.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# judge-svc runs the Vertex AI prediction container and uploads frames
resource "google_storage_bucket_iam_member" "frames_judge_svc_write" {
  bucket = google_storage_bucket.frames.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:judge-svc@${var.project_id}.iam.gserviceaccount.com"
}

# Default compute SA also used by the judge container at runtime
resource "google_storage_bucket_iam_member" "frames_compute_sa_write" {
  bucket = google_storage_bucket.frames.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_storage_bucket" "models" {
  name                        = var.model_gcs_bucket
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.apis]
}


# ── Artifact Registry ─────────────────────────────────────────────────────────

#checkov:skip=CKV_GCP_84:CMEK not required for thesis project
resource "google_artifact_registry_repository" "printermonitor" {
  #checkov:skip=CKV_GCP_84:CMEK not required for thesis project
  repository_id = "printermonitor"
  format        = "DOCKER"
  location      = var.region
  description   = "Container images for PrinterMonitor v2"

  depends_on = [google_project_service.apis]
}

# ── Pub/Sub ───────────────────────────────────────────────────────────────────

#checkov:skip=CKV_GCP_83:CMEK not required for thesis project — Google-managed encryption is sufficient
resource "google_pubsub_topic" "frames" {
  #checkov:skip=CKV_GCP_83:CMEK not required for thesis project
  name    = "frames-in"
  project = var.project_id

  message_retention_duration = "3600s"

  depends_on = [google_project_service.apis]
}

#checkov:skip=CKV_GCP_83:CMEK not required for thesis project
resource "google_pubsub_topic" "detections" {
  #checkov:skip=CKV_GCP_83:CMEK not required for thesis project
  name    = "detections-out"
  project = var.project_id

  message_retention_duration = "3600s"

  depends_on = [google_project_service.apis]
}





# ── Firestore ─────────────────────────────────────────────────────────────────

resource "google_firestore_database" "default" {
  project     = var.project_id
  name        = "(default)"
  location_id = "eur3"
  type        = "FIRESTORE_NATIVE"

  lifecycle {
    prevent_destroy = true
    ignore_changes  = all
  }

  depends_on = [google_project_service.apis]
}

resource "google_firestore_index" "alerts_camera_ts" {
  project    = var.project_id
  database   = google_firestore_database.default.name
  collection = "alerts"

  fields {
    field_path = "camera_id"
    order      = "ASCENDING"
  }
  fields {
    field_path = "timestamp"
    order      = "DESCENDING"
  }

  depends_on = [google_firestore_database.default]
}

# ── Service Accounts ──────────────────────────────────────────────────────────

# Note: sa-mediamtx and sa-frame-extractor run on the Raspberry Pi,
# not on Cloud Run. The frame extractor SA key is deployed to the Pi.

resource "google_service_account" "frame_extractor" {
  account_id   = "sa-frame-extractor"
  display_name = "Frame Extractor Service (runs on Pi)"
}

resource "google_service_account" "alert_manager" {
  account_id   = "sa-alert-manager"
  display_name = "AlertManager Cloud Function"
}

# ── IAM Bindings ──────────────────────────────────────────────────────────────

resource "google_pubsub_topic_iam_member" "frame_extractor_publisher" {
  project = var.project_id
  topic   = google_pubsub_topic.frames.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.frame_extractor.email}"
}

resource "google_project_iam_member" "alert_manager_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.alert_manager.email}"
}

# Required for firebase_admin.messaging.send() to authenticate with FCM
resource "google_project_iam_member" "alert_manager_firebase_admin" {
  project = var.project_id
  role    = "roles/firebase.admin"
  member  = "serviceAccount:${google_service_account.alert_manager.email}"
}


resource "google_project_iam_member" "alert_manager_eventarc" {
  project = var.project_id
  role    = "roles/eventarc.eventReceiver"
  member  = "serviceAccount:${google_service_account.alert_manager.email}"
}

resource "google_project_iam_member" "alert_manager_pubsub_subscriber" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.alert_manager.email}"
}

data "google_project" "project" {
  project_id = var.project_id
}

#checkov:skip=CKV_GCP_41:Required for Pub/Sub to authenticate Eventarc triggers to Cloud Run
#checkov:skip=CKV_GCP_49:Required for Pub/Sub to authenticate Eventarc triggers to Cloud Run
resource "google_project_iam_member" "pubsub_sa_token_creator" {
  #checkov:skip=CKV_GCP_41:Required for Pub/Sub Eventarc auth
  #checkov:skip=CKV_GCP_49:Required for Pub/Sub Eventarc auth
  project = var.project_id
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# ── Vertex AI ─────────────────────────────────────────────────────────────────
# Endpoint deployed manually and managed outside Terraform.
# Set var.vertex_endpoint_id after deploying:
#   gcloud ai endpoints list --region=europe-west1 --project=printermonitor-488112

# ── Cloud Function: AlertManager ─────────────────────────────────────────────

resource "google_storage_bucket" "functions_source" {
  name                        = "${var.project_id}-functions-source"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true

  depends_on = [google_project_service.apis]
}

data "archive_file" "alert_manager_source" {
  type        = "zip"
  source_dir  = "${path.module}/../services/alert-manager"
  output_path = "${path.module}/alert-manager.zip"
}

resource "google_storage_bucket_object" "alert_manager_source" {
  name   = "alert-manager-${data.archive_file.alert_manager_source.output_md5}.zip"
  bucket = google_storage_bucket.functions_source.name
  source = data.archive_file.alert_manager_source.output_path
}

resource "google_cloudfunctions2_function" "alert_manager" {
  name     = "alert-manager"
  location = var.region

  build_config {
    runtime     = "python312"
    entry_point = "handle_detection"

    source {
      storage_source {
        bucket = google_storage_bucket.functions_source.name
        object = google_storage_bucket_object.alert_manager_source.name
      }
    }
  }

  service_config {
    service_account_email = google_service_account.alert_manager.email
    max_instance_count    = 10
    available_memory      = "256M"
    timeout_seconds       = 120
    ingress_settings      = "ALLOW_INTERNAL_ONLY"

    environment_variables = {
      GCP_PROJECT          = var.project_id
      FIRESTORE_COLLECTION = "alerts"
      CONF_THRESHOLD       = var.conf_threshold
      GMAIL_ADDRESS        = var.gmail_address
      GMAIL_APP_PASSWORD   = var.gmail_app_password
    }
  }

  event_trigger {
    trigger_region        = var.region
    event_type            = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic          = google_pubsub_topic.detections.id
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.alert_manager.email
  }

  depends_on = [
    google_project_service.apis,
    google_storage_bucket_object.alert_manager_source,
    google_firestore_database.default,
    google_pubsub_topic.detections,
  ]
}

# ── Pub/Sub: Budget notifications ─────────────────────────────────────────────

#checkov:skip=CKV_GCP_83:CMEK not required for thesis project
resource "google_pubsub_topic" "budget_notifications" {
  #checkov:skip=CKV_GCP_83:CMEK not required for thesis project
  name    = "budget-notifications"
  project = var.project_id

  depends_on = [google_project_service.apis]
}

resource "google_pubsub_subscription" "budget_notifications_sub" {
  name    = "budget-notifications-sub"
  topic   = google_pubsub_topic.budget_notifications.name
  project = var.project_id

  ack_deadline_seconds = 60
}

# IAM for billing budget publisher granted manually

# ── Billing Budget ─────────────────────────────────────────────────────────────
# Managed manually via GCP Console — CI service account lacks
# billingAccounts.budgets permissions. Budget ID: 695e9489-62cb-4e37-82cc-2d0e1ed4c8e0
# To bring back under Terraform, grant roles/billing.costsManager to the CI SA.

# ── Cloud Function: Budget Notifier ───────────────────────────────────────────

data "archive_file" "budget_notifier_source" {
  type = "zip"
  # Both handlers (handle_detection + handle_budget_alert) live in the same
  # services/alert-manager directory. Using that as source for both functions.
  source_dir  = "${path.module}/../services/alert-manager"
  output_path = "${path.module}/budget-notifier.zip"
}

resource "google_storage_bucket_object" "budget_notifier_source" {
  name   = "budget-notifier-${data.archive_file.budget_notifier_source.output_md5}.zip"
  bucket = google_storage_bucket.functions_source.name
  source = data.archive_file.budget_notifier_source.output_path
}

resource "google_cloudfunctions2_function" "budget_notifier" {
  name     = "budget-notifier"
  location = var.region

  build_config {
    runtime     = "python312"
    entry_point = "handle_budget_alert"

    source {
      storage_source {
        bucket = google_storage_bucket.functions_source.name
        object = google_storage_bucket_object.budget_notifier_source.name
      }
    }
  }

  service_config {
    service_account_email = google_service_account.alert_manager.email
    max_instance_count    = 3
    available_memory      = "256M"
    timeout_seconds       = 60
    ingress_settings      = "ALLOW_INTERNAL_ONLY"

    environment_variables = {
      GCP_PROJECT        = var.project_id
      GMAIL_ADDRESS      = var.gmail_address
      GMAIL_APP_PASSWORD = var.gmail_app_password
    }
  }

  event_trigger {
    trigger_region        = var.region
    event_type            = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic          = google_pubsub_topic.budget_notifications.id
    retry_policy          = "RETRY_POLICY_DO_NOT_RETRY"
    service_account_email = google_service_account.alert_manager.email
  }

  depends_on = [
    google_project_service.apis,
    google_storage_bucket_object.budget_notifier_source,
    google_pubsub_topic.budget_notifications,
  ]
}

locals {
  # Once var.cloudflare_tunnel_hostname is set, this becomes your stable WebRTC host
  mediamtx_host = var.cloudflare_tunnel_hostname != "" ? var.cloudflare_tunnel_hostname : "PENDING-SET-cloudflare_tunnel_hostname-variable"
}

resource "google_project_iam_member" "frame_extractor_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.frame_extractor.email}"
}

# ══════════════════════════════════════════════════════════════════════════════
# ADDITION: Dispatcher — bridges frames-in → Vertex AI judge
# ══════════════════════════════════════════════════════════════════════════════

# ── Service Account ───────────────────────────────────────────────────────────

resource "google_service_account" "dispatcher" {
  account_id   = "sa-dispatcher"
  display_name = "Dispatcher Cloud Function (frames-in → Vertex AI)"
}

resource "google_project_iam_member" "dispatcher_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.dispatcher.email}"
}

resource "google_project_iam_member" "dispatcher_eventarc" {
  project = var.project_id
  role    = "roles/eventarc.eventReceiver"
  member  = "serviceAccount:${google_service_account.dispatcher.email}"
}

resource "google_project_iam_member" "dispatcher_pubsub_subscriber" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.dispatcher.email}"
}

# Dispatcher → publish detections-out (results from Vertex AI)
resource "google_pubsub_topic_iam_member" "dispatcher_detections_publisher" {
  project = var.project_id
  topic   = google_pubsub_topic.detections.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.dispatcher.email}"
}

# Vertex AI service agent → publish detections-out
resource "google_pubsub_topic_iam_member" "vertex_ai_detections_publisher" {
  project = var.project_id
  topic   = google_pubsub_topic.detections.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-aiplatform.iam.gserviceaccount.com"
}

# Vertex AI custom code service agent → publish detections-out
resource "google_pubsub_topic_iam_member" "vertex_ai_cc_detections_publisher" {
  project = var.project_id
  topic   = google_pubsub_topic.detections.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"
}

# Allow Vertex AI service agent to impersonate judge-svc when running prediction container
resource "google_service_account_iam_member" "vertex_ai_judge_svc_user" {
  service_account_id = "projects/${var.project_id}/serviceAccounts/judge-svc@${var.project_id}.iam.gserviceaccount.com"
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-aiplatform.iam.gserviceaccount.com"
}

# Default compute SA used by judge custom prediction container → publish detections-out
resource "google_pubsub_topic_iam_member" "compute_sa_detections_publisher" {
  project = var.project_id
  topic   = google_pubsub_topic.detections.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

# ── Source bundle ─────────────────────────────────────────────────────────────

data "archive_file" "dispatcher_source" {
  type        = "zip"
  source_dir  = "${path.module}/../services/dispatcher"
  output_path = "${path.module}/dispatcher.zip"
}

resource "google_storage_bucket_object" "dispatcher_source" {
  name   = "dispatcher-${data.archive_file.dispatcher_source.output_md5}.zip"
  bucket = google_storage_bucket.functions_source.name
  source = data.archive_file.dispatcher_source.output_path
}

# ── Cloud Function ────────────────────────────────────────────────────────────

resource "google_cloudfunctions2_function" "dispatcher" {
  name     = "dispatcher"
  location = var.region

  build_config {
    runtime     = "python312"
    entry_point = "dispatch_frame"

    source {
      storage_source {
        bucket = google_storage_bucket.functions_source.name
        object = google_storage_bucket_object.dispatcher_source.name
      }
    }
  }

  service_config {
    service_account_email = google_service_account.dispatcher.email
    max_instance_count    = 10
    available_memory      = "512M" # frames are ~100-200KB base64
    timeout_seconds       = 60
    ingress_settings      = "ALLOW_INTERNAL_ONLY"

    environment_variables = {
      GCP_PROJECT        = var.project_id
      VERTEX_ENDPOINT_ID = var.vertex_endpoint_id
      VERTEX_REGION      = var.region
    }
  }

  event_trigger {
    trigger_region        = var.region
    event_type            = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic          = google_pubsub_topic.frames.id
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.dispatcher.email
  }

  depends_on = [
    google_project_service.apis,
    google_storage_bucket_object.dispatcher_source,
    google_pubsub_topic.frames,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "dispatcher_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloudfunctions2_function.dispatcher.service_config[0].service
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.dispatcher.email}"
}

resource "google_cloud_run_v2_service_iam_member" "alert_manager_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloudfunctions2_function.alert_manager.service_config[0].service
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.alert_manager.email}"
}

resource "google_cloud_run_v2_service_iam_member" "budget_notifier_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloudfunctions2_function.budget_notifier.service_config[0].service
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.alert_manager.email}"
}

# ── Dead-letter queues ────────────────────────────────────────────────────────
# Catch poison-pill messages (corrupt frames / unhandled exceptions) that exhaust
# their retry budget so they don't block the Eventarc subscription indefinitely.
# Covered pipelines: frames-in → dispatcher, detections-out → alert-manager.
# budget-notifier uses RETRY_POLICY_DO_NOT_RETRY so DLQ is not needed there.

#checkov:skip=CKV_GCP_83:CMEK not required for thesis project
resource "google_pubsub_topic" "frames_dead_letters" {
  #checkov:skip=CKV_GCP_83:CMEK not required for thesis project
  name    = "frames-in-dead-letters"
  project = var.project_id

  depends_on = [google_project_service.apis]
}

#checkov:skip=CKV_GCP_83:CMEK not required for thesis project
resource "google_pubsub_topic" "detections_dead_letters" {
  #checkov:skip=CKV_GCP_83:CMEK not required for thesis project
  name    = "detections-out-dead-letters"
  project = var.project_id

  depends_on = [google_project_service.apis]
}

# Pub/Sub service agent must be able to publish to DLQ topics when forwarding
# undeliverable messages.
resource "google_pubsub_topic_iam_member" "frames_dlq_publisher" {
  project = var.project_id
  topic   = google_pubsub_topic.frames_dead_letters.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

resource "google_pubsub_topic_iam_member" "detections_dlq_publisher" {
  project = var.project_id
  topic   = google_pubsub_topic.detections_dead_letters.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# Pull subscriptions on the DLQ topics — keeps dead-lettered messages for 7 days
# so they can be inspected instead of silently expiring.
resource "google_pubsub_subscription" "frames_dead_letters_sub" {
  name    = "frames-in-dead-letters-sub"
  topic   = google_pubsub_topic.frames_dead_letters.name
  project = var.project_id

  message_retention_duration = "604800s"
  ack_deadline_seconds       = 60
}

resource "google_pubsub_subscription" "detections_dead_letters_sub" {
  name    = "detections-out-dead-letters-sub"
  topic   = google_pubsub_topic.detections_dead_letters.name
  project = var.project_id

  message_retention_duration = "604800s"
  ack_deadline_seconds       = 60
}

# Wire the dead-letter policy to the Eventarc-managed subscriptions.
# Eventarc creates and owns these subscriptions, so Terraform cannot set
# dead_letter_policy on them declaratively — null_resource + local-exec is the
# standard workaround. Triggers re-run if the DLQ topic is replaced.
resource "null_resource" "dispatcher_dlq" {
  triggers = {
    dlq_topic_id = google_pubsub_topic.frames_dead_letters.id
  }

  provisioner "local-exec" {
    command = <<-EOT
      gcloud pubsub subscriptions update \
        eventarc-europe-west1-dispatcher-635712-sub-152 \
        --dead-letter-topic=${google_pubsub_topic.frames_dead_letters.id} \
        --max-delivery-attempts=5 \
        --project=${var.project_id}
    EOT
  }

  depends_on = [
    google_pubsub_topic.frames_dead_letters,
    google_pubsub_topic_iam_member.frames_dlq_publisher,
  ]
}

resource "null_resource" "alert_manager_dlq" {
  triggers = {
    dlq_topic_id = google_pubsub_topic.detections_dead_letters.id
  }

  provisioner "local-exec" {
    command = <<-EOT
      gcloud pubsub subscriptions update \
        eventarc-europe-west1-alert-manager-030900-sub-669 \
        --dead-letter-topic=${google_pubsub_topic.detections_dead_letters.id} \
        --max-delivery-attempts=5 \
        --project=${var.project_id}
    EOT
  }

  depends_on = [
    google_pubsub_topic.detections_dead_letters,
    google_pubsub_topic_iam_member.detections_dlq_publisher,
  ]
}

# ── Cloud Monitoring ──────────────────────────────────────────────────────────
# Two alerting policies: Cloud Function 5xx spike and Pub/Sub backlog stall.
# Notifications go to the same Gmail address used for defect alerts.

resource "google_monitoring_notification_channel" "email_alerts" {
  display_name = "PrinterMonitor Email Alerts"
  type         = "email"
  project      = var.project_id

  labels = {
    email_address = var.gmail_address
  }
}

# Alert when any Cloud Run service (all Gen2 functions run on Cloud Run) returns
# more than 3 server errors in a 5-minute window — indicates a crashing function.
resource "google_monitoring_alert_policy" "function_errors" {
  display_name = "PrinterMonitor: Cloud Function Errors"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "5xx responses > 3 in 5 min"
    condition_threshold {
      filter = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\""

      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 3

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email_alerts.name]

  alert_strategy {
    auto_close = "1800s"
  }

  depends_on = [google_monitoring_notification_channel.email_alerts]
}

# Alert when a Pub/Sub message sits undelivered for more than 60 seconds on either
# of the two main pipeline subscriptions — indicates the dispatcher or alert-manager
# is down, crashed, or the Vertex AI endpoint is unavailable.
resource "google_monitoring_alert_policy" "pubsub_backlog" {
  display_name = "PrinterMonitor: Pub/Sub Backlog Stalled"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "oldest_undelivered_message_age > 60s"
    condition_threshold {
      filter = "resource.type=\"pubsub_subscription\" AND metric.type=\"pubsub.googleapis.com/subscription/oldest_undelivered_message_age\" AND resource.labels.subscription_id=~\"eventarc-europe-west1-(dispatcher|alert-manager).*\""

      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 60

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MAX"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email_alerts.name]

  alert_strategy {
    auto_close = "1800s"
  }

  depends_on = [google_monitoring_notification_channel.email_alerts]
}