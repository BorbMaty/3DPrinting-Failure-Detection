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

resource "google_storage_bucket" "models" {
  name                        = var.model_gcs_bucket
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket_object" "model" {
  name   = var.model_gcs_path
  bucket = google_storage_bucket.models.name
  source = "/home/lucy/Desktop/datasetmaybefinal/yolov8x_finetune_weighted/weights/best.pt"

  depends_on = [google_storage_bucket.models]
}

# ── Artifact Registry ─────────────────────────────────────────────────────────

resource "google_artifact_registry_repository" "printermonitor" {
  repository_id = "printermonitor"
  format        = "DOCKER"
  location      = var.region
  description   = "Container images for PrinterMonitor v2"

  depends_on = [google_project_service.apis]
}

# ── Pub/Sub ───────────────────────────────────────────────────────────────────

resource "google_pubsub_topic" "frames" {
  name    = "frames-in"
  project = var.project_id

  message_retention_duration = "3600s"

  depends_on = [google_project_service.apis]
}

resource "google_pubsub_topic" "detections" {
  name    = "detections-out"
  project = var.project_id

  message_retention_duration = "3600s"

  depends_on = [google_project_service.apis]
}

resource "google_pubsub_topic" "frames_dead_letter" {
  name    = "frames-in-dead-letter"
  project = var.project_id

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

resource "google_service_account" "judge" {
  account_id   = "sa-judge"
  display_name = "Judge Service (Vertex AI)"
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

resource "google_pubsub_topic_iam_member" "judge_publisher" {
  project = var.project_id
  topic   = google_pubsub_topic.detections.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.judge.email}"
}

resource "google_storage_bucket_iam_member" "judge_model_reader" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.judge.email}"
}

resource "google_project_iam_member" "alert_manager_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.alert_manager.email}"
}

# Required for firebase_admin.messaging.send() to authenticate with FCM
resource "google_project_iam_member" "alert_manager_firebase_admin" {
  project = var.project_id
  role    = "roles/firebase.sdkAdminServiceAgent"
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

resource "google_project_iam_member" "pubsub_sa_token_creator" {
  project = var.project_id
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# ── Vertex AI ─────────────────────────────────────────────────────────────────
# Endpoint deployed manually:
#   Endpoint ID : 9105488997194399744
#   Model ID    : 7050633706276388864
#   Machine type: n1-standard-4
# Managed outside Terraform to avoid accidental destruction.

# ── Cloud Function: AlertManager ─────────────────────────────────────────────

resource "google_storage_bucket" "functions_source" {
  name                        = "${var.project_id}-functions-source"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  depends_on = [google_project_service.apis]
}

data "archive_file" "alert_manager_source" {
  type        = "zip"
  source_dir  = "${path.module}/../services/alert-manager"
  output_path = "/tmp/alert-manager.zip"
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

    environment_variables = {
      GCP_PROJECT          = var.project_id
      FIRESTORE_COLLECTION = "alerts"
      CONF_THRESHOLD       = var.conf_threshold
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

resource "google_pubsub_topic" "budget_notifications" {
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

resource "google_billing_budget" "monthly" {
  billing_account = "0124C1-D405D9-F46DB8"
  display_name    = "PrinterMonitor $5 Alert"

  budget_filter {
    projects = ["projects/895714392909"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = "5"
    }
  }

  threshold_rules {
    threshold_percent = 0.5 # alert at $2.50
    spend_basis       = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 1.0 # alert at $5.00
    spend_basis       = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 1.2 # alert at $6.00 (overage)
    spend_basis       = "CURRENT_SPEND"
  }

  all_updates_rule {
    pubsub_topic                     = google_pubsub_topic.budget_notifications.id
    schema_version                   = "1.0"
    monitoring_notification_channels = []
  }
}

# ── Cloud Function: Budget Notifier ───────────────────────────────────────────

data "archive_file" "budget_notifier_source" {
  type = "zip"
  # Both handlers (handle_detection + handle_budget_alert) live in the same
  # services/alert-manager directory. Using that as source for both functions.
  source_dir  = "${path.module}/../services/alert-manager"
  output_path = "/tmp/budget-notifier.zip"
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

    environment_variables = {
      GCP_PROJECT = var.project_id
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

# ── Source bundle ─────────────────────────────────────────────────────────────

data "archive_file" "dispatcher_source" {
  type        = "zip"
  source_dir  = "${path.module}/../services/dispatcher"
  output_path = "/tmp/dispatcher.zip"
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