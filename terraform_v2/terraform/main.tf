terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
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

resource "google_pubsub_subscription" "frames_push" {
  name    = "frames-in-judge-push"
  topic   = google_pubsub_topic.frames.name
  project = var.project_id

  ack_deadline_seconds = 60

  push_config {
    push_endpoint = "https://judge-placeholder.run.app/predict"
  }

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.frames_dead_letter.id
    max_delivery_attempts = 5
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "300s"
  }

  depends_on = [google_pubsub_topic.frames]
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

resource "google_service_account" "mediamtx" {
  account_id   = "sa-mediamtx"
  display_name = "MediaMTX RTSP/WebRTC Service"
}

resource "google_service_account" "frame_extractor" {
  account_id   = "sa-frame-extractor"
  display_name = "Frame Extractor Service"
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

# ── Cloud Run: MediaMTX ───────────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "mediamtx" {
  provider = google-beta
  name     = "mediamtx"
  location = var.region

  template {
    service_account = google_service_account.mediamtx.email

    scaling {
      min_instance_count = 1
      max_instance_count = 3
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/printermonitor/mediamtx:latest"

      env {
        name  = "MTX_PROTOCOLS"
        value = "tcp"
      }
      env {
        name  = "MTX_WEBRTCADDRESS"
        value = ":8080"
      }
      env {
        name  = "MTX_LOGLEVEL"
        value = "info"
      }

      ports {
        name           = "http1"
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle          = false
        startup_cpu_boost = true
      }

      liveness_probe {
        http_get { path = "/" }
        initial_delay_seconds = 10
        period_seconds        = 30
        failure_threshold     = 3
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.apis,
    google_artifact_registry_repository.printermonitor,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "mediamtx_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.mediamtx.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Cloud Run: Frame Extractor ────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "frame_extractor" {
  provider = google-beta
  name     = "frame-extractor"
  location = var.region

  template {
    service_account = google_service_account.frame_extractor.email

    scaling {
      min_instance_count = 1
      max_instance_count = 1
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/printermonitor/frame-extractor:latest"

      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }
      env {
        name  = "FRAMES_TOPIC"
        value = google_pubsub_topic.frames.name
      }
      env {
        name  = "RTSP_URL"
        value = "rtsp://100.73.126.106:8554/cam1"
      }
      env {
        name  = "CAPTURE_FPS"
        value = var.capture_fps
      }
      env {
        name  = "JPEG_QUALITY"
        value = "70"
      }
      env {
        name  = "FRAME_WIDTH"
        value = "640"
      }
      env {
        name  = "FRAME_HEIGHT"
        value = "480"
      }

      ports {
        name           = "http1"
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle          = false
        startup_cpu_boost = true
      }

      liveness_probe {
        http_get { path = "/healthz" }
        initial_delay_seconds = 15
        period_seconds        = 30
        failure_threshold     = 3
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.apis,
    google_artifact_registry_repository.printermonitor,
    google_pubsub_topic.frames,
  ]
}

# ── Vertex AI ─────────────────────────────────────────────────────────────────

resource "google_vertex_ai_endpoint" "judge" {
  name         = "judge-endpoint"
  display_name = "PrinterMonitor Judge Endpoint"
  location     = var.region

  depends_on = [google_project_service.apis]
}


resource "null_resource" "judge_model_deploy" {
  triggers = {
    image = "${var.region}-docker.pkg.dev/${var.project_id}/printermonitor/judge:latest"
  }

  provisioner "local-exec" {
    command = <<-EOT
      MODEL_ID=$(gcloud ai models upload \
        --region=${var.region} \
        --display-name=yolov8x-printermonitor \
        --container-image-uri=${var.region}-docker.pkg.dev/${var.project_id}/printermonitor/judge:latest \
        --container-predict-route=/predict \
        --container-health-route=/healthz \
        --container-ports=8080 \
        --container-env-vars=GCP_PROJECT=${var.project_id},DETECTIONS_TOPIC=detections-out,MODEL_PATH=/app/best.pt,CONF_THRESHOLD=${var.conf_threshold} \
        --format="value(model)" \
        --project=${var.project_id} || true)
      gcloud ai endpoints deploy-model ${google_vertex_ai_endpoint.judge.name} \
        --region=${var.region} \
        --model=$MODEL_ID \
        --display-name=judge-deployed \
        --machine-type=n1-standard-4 \
        --accelerator=type=nvidia-tesla-t4,count=1 \
        --min-replica-count=0 \
        --max-replica-count=2 \
        --project=${var.project_id} || true
    EOT
  }

  depends_on = [
    google_vertex_ai_endpoint.judge,
    google_artifact_registry_repository.printermonitor,
  ]
}

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
    timeout_seconds       = 60

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
