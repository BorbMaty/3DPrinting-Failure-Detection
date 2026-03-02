resource "google_cloud_run_v2_service" "judge" {
  name         = "judge"
  location     = var.region
  # Pub/Sub push uses OIDC — no need for public ingress
  ingress      = "INGRESS_TRAFFIC_INTERNAL_ONLY"
  # GPU workloads are still in BETA launch stage on Cloud Run
  launch_stage = "BETA"

  deletion_protection = false

  template {
    service_account = google_service_account.judge.email

    # Required for Cloud Run GPU — tells the scheduler to place on an L4 node
    node_selector {
      accelerator = "nvidia-l4"
    }
    gpu_zonal_redundancy_disabled = true

    containers {
      image = var.image_judge

      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }

      env {
        name  = "DETECTIONS_TOPIC"
        value = google_pubsub_topic.detections_out.name
      }

      resources {
        limits = {
          cpu              = "4"
          memory           = "16Gi"
          "nvidia.com/gpu" = "1"
        }
        # Keep CPU allocated while processing — important for GPU workloads
        cpu_idle = false
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 1  # GPU quota is tight; keep to 1
    }

    # GPU cold-starts are slow; give it time
    timeout = "300s"

    max_instance_request_concurrency = 1  # 1 GPU = 1 request at a time
  }

  lifecycle {
    ignore_changes = [template[0].scaling]
  }
}