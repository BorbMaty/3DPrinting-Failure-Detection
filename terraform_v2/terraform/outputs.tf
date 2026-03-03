output "mediamtx_whep_url" {
  description = "WebRTC WHEP URL (Cloudflare tunnel on Pi)"
  value       = "https://<cloudflare-tunnel-url>/whep"
}

output "vertex_endpoint_id" {
  description = "Vertex AI Endpoint ID for the Judge Service"
  value       = "9105488997194399744"
}

output "frames_topic" {
  description = "Pub/Sub frames-in topic name"
  value       = google_pubsub_topic.frames.name
}

output "detections_topic" {
  description = "Pub/Sub detections-out topic name"
  value       = google_pubsub_topic.detections.name
}

output "model_gcs_uri" {
  description = "GCS URI of the uploaded YOLOv8x model"
  value       = "gs://${google_storage_bucket.models.name}/${google_storage_bucket_object.model.name}"
}

output "artifact_registry" {
  description = "Docker image registry prefix"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/printermonitor"
}

output "firestore_database" {
  description = "Firestore database name"
  value       = google_firestore_database.default.name
}

output "alert_manager_function" {
  description = "AlertManager Cloud Function name"
  value       = google_cloudfunctions2_function.alert_manager.name
}
