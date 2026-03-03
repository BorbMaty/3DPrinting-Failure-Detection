output "mediamtx_url" {
  description = "MediaMTX Cloud Run URL (WebRTC WHEP base)"
  value       = google_cloud_run_v2_service.mediamtx.uri
}

output "mediamtx_whep_url" {
  description = "Full WHEP endpoint for browser WebRTC player"
  value       = google_cloud_run_v2_service.mediamtx.uri != null ? "${google_cloud_run_v2_service.mediamtx.uri}/whep" : ""
}

output "frame_extractor_url" {
  description = "Frame Extractor Cloud Run URL"
  value       = google_cloud_run_v2_service.frame_extractor.uri
}

output "vertex_endpoint_id" {
  description = "Vertex AI Endpoint ID for the Judge Service"
  value       = google_vertex_ai_endpoint.judge.id
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