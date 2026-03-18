output "mediamtx_whep_url" {
  description = "WebRTC WHEP URL (Cloudflare tunnel on Pi)"
  value       = "https://${local.mediamtx_host}/whep"
}

output "vertex_endpoint_id" {
  description = "Vertex AI Endpoint ID for the Judge Service"
  value       = var.vertex_endpoint_id
}

output "frames_topic" {
  description = "Pub/Sub frames-in topic name"
  value       = google_pubsub_topic.frames.name
}

output "detections_topic" {
  description = "Pub/Sub detections-out topic name"
  value       = google_pubsub_topic.detections.name
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

output "mediamtx_host" {
  description = "Stable Cloudflare tunnel hostname for WebRTC (set cloudflare_tunnel_hostname variable)"
  value       = local.mediamtx_host
}

output "dashboard_host_update_command" {
  description = "One-liner to update the HOST in index.html after tunnel hostname is set"
  value       = "sed -i 's|window.MEDIAMTX_HOST || \".*\"|window.MEDIAMTX_HOST || \"${local.mediamtx_host}\"|' index.html"
}

output "dispatcher_function" {
  description = "Dispatcher Cloud Function name"
  value       = google_cloudfunctions2_function.dispatcher.name
}