
resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = "printer"
  format        = "DOCKER"
}