resource "google_storage_bucket" "frames" {
  name                        = "${var.project_id}-frames"
  location                    = upper(var.region) # pl. EUROPE-WEST1
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }
}