variable "project_id" {
  type        = string
  description = "Your Google Cloud project ID"
}

variable "region" {
  type        = string
  description = "Default region"
  default     = "europe-west1"
}

variable "zone" {
  type        = string
  description = "Compute zone (must be inside the chosen region)"
  default     = "europe-west1-c"
}

variable "image_mediamtx" {
  description = "Docker image URI for MediaMTX"
  type        = string
}

variable "image_extractor" {
  description = "Docker image URI for frame extractor"
  type        = string
}

variable "image_judge" {
  description = "Docker image URI for judge (GPU)"
  type        = string
}

variable "mediamtx_base_url" {
  type        = string
  description = "Base URL of the MediaMTX Cloud Run service, e.g. https://mediamtx-xxxxx.europe-west1.run.app"
}

variable "extractor_stream_url" {
  type        = string
  description = "Snapshot/JPEG URL for the frame extractor to pull from."
}

variable "alertmanager_zip_path" {
  type        = string
  description = "Local path to the alertmanager function zip, e.g. ./alertmanager.zip"
  default     = "./alertmanager.zip"
}