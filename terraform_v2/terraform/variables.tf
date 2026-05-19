variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "printermonitor-488112"
}

variable "region" {
  description = "Primary GCP region"
  type        = string
  default     = "europe-west1"
}

variable "model_gcs_bucket" {
  description = "GCS bucket to store the YOLOv8x model"
  type        = string
  default     = "printermonitor-488112-models"
}


variable "conf_threshold" {
  description = "YOLO confidence threshold for detections"
  type        = string
  default     = "0.35"
}


variable "cloudflare_tunnel_hostname" {
  description = "Permanent hostname for the Cloudflare tunnel (e.g. printermonitor.yourdomain.com). Set after creating a Named Tunnel."
  type        = string
  default     = ""
}

variable "vertex_endpoint_id" {
  description = "Vertex AI Endpoint ID for the Judge service"
  type        = string
  default     = "6900414029643120640"
}
variable "gmail_address" {
  description = "Gmail address for alert notifications"
  type        = string
  sensitive   = true
}

variable "gmail_app_password" {
  description = "Gmail app password for alert notifications"
  type        = string
  sensitive   = true
}

