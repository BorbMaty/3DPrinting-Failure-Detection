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

variable "model_gcs_path" {
  description = "Path inside the bucket where best.pt lives"
  type        = string
  default     = "yolov8x/best.pt"
}

variable "conf_threshold" {
  description = "YOLO confidence threshold for detections"
  type        = string
  default     = "0.35"
}

variable "capture_fps" {
  description = "Frames per second the frame extractor captures"
  type        = number
  default     = 2
}
variable "cloudflare_tunnel_token" {
  description = "Cloudflare Tunnel token (from cloudflared tunnel create printermonitor). Set via TF_VAR_cloudflare_tunnel_token env var or terraform.tfvars. Leave empty until permanent tunnel is created."
  type        = string
  default     = ""
  sensitive   = true
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