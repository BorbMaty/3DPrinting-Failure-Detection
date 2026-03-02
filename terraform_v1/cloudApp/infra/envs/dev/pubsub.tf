resource "google_pubsub_topic" "frames_in" {
  name = "frames-in"
}

resource "google_pubsub_topic" "detections_out" {
  name = "detections-out"
}