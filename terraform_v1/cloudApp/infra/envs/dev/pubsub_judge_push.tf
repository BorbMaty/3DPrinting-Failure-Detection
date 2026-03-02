resource "google_pubsub_subscription" "judge_frames_push" {
  name  = "judge-frames-push"
  topic = google_pubsub_topic.frames_in.id

  push_config {
    push_endpoint = "${google_cloud_run_v2_service.judge.uri}/pubsub"

    oidc_token {
      service_account_email = google_service_account.judge.email
      audience              = google_cloud_run_v2_service.judge.uri
    }
  }

  ack_deadline_seconds = 30
}