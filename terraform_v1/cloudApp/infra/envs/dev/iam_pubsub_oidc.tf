# Allow the Pub/Sub service agent to mint OIDC tokens for the judge SA.
# This is what lets the push subscription authenticate to Cloud Run.
# NOTE: data.google_project.project is declared in iam.tf — do not redeclare here.

resource "google_service_account_iam_member" "pubsub_can_mint_judge_oidc" {
  service_account_id = google_service_account.judge.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

