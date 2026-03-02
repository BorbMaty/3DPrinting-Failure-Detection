resource "google_compute_network" "main" {
  name                    = "main-net"
  auto_create_subnetworks = true
}

resource "google_compute_address" "gpu_vm_ip" {
  name   = "gpu-worker-ip"
  region = var.region
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["gpu-worker"]
}

resource "google_compute_firewall" "allow_mediamtx" {
  name    = "allow-mediamtx"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["8554", "8888", "8889"]
  }

  allow {
    protocol = "udp"
    ports    = ["8554", "8000-8002"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["gpu-worker"]
}

resource "google_compute_instance" "gpu_worker" {
  name         = "gpu-worker"
  zone         = var.zone
  machine_type = "g2-standard-4"

  boot_disk {
    initialize_params {
      image = "projects/debian-cloud/global/images/family/debian-12"
      size  = 50
    }
  }

  network_interface {
    network = google_compute_network.main.name
    access_config {
      nat_ip = google_compute_address.gpu_vm_ip.address
    }
  }

  service_account {
    email  = google_service_account.inference.email
    scopes = ["cloud-platform"]
  }

  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
    preemptible         = false
  }

  metadata = {
    enable-oslogin = "TRUE"
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io docker-compose
    systemctl enable docker
    systemctl start docker

    mkdir -p /opt/mediamtx
    cat > /opt/mediamtx/docker-compose.yml <<'COMPOSE'
    version: "3"
    services:
      mediamtx:
        image: europe-west1-docker.pkg.dev/printermonitor-488112/printer/mediamtx:latest
        restart: always
        network_mode: host
        environment:
          - GCP_PROJECT=printermonitor-488112
          - PUBSUB_TOPIC=frames-in
          - STREAM_URL=http://localhost:8888/cam1/index.jpg
          - FPS=2
          - CAMERA_ID=cam1
          - JPEG_QUALITY=70
    COMPOSE

    gcloud auth configure-docker europe-west1-docker.pkg.dev --quiet
    docker-compose -f /opt/mediamtx/docker-compose.yml pull
    docker-compose -f /opt/mediamtx/docker-compose.yml up -d
  EOF

  tags = ["gpu-worker"]
}

output "gpu_vm_ip" {
  value = google_compute_address.gpu_vm_ip.address
}