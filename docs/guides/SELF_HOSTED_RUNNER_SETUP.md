# Self-Hosted GitHub Actions Runner Setup

> **Purpose**: Enable automated payload publishing from the GPU server where `master_dataset_15T.parquet` resides.

---

## Why Self-Hosted?

The 5.8 GB Parquet file is `.gitignored` and only exists on the GPU server. GitHub-hosted runners cannot access this file, so we need a self-hosted runner on the GPU server itself.

---

## One-Time Setup (GPU Server)

### 1. Install GitHub Actions Runner

```bash
# Create runner directory
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download runner (Linux x64)
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
```

### 2. Configure Runner

You'll need a **GitHub Personal Access Token (PAT)** with `repo` and `admin:org` scopes, or use the token from **Settings → Actions → Runners → New self-hosted runner**.

```bash
./config.sh --url https://github.com/YOUR_USERNAME/ml-agent-workspace \
            --token YOUR_RUNNER_TOKEN \
            --name gpu-server-runner \
            --labels self-hosted,gpu,linux,payload-builder \
            --work _work
```

**Important**: Replace `YOUR_USERNAME` with your GitHub username and `YOUR_RUNNER_TOKEN` with the token from GitHub.

### 3. Install as systemd Service

This ensures the runner starts automatically on boot:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

Expected output:
```
● actions.runner.YOUR_USERNAME-ml-agent-workspace.gpu-server-runner.service
   Active: active (running)
```

### 4. Verify in GitHub

1. Go to **Settings → Actions → Runners**
2. You should see `gpu-server-runner` with status **"Idle"** (green)

---

## Alternative: Dockerized Runner Setup (No Sudo Required)

If you do not have root or `sudo` privileges on the GPU server, you can run the self-hosted runner inside a Docker container. This setup uses `docker-compose` to run the runner and mounts the host's `data/` directory read-only so that it can access the massive `.gitignored` parquet files.

### 1. Update the `.env` File
Add the following configuration parameters to your `.env` file on the GPU server:
```bash
# GitHub details for the self-hosted runner
GITHUB_REPOSITORY="YOUR_USERNAME/ml-agent-workspace"
GITHUB_TOKEN="YOUR_PERSONAL_ACCESS_TOKEN_OR_RUNNER_TOKEN"
RUNNER_NAME="gpu-docker-runner"
```
> [!NOTE]
> `GITHUB_TOKEN` should be a GitHub Personal Access Token (PAT) with `repo` scopes, or a temporary registration token retrieved from **Settings → Actions → Runners → New self-hosted runner**.

### 2. Build the Runner Image
Build the container using the configured Dockerfile:
```bash
docker compose build github-runner
```

### 3. Spin up the Container
Run the container in the background:
```bash
docker compose up -d github-runner
```

### 4. Monitor & Verify Logs
Check that the runner starts up, registers itself, and transitions to listening for jobs:
```bash
docker compose logs -f github-runner
```
Once started, check **Settings → Actions → Runners** in GitHub. You should see `gpu-docker-runner` registered with status **"Idle"**.

---

## GitHub Secrets Configuration

Add the following secrets via **Settings → Secrets and variables → Actions → New repository secret**:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `WANDB_API_KEY` | Your W&B API key | For downloading model artifacts |
| `R2_ACCOUNT_ID` | Cloudflare R2 account ID | From R2 dashboard |
| `R2_ACCESS_KEY_ID` | R2 access key ID | From R2 → Manage R2 API Tokens |
| `R2_SECRET_ACCESS_KEY` | R2 secret access key | From R2 API token creation |
| `R2_BUCKET_NAME` | Your R2 bucket name | e.g., `ml-payloads` |
| `R2_PUBLIC_URL` | R2 public bucket URL | e.g., `https://pub-xxx.r2.dev` |

---

## Running the Workflow

### Manual Trigger

1. Go to **Actions** tab in GitHub
2. Select **"Build and Publish Payloads to R2"** workflow
3. Click **"Run workflow"**
4. Fill in parameters:
   - **artifact_ref**: Single W&B artifact (optional)
   - **all_artifacts**: Process all artifacts (checkbox)
   - **start_date**: Analysis window start (YYYY-MM-DD)
   - **end_date**: Analysis window end (YYYY-MM-DD)
5. Click **"Run workflow"**

### Monitoring

- Real-time logs appear in the **Actions** tab
- Each step shows progress (export, registry build, R2 upload)
- Failures send email notifications (configure in GitHub settings)

---

## Troubleshooting

### Runner Shows "Offline"

```bash
# On GPU server
cd ~/actions-runner
sudo ./svc.sh status

# If not running:
sudo ./svc.sh start
```

### Workflow Fails: "No runner available"

- Verify labels match: workflow uses `[self-hosted, gpu, payload-builder]`
- Check runner config: `cat ~/actions-runner/.runner`

### Permission Denied Errors

```bash
# Ensure runner user has access to repo
ls -la ~/ml-agent-workspace/data/processed/master_dataset_15T.parquet

# If needed, adjust permissions
chmod +r ~/ml-agent-workspace/data/processed/master_dataset_15T.parquet
```

### R2 Upload Fails: Invalid Credentials

- Verify secrets in GitHub Settings → Secrets
- Test locally:
  ```bash
  python scripts/upload_to_r2.py --dry-run
  ```

---

## Maintenance

### Update Runner

```bash
cd ~/actions-runner
sudo ./svc.sh stop
./config.sh remove --token YOUR_NEW_TOKEN
# Download new version and re-run setup
sudo ./svc.sh install
sudo ./svc.sh start
```

### Remove Runner

```bash
cd ~/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token YOUR_TOKEN
```

---

## Alternative: CRON Script (No GitHub Actions)

If you prefer local automation without GitHub Actions, use `scripts/publish_payloads_cron.sh`:

```bash
#!/bin/bash
set -euo pipefail

REPO_ROOT="/path/to/ml-agent-workspace"
cd "$REPO_ROOT"

# Load environment
source .env

# Pull latest code
git pull origin main

# Export payloads
python src/export_paginated_payload.py \
    --all_artifacts \
    --start_date 2026-06-01 \
    --end_date 2026-07-01 \
    --output_dir frontend/public/data/runs

# Build registry
python scripts/build_run_registry.py frontend/public/data/runs

# Upload to R2
python scripts/upload_to_r2.py \
    --runs-dir frontend/public/data/runs \
    --prefix runs

# Clean up
rm -rf frontend/public/data/runs/*

echo "Payload publish complete at $(date)" >> logs/publish.log
```

**Crontab Entry** (Daily at 2 AM):
```cron
0 2 * * * /path/to/ml-agent-workspace/scripts/publish_payloads_cron.sh
```

---

## Security Notes

- **Never commit** the `.runner` file or GitHub token
- **Rotate secrets** every 90 days
- **Use minimal scopes** for PATs (only `repo` + `admin:org`)
- **Firewall**: Runner only needs outbound HTTPS (443) access

---

## References

- [GitHub Self-Hosted Runners Docs](https://docs.github.com/en/actions/hosting-your-own-runners)
- [Cloudflare R2 API Docs](https://developers.cloudflare.com/r2/)
- [W&B Artifact API](https://docs.wandb.ai/ref/python/artifact)
