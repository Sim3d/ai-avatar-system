<#
.SYNOPSIS
  Local equivalent of .github/workflows/deploy.yml, using your AWS SSO session
  instead of GitHub Actions secrets.

.DESCRIPTION
  Builds the backend and frontend Docker images, pushes them to ECR, then
  forces a new deployment of the three ECS services (avatar-backend,
  avatar-frontend, avatar-worker) and waits for them to stabilise.

  Prerequisites:
    - Docker Desktop running
    - AWS CLI v2 installed and signed in: aws sso login [--profile <name>]
    - terraform apply has already created the ECR repos and ECS cluster

.EXAMPLE
  .\deploy-local.ps1
  .\deploy-local.ps1 -ImageTag v0.2.0 -Profile pimtrack-dev
  .\deploy-local.ps1 -SkipWait
#>
[CmdletBinding()]
param(
    [string]$Cluster = "avatar-cluster-production",
    [string]$ImageTag = "latest",
    [string]$Region = "us-east-1",
    # AWS CLI profile (leave empty to use the default profile / env credentials)
    [string]$Profile = "",
    # Skip the (long) wait for ECS services to stabilise
    [switch]$SkipWait
)

$ErrorActionPreference = "Stop"
$awsArgs = @("--region", $Region, "--no-cli-pager")
if ($Profile) { $awsArgs += @("--profile", $Profile) }

function Invoke-Aws {
    param([Parameter(ValueFromRemainingArguments)][string[]]$Args)
    aws @Args @awsArgs
    if ($LASTEXITCODE -ne 0) { throw "aws $($Args -join ' ') failed (exit $LASTEXITCODE)" }
}

# --- Resolve ECR registry from the current identity -------------------------
Write-Host "==> Verifying AWS session..." -ForegroundColor Cyan
$identity = Invoke-Aws sts get-caller-identity --output json | ConvertFrom-Json
$registry = "$($identity.Account).dkr.ecr.$Region.amazonaws.com"
Write-Host "    Account : $($identity.Account)"
Write-Host "    Identity: $($identity.Arn)"
Write-Host "    Registry: $registry"

# --- Login to ECR ------------------------------------------------------------
Write-Host "==> Logging in to ECR..." -ForegroundColor Cyan
Invoke-Aws ecr get-login-password | docker login --username AWS --password-stdin $registry
if ($LASTEXITCODE -ne 0) { throw "docker login failed" }

# --- Build and push ----------------------------------------------------------
$repoRoot = $PSScriptRoot
foreach ($svc in @(
    @{ Name = "avatar-backend";  Context = Join-Path $repoRoot "backend" },
    @{ Name = "avatar-frontend"; Context = Join-Path $repoRoot "frontend" }
)) {
    Write-Host "==> Building $($svc.Name):$ImageTag..." -ForegroundColor Cyan
    docker build -t "$registry/$($svc.Name):$ImageTag" -t "$registry/$($svc.Name):latest" $svc.Context
    if ($LASTEXITCODE -ne 0) { throw "docker build $($svc.Name) failed" }

    Write-Host "==> Pushing $($svc.Name)..." -ForegroundColor Cyan
    docker push "$registry/$($svc.Name):$ImageTag"
    if ($LASTEXITCODE -ne 0) { throw "docker push $($svc.Name):$ImageTag failed" }
    if ($ImageTag -ne "latest") {
        docker push "$registry/$($svc.Name):latest"
        if ($LASTEXITCODE -ne 0) { throw "docker push $($svc.Name):latest failed" }
    }
}

# --- Roll the ECS services ---------------------------------------------------
# The worker runs the same backend image, so it is rolled too.
$services = @("avatar-backend", "avatar-frontend", "avatar-worker")
foreach ($svc in $services) {
    Write-Host "==> Rolling $svc..." -ForegroundColor Cyan
    Invoke-Aws ecs update-service --cluster $Cluster --service $svc --force-new-deployment --output text --query "service.serviceName" | Out-Null
}

if ($SkipWait) {
    Write-Host "==> Deployment triggered (wait skipped). Check status with:" -ForegroundColor Yellow
    Write-Host "    aws ecs describe-services --cluster $Cluster --services $($services -join ' ')"
} else {
    Write-Host "==> Waiting for services to stabilise (backend cold start loads ~9GB of weights, this can take a while)..." -ForegroundColor Cyan
    Invoke-Aws ecs wait services-stable --cluster $Cluster --services @services
    Write-Host "==> Deployment complete." -ForegroundColor Green
}
