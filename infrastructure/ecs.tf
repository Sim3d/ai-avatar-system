# ─────────────────────────────────────────────────────────────────────────────
# ECS compute: the part that actually runs the application.
#
# main.tf provisions the data plane (VPC, RDS, ElastiCache, S3, CloudFront,
# ALB, ECR) and an empty ECS cluster. Until this file existed, `terraform
# apply` finished with nothing running: no task definitions, no services, no
# target groups, no listeners. deploy.sh then printed "Infrastructure deployed
# successfully" over an empty cluster, which is a costly thing to be told.
#
# EC2, not Fargate. Fargate has no GPU support, and MuseTalk without a GPU
# takes 30-90 s per sentence — so the cluster is backed by a GPU Auto Scaling
# group through a capacity provider.
# ─────────────────────────────────────────────────────────────────────────────

locals {
  name           = "${var.project_name}-${var.environment}"
  backend_image  = var.backend_image != "" ? var.backend_image : "${aws_ecr_repository.backend.repository_url}:latest"
  frontend_image = var.frontend_image != "" ? var.frontend_image : "${aws_ecr_repository.frontend.repository_url}:latest"
}

# ── Secrets ──────────────────────────────────────────────────────────────────
# Held in SSM and injected by the agent at start, so they never appear in the
# task definition, which is world-readable to anyone with ecs:DescribeTaskDefinition.

resource "aws_ssm_parameter" "db_password" {
  name  = "/${local.name}/DATABASE_PASSWORD"
  type  = "SecureString"
  value = random_password.db_password.result
}

# Auto-generated when the variables are left empty, so no secret ever needs to
# be passed on the command line or shared between operators — it lives only in
# the (encrypted, remote) state and in SSM.
resource "random_password" "jwt_secret_key" {
  length  = 64
  special = false
}

resource "random_password" "secret_key" {
  length  = 64
  special = false
}

resource "aws_ssm_parameter" "jwt_secret_key" {
  name  = "/${local.name}/JWT_SECRET_KEY"
  type  = "SecureString"
  value = var.jwt_secret_key != "" ? var.jwt_secret_key : random_password.jwt_secret_key.result
}

resource "aws_ssm_parameter" "secret_key" {
  name  = "/${local.name}/SECRET_KEY"
  type  = "SecureString"
  value = var.secret_key != "" ? var.secret_key : random_password.secret_key.result
}

resource "aws_ssm_parameter" "anthropic_api_key" {
  count = var.anthropic_api_key != "" ? 1 : 0
  name  = "/${local.name}/ANTHROPIC_API_KEY"
  type  = "SecureString"
  value = var.anthropic_api_key
}

resource "aws_ssm_parameter" "openai_api_key" {
  count = var.openai_api_key != "" ? 1 : 0
  name  = "/${local.name}/OPENAI_API_KEY"
  type  = "SecureString"
  value = var.openai_api_key
}

# ── IAM ──────────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task_execution" {
  name               = "${local.name}-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# The managed policy above covers ECR and logs but NOT reading SSM parameters,
# which is what the `secrets` blocks below need at container start.
data "aws_iam_policy_document" "task_execution_secrets" {
  statement {
    actions   = ["ssm:GetParameters", "ssm:GetParameter"]
    resources = ["arn:aws:ssm:${var.aws_region}:*:parameter/${local.name}/*"]
  }
  statement {
    actions   = ["kms:Decrypt"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "task_execution_secrets" {
  name   = "${local.name}-read-secrets"
  role   = aws_iam_role.task_execution.id
  policy = data.aws_iam_policy_document.task_execution_secrets.json
}

# Task role: what the application itself may do at runtime (S3 media only).
resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "task_s3" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.media.arn}/*"]
  }
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.media.arn]
  }
}

resource "aws_iam_role_policy" "task_s3" {
  name   = "${local.name}-s3-media"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_s3.json
}

# Claude via Amazon Bedrock — the app authenticates with this task role, so no
# LLM API key is stored anywhere. Covers foundation models and the regional /
# global inference profiles (us.anthropic.*, global.anthropic.*).
data "aws_iam_policy_document" "task_bedrock" {
  statement {
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    resources = [
      "arn:aws:bedrock:*::foundation-model/anthropic.*",
      "arn:aws:bedrock:*:*:inference-profile/*.anthropic.*",
    ]
  }
}

resource "aws_iam_role_policy" "task_bedrock" {
  name   = "${local.name}-bedrock"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_bedrock.json
}

# ECS Exec: lets developers open a shell in running tasks (read worker logs,
# hot-patch code) without a full rebuild/redeploy cycle.
data "aws_iam_policy_document" "task_exec" {
  statement {
    actions = [
      "ssmmessages:CreateControlChannel",
      "ssmmessages:CreateDataChannel",
      "ssmmessages:OpenControlChannel",
      "ssmmessages:OpenDataChannel",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "task_exec" {
  name   = "${local.name}-ecs-exec"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_exec.json
}

# EC2 instances that join the cluster.
data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_instance" {
  name               = "${local.name}-ecs-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "ecs_instance" {
  role       = aws_iam_role.ecs_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

# Lets you reach a stuck instance without opening SSH to the world.
resource "aws_iam_role_policy_attachment" "ecs_instance_ssm" {
  role       = aws_iam_role.ecs_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ecs_instance" {
  name = "${local.name}-ecs-instance"
  role = aws_iam_role.ecs_instance.name
}

# ── GPU capacity ─────────────────────────────────────────────────────────────

# The ECS-optimised GPU AMI, resolved at plan time rather than hardcoded — a
# pinned AMI id rots and is region-specific.
data "aws_ssm_parameter" "ecs_gpu_ami" {
  name = "/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id"
}

resource "aws_security_group" "ecs_instances" {
  name        = "${local.name}-ecs-instances"
  description = "ECS container instances"
  vpc_id      = module.vpc.vpc_id

  # Only from the load balancer. The instances sit in private subnets and are
  # never directly reachable; use SSM Session Manager for access.
  ingress {
    description     = "From ALB"
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name}-ecs-instances" }
}

resource "aws_launch_template" "ecs_gpu" {
  name_prefix   = "${local.name}-gpu-"
  image_id      = data.aws_ssm_parameter.ecs_gpu_ami.value
  instance_type = var.gpu_instance_type

  iam_instance_profile {
    arn = aws_iam_instance_profile.ecs_instance.arn
  }

  vpc_security_group_ids = [aws_security_group.ecs_instances.id]

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      # MuseTalk weights are ~9 GB and the CUDA image is large; the 30 GB
      # default fills up and tasks then fail to pull with no obvious cause.
      volume_size           = 100
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  user_data = base64encode(<<-EOT
    #!/bin/bash
    echo "ECS_CLUSTER=${aws_ecs_cluster.main.name}" >> /etc/ecs/ecs.config
    echo "ECS_ENABLE_GPU_SUPPORT=true" >> /etc/ecs/ecs.config
  EOT
  )

  metadata_options {
    http_tokens                 = "required" # IMDSv2 only
    http_endpoint               = "enabled"
    http_put_response_hop_limit = 2 # containers reach the task metadata endpoint
  }

  tag_specifications {
    resource_type = "instance"
    tags          = { Name = "${local.name}-gpu" }
  }
}

resource "aws_autoscaling_group" "ecs_gpu" {
  name                = "${local.name}-gpu"
  vpc_zone_identifier = module.vpc.private_subnets
  min_size            = var.gpu_instance_count
  max_size            = var.gpu_instance_count
  desired_capacity    = var.gpu_instance_count

  launch_template {
    id      = aws_launch_template.ecs_gpu.id
    version = "$Latest"
  }

  # Required by the ECS managed-scaling capacity provider.
  protect_from_scale_in = true

  tag {
    key                 = "AmazonECSManaged"
    value               = "true"
    propagate_at_launch = true
  }

  lifecycle {
    # ECS owns the desired count once managed scaling is attached; Terraform
    # fighting it would cause instances to flap.
    ignore_changes = [desired_capacity]
  }
}

resource "aws_ecs_capacity_provider" "gpu" {
  name = "${local.name}-gpu"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.ecs_gpu.arn
    managed_termination_protection = "ENABLED"

    managed_scaling {
      status          = "ENABLED"
      target_capacity = 100
    }
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = [aws_ecs_capacity_provider.gpu.name]

  default_capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    weight            = 1
  }
}

# ── Logs ─────────────────────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "backend" {
  name              = "/ecs/${local.name}/backend"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "frontend" {
  name              = "/ecs/${local.name}/frontend"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/ecs/${local.name}/worker"
  retention_in_days = 14
}

# ── Shared container config ──────────────────────────────────────────────────

locals {
  # Use the username reported by the RDS instance itself: main.tf hardcodes it,
  # so referencing var.db_username here could silently drift (and did).
  db_url    = "postgresql://${aws_db_instance.postgres.username}:${urlencode(random_password.db_password.result)}@${aws_db_instance.postgres.address}:5432/${aws_db_instance.postgres.db_name}"
  redis_url = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:6379"

  backend_env = [
    { name = "ENVIRONMENT", value = "production" },
    # The app refuses to start with DEBUG=true in production — it would expose
    # interactive docs, raw exception text and create_all instead of Alembic.
    { name = "DEBUG", value = "false" },
    { name = "DATABASE_URL", value = local.db_url },
    { name = "REDIS_URL", value = "${local.redis_url}/0" },
    { name = "CELERY_BROKER_URL", value = "${local.redis_url}/1" },
    { name = "CELERY_RESULT_BACKEND", value = "${local.redis_url}/2" },
    { name = "USE_LOCAL_STORAGE", value = "false" },
    { name = "S3_BUCKET_NAME", value = aws_s3_bucket.media.bucket },
    { name = "AWS_REGION", value = var.aws_region },
    { name = "CLOUDFRONT_DOMAIN", value = aws_cloudfront_distribution.media_cdn.domain_name },
    # Also enforced at startup: production requires a Secure cookie and https
    # origins, so a misconfiguration fails fast instead of shipping insecurely.
    { name = "AUTH_COOKIE_SECURE", value = "true" },
    # CORS: CloudFront (https API proxy) plus any extra origins supplied via
    # var.cors_origins (e.g. the Konekt shell in dev). The ALB frontend calls
    # the API on the same origin, so it does not need a CORS entry; http://
    # origins are rejected by the backend's production config validation.
    { name = "CORS_ORIGINS", value = join(",", compact([
      "https://${aws_cloudfront_distribution.media_cdn.domain_name}",
      var.cors_origins,
    ])) },
    { name = "FRONTEND_URL", value = "https://${aws_lb.main.dns_name}" },
    { name = "BACKEND_URL", value = "https://${aws_lb.main.dns_name}" },
    # LLM via Amazon Bedrock: authenticated by the task IAM role, no API key.
    { name = "LLM_PROVIDER", value = var.llm_provider },
    { name = "LLM_MODEL", value = var.llm_model },
    # One API process => one animator => one MuseTalk worker on the GPU. With
    # the default 4, each process would load its own copy of the models.
    { name = "UVICORN_WORKERS", value = "1" },
    # Keep model caches (face_alignment, HF hub) on the persistent weights
    # volume so they aren't re-downloaded on every task start.
    { name = "TORCH_HOME", value = "/app/models/MuseTalk/models/.torch" },
    { name = "HF_HOME", value = "/app/models/MuseTalk/models/.hf" },
    # GFPGAN pass on lip-synced frames sharpens MuseTalk's 256px mouth region,
    # but runs per frame and dominated chunk latency (~10-25s per sentence on
    # an A10G). Off for responsiveness; set "gfpgan" to trade latency for detail.
    { name = "FACE_RESTORE", value = "off" },
  ]

  backend_secrets = concat(
    [
      { name = "JWT_SECRET_KEY", valueFrom = aws_ssm_parameter.jwt_secret_key.arn },
      { name = "SECRET_KEY", valueFrom = aws_ssm_parameter.secret_key.arn },
    ],
    var.anthropic_api_key != "" ? [{ name = "ANTHROPIC_API_KEY", valueFrom = aws_ssm_parameter.anthropic_api_key[0].arn }] : [],
    var.openai_api_key != "" ? [{ name = "OPENAI_API_KEY", valueFrom = aws_ssm_parameter.openai_api_key[0].arn }] : [],
  )
}

# ── Task definitions ─────────────────────────────────────────────────────────

resource "aws_ecs_task_definition" "backend" {
  family                   = "${local.name}-backend"
  network_mode             = "awsvpc"
  requires_compatibilities = ["EC2"]
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn
  # Sized for g5.xlarge (4 vCPU / 16 GiB, ~15791 MiB registered with ECS).
  # Must coexist with worker (2048) and frontend (1024) on the same instance:
  # 12000 + 2048 + 1024 = 15072, leaving headroom for the agent.
  cpu    = 3072
  memory = 12000

  # Cloned voice samples and their index are written to /app/voice_profiles.
  # A host bind mount keeps them across task redeploys; they are still lost if
  # the EC2 instance itself is replaced (move to EFS or S3 if that matters).
  volume {
    name      = "voice-profiles"
    host_path = "/opt/avatar/voice_profiles"
  }

  # MuseTalk weights (~9 GB) are downloaded on first boot by entrypoint.sh.
  # Mounted over the clone's models/ dir only, so the code baked into the image
  # stays visible. Survives redeploys; re-downloaded if the instance is replaced.
  volume {
    name      = "musetalk-models"
    host_path = "/opt/avatar/musetalk_models"
  }

  container_definitions = jsonencode([{
    name         = "backend"
    image        = local.backend_image
    essential    = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    mountPoints = [
      { sourceVolume = "voice-profiles", containerPath = "/app/voice_profiles", readOnly = false },
      { sourceVolume = "musetalk-models", containerPath = "/app/models/MuseTalk/models", readOnly = false },
    ]
    environment  = local.backend_env
    secrets      = local.backend_secrets
    # One whole GPU. ECS has no fractional GPU allocation, which is why the
    # worker cannot also reserve one on a single-GPU instance.
    resourceRequirements = [{ type = "GPU", value = "1" }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.backend.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
    healthCheck = {
      # /health returns 503 when Postgres or Redis is unreachable, so this
      # genuinely reflects readiness rather than just "the process is up".
      command  = ["CMD-SHELL", "python -c \"import urllib.request;urllib.request.urlopen('http://localhost:8000/health')\" || exit 1"]
      interval = 30
      timeout  = 10
      retries  = 5
      # Model loading on a cold start is slow; a shorter grace kills the task
      # mid-load and it never converges.
      startPeriod = 300
    }
  }])
}

resource "aws_ecs_task_definition" "frontend" {
  family                   = "${local.name}-frontend"
  network_mode             = "awsvpc"
  requires_compatibilities = ["EC2"]
  execution_role_arn       = aws_iam_role.task_execution.arn
  cpu                      = 512
  memory                   = 1024

  container_definitions = jsonencode([{
    name         = "frontend"
    image        = local.frontend_image
    essential    = true
    portMappings = [{ containerPort = 3000, protocol = "tcp" }]
    environment = [
      { name = "NEXT_PUBLIC_API_URL", value = "https://${aws_lb.main.dns_name}" },
      { name = "NEXT_PUBLIC_WS_URL", value = "wss://${aws_lb.main.dns_name}" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.frontend.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name}-worker"
  network_mode             = "awsvpc"
  requires_compatibilities = ["EC2"]
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn
  cpu                      = 512
  memory                   = 2048

  container_definitions = jsonencode([{
    name        = "worker"
    image       = local.backend_image
    essential   = true
    command     = ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "--concurrency=2"]
    environment = local.backend_env
    secrets     = local.backend_secrets
    # Off by default. With one GPU instance the backend already holds the only
    # device, and a second GPU-reserving task would stay PENDING forever.
    resourceRequirements = var.worker_uses_gpu ? [{ type = "GPU", value = "1" }] : []
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.worker.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

# ── ALB routing ──────────────────────────────────────────────────────────────
# The ALB existed but had no listener and no targets, so nothing it received
# could reach a container.

resource "aws_lb_target_group" "backend" {
  name        = "${local.name}-backend"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip" # awsvpc task networking

  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }

  # WebSocket turns are long-lived; the 300 s default would cut a conversation
  # off mid-reply.
  deregistration_delay = 30
  stickiness {
    type            = "lb_cookie"
    enabled         = true
    cookie_duration = 86400
  }
}

resource "aws_lb_target_group" "frontend" {
  name        = "${local.name}-frontend"
  port        = 3000
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip"

  health_check {
    path                = "/"
    matcher             = "200-399"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }

  # A draining task keeps its ENI; with the 4-ENI g5.xlarge limit the 300 s
  # default blocked the backend from being placed during a rollout.
  deregistration_delay = 10
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  # Frontend serves everything not claimed by a rule below.
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.frontend.arn
  }
}

resource "aws_lb_listener_rule" "api" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.backend.arn
  }

  condition {
    # /ws/* carries the WebSocket upgrade; /health and /metrics are probed
    # directly. ALBs handle WebSocket upgrades on a normal HTTP listener.
    path_pattern {
      values = ["/api/*", "/ws/*", "/health", "/metrics", "/uploads/*"]
    }
  }
}

# ── Services ─────────────────────────────────────────────────────────────────

resource "aws_security_group" "ecs_tasks" {
  name        = "${local.name}-ecs-tasks"
  description = "awsvpc task ENIs"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "From ALB"
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name}-ecs-tasks" }
}

resource "aws_ecs_service" "backend" {
  name            = "${var.project_name}-backend"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = 1

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    weight            = 1
  }

  network_configuration {
    subnets         = module.vpc.private_subnets
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.backend.arn
    container_name   = "backend"
    container_port   = 8000
  }

  # Cold start loads ~9 GB of weights; without this the ALB marks the task
  # unhealthy and ECS kills it before it can finish.
  health_check_grace_period_seconds = 600

  # One GPU and one instance means a replacement cannot start before the old
  # task releases the device.
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  enable_execute_command = true

  depends_on = [aws_lb_listener.http, aws_iam_role_policy.task_execution_secrets]

  lifecycle {
    # deploy.sh / CI update the image via a new task definition revision.
    ignore_changes = [task_definition]
  }
}

resource "aws_ecs_service" "frontend" {
  name            = "${var.project_name}-frontend"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.frontend.arn
  desired_count   = 1

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    weight            = 1
  }

  network_configuration {
    subnets         = module.vpc.private_subnets
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.frontend.arn
    container_name   = "frontend"
    container_port   = 3000
  }

  # Every awsvpc task takes an ENI and a g5.xlarge only has 4 (one is the
  # host's). Starting a replacement before stopping the old task would steal
  # the ENI the backend needs, leaving it stuck in PROVISIONING.
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  depends_on = [aws_lb_listener.http]

  lifecycle {
    ignore_changes = [task_definition]
  }
}

resource "aws_ecs_service" "worker" {
  name            = "${var.project_name}-worker"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = 1

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    weight            = 1
  }

  network_configuration {
    subnets         = module.vpc.private_subnets
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  # No load balancer: the worker consumes from Redis and serves no traffic.

  # Same 4-ENI budget as the frontend: never run old and new side by side.
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  lifecycle {
    ignore_changes = [task_definition]
  }
}

# ── Outputs ──────────────────────────────────────────────────────────────────

output "application_url" {
  description = "Where the app is reachable once the services are healthy."
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "ecs_services" {
  description = "Service names, for `aws ecs update-service --force-new-deployment`."
  value = {
    backend  = aws_ecs_service.backend.name
    frontend = aws_ecs_service.frontend.name
    worker   = aws_ecs_service.worker.name
  }
}
