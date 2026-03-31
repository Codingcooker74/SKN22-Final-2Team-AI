#!/bin/bash
set -euo pipefail

compose_file="docker-compose.yml"
if [[ ! -f "$compose_file" ]]; then
  echo "docker-compose.yml not found"
  exit 1
fi

mapfile -t registries < <(
  grep -E '^[[:space:]]*image:[[:space:]]*' "$compose_file" \
  | awk '{print $2}' \
  | cut -d/ -f1 \
  | grep -E '\.dkr\.ecr\..*\.amazonaws\.com$' \
  | sort -u
)

if [[ ${#registries[@]} -eq 0 ]]; then
  echo "No ECR registries found in docker-compose.yml"
  exit 0
fi

for registry in "${registries[@]}"; do
  region="$(echo "$registry" | cut -d. -f4)"
  aws ecr get-login-password --region "$region" \
    | docker login --username AWS --password-stdin "$registry"
done
