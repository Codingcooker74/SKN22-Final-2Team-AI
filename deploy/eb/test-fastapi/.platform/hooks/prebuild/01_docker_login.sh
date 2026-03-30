#!/bin/bash
set -euo pipefail

if [[ -z "${DOCKER_USERNAME:-}" || -z "${DOCKER_PASSWORD:-}" ]]; then
  echo "Docker Hub credentials are not set in the Elastic Beanstalk environment."
  exit 1
fi

echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
