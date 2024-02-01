#!/usr/bin/bash

docker compose up -d
xhost +local:
docker compose exec python bash
xhost -local:
