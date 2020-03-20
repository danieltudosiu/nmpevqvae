#!/usr/bin/env bash
docker build . -f Dockerfile\
 -t danieltudosiu/amigo_docker:neuromorphometry \
 --build-arg USER_NAME=danieltudosiu \
 --build-arg USER_ID=1004 \
