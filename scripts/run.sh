#!/bin/bash

echo "Giving container permission to use screen."
xhost +local:

# Parse CLI args
CONTAINER_NAME=$1
if [ -z "${CONTAINER_NAME}" ]; then
    CONTAINER_NAME=planning
fi

CONTAINER_DATA_DIR=$2
if [ -z "${CONTAINER_DATA_DIR}" ]; then
  BASENAME=$(basename ${HOST_DATA_DIR})
  CONTAINER_DATA_DIR=/app/${CONTAINER_NAME}
fi

HOST_DATA_DIR=$3
if [ -z "${HOST_DATA_DIR}" ]; then
    HOST_DATA_DIR=$(pwd)
fi

# Echo back current config
echo "Container name         : ${CONTAINER_NAME}"
echo "Data directory name    : ${HOST_DATA_DIR}"
echo "Target data dir        : ${CONTAINER_DATA_DIR}"
echo "*** RUNNING IN DEVELOPMENT MODE ***********"
echo "*                                         *"
echo "* ALL DATA STORED LOCALLY IN THIS         *"
echo "* CONTAINER WILL BE REFLECTED LOCALLY     *"
echo "*******************************************"
echo ""

GITCONFIG=$HOME/.gitconfig
CONTAINER_ID=`docker ps -aqf "name=^/${CONTAINER_NAME}$"`

if [ -z "${CONTAINER_ID}" ]; then
  echo "Creating new container."
  sudo docker run -it --privileged --rm \
      --network host \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v $(pwd):/app:rw \
      -e DISPLAY=unix$DISPLAY \
      --name=${CONTAINER_NAME} \
      ${CONTAINER_NAME}:latest \
      /bin/bash

else
  echo "Found running ${CONTAINER_NAME} container, attaching bash..."
  sudo docker exec -it ${CONTAINER_ID} bash

fi
