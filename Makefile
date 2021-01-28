# Build, run, and upload the environment from this makefile
#
# ------------------------------------------------

# Image name is set to the repo name
IMAGE_NAME_CAPS := $(shell basename -s .git `git config --get remote.origin.url`)
IMAGE_NAME := $(shell echo "${IMAGE_NAME_CAPS}" | tr '[:upper:]' '[:lower:]')

# The image tag is set to either the git tag of the checked out commit
# or set to "latest" if the checked out commit does not have a tag
GIT_TAG := $(shell git describe --exact-match --tags 2>/dev/null)
TAG := $(if $(GIT_TAG),$(GIT_TAG),latest)


.PHONY: all
all: build run

.PHONY: build
build:
		sudo ./scripts/build.sh ${IMAGE_NAME} ${TAG}
ifneq ($(GIT_TAG),)
	./scripts/build.sh ${IMAGE_NAME} "latest"
endif

.PHONY: run
run:
		sudo ./scripts/run.sh ${IMAGE_NAME} ${TAG}
