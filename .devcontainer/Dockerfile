FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# Install Python and git
RUN apt-get update && apt-get install -y python3 python3-pip git

# Install requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

# Add non-root user
RUN adduser --disabled-password --gecos "" devcontainer
USER devcontainer