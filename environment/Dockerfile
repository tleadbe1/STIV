# hash:sha256:4651801fcc194b1db32cd0897f9f12bd9d7cda7edb11fc6ece960b9003f32f59
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    jax==0.4.18 \
    jaxlib==0.4.18 \
    matplotlib==3.8.0 \
    numpy==1.26.0 \
    scipy==1.11.3
