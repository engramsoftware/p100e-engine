# p100e-server container, text only. CUDA 12.9 is the last toolkit for the P100.
#   docker build -t p100e-server .
#   docker build --build-arg GPU_ARCH="60;75;86" -t p100e-server .     also for the GTX 1660 and RTX 3070
# The image holds no model and no keys. See docs/KUBERNETES.md.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS build
ARG GPU_ARCH=60
ARG JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake ninja-build git python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCH}" \
    && cmake --build build -j "${JOBS}" --target p100e-server p100e-server-fake

FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04
COPY --from=build /src/build/p100e-server /usr/local/bin/p100e-server
USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/p100e-server"]
