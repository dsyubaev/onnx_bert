ARG BUILD_PLATFORM=linux/amd64

FROM --platform=${BUILD_PLATFORM} golang:1.23.3-bullseye as runtime

WORKDIR /tmp
# кладем бинарь для onnxruntime в /usr/lib/libonnxruntime.so
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz \
    && tar -xzf onnxruntime-linux-x64-1.20.0.tgz \
    && mv ./onnxruntime-linux-x64-1.20.0/lib/libonnxruntime.so.1.20.0 /usr/lib/libonnxruntime.so


# кладем бинарь для токенайзера /usr/lib/libtokenizers.a
RUN wget https://github.com/daulet/tokenizers/releases/download/v1.20.2/libtokenizers.linux-amd64.tar.gz \
    && tar -C /usr/lib -xzf libtokenizers.linux-amd64.tar.gz

FROM --platform=${BUILD_PLATFORM} runtime as build    
WORKDIR /app
COPY . /app
RUN go build .

ENTRYPOINT ["/app/onnx_bert"]