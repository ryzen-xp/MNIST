FROM rust:1-bookworm AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY static ./static
COPY guide/config.json guide/model.bpk ./guide/

RUN cargo build --release --locked

FROM debian:bookworm-slim

WORKDIR /app

COPY --from=builder /app/target/release/mnist ./mnist
COPY --from=builder /app/guide/config.json /app/guide/config.json
COPY --from=builder /app/guide/model.bpk /app/guide/model.bpk

ENV PORT=10000
EXPOSE 10000

CMD ["./mnist"]
