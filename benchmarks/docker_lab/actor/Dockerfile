FROM alpine:3.20
RUN apk add --no-cache curl python3 bind-tools jq
RUN adduser -D -u 10001 agent
USER agent
WORKDIR /work
