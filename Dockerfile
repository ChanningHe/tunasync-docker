FROM ustcmirror/base:alpine
LABEL maintainer="iBug <docker@ibugone.com>"
LABEL bind_support=true

RUN <<EOF
    set -euo pipefail
    apk add --no-cache --update \
        wget perl ca-certificates git python3 py3-requests dcron tini su-exec aria2
    mkdir -p /usr/local/lib/tunasync
EOF

ADD tunasync /usr/local/lib/tunasync
ADD sync.sh sync.py /
RUN chmod +x /sync.sh

ADD entrypoint.sh /
RUN chmod +x /entrypoint.sh

# Default configuration for multi-threaded downloads
ENV CRON=""
# Default UID/GID for file system access
ENV PUID=1000
ENV PGID=1000
ENV ARIA2_DEBUG=false
ENV INDEX_DEBUG=false

# 并行下载任务数
ENV PARALLEL_DOWNLOADS=4


# Use tini as the entrypoint to handle PID 1 responsibilities
ENTRYPOINT ["/sbin/tini", "--", "/entrypoint.sh"]
