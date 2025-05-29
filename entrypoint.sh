#!/bin/sh
set -e

# 创建日志目录
mkdir -p /log
LOG_FILE="/log/tunasync.log"
touch "$LOG_FILE"
echo "$(date): 容器启动" | tee -a "$LOG_FILE"

# Show current download configuration
echo "Starting APT sync with configuration:" | tee -a "$LOG_FILE"

# 处理用户权限
if [ -n "${PUID}" ] && [ -n "${PGID}" ] && [ "${PUID}" != "0" ] && [ "${PGID}" != "0" ]; then
    echo "以UID=${PUID}, GID=${PGID}运行" | tee -a "$LOG_FILE"
    
    # 创建用户组
    if ! getent group ${PGID} >/dev/null; then
        addgroup -g ${PGID} tunesync
    fi
    
    # 创建用户
    if ! getent passwd ${PUID} >/dev/null; then
        EXISTING_GROUP=$(getent group ${PGID} | cut -d: -f1)
        adduser -D -u ${PUID} -G ${EXISTING_GROUP} tunesync
    fi
    
    # 设置目录权限
    chown -R ${PUID}:${PGID} /usr/local/lib/tunasync
    # 确保日志目录有权限
    chown -R ${PUID}:${PGID} /log
    
    # 以指定用户执行命令
    SYNC_CMD="su-exec ${PUID}:${PGID} /sync.sh"
else
    echo "以root身份运行 (UID=0, GID=0)" | tee -a "$LOG_FILE"
    SYNC_CMD="/sync.sh"
fi

# Execute sync immediately when container starts
${SYNC_CMD}

# Configure scheduling
if [ -n "$CRON" ]; then
    echo "Setting up cron schedule: $CRON" | tee -a "$LOG_FILE"
    echo "$CRON ${SYNC_CMD} 2>&1 | tee -a $LOG_FILE /proc/1/fd/1 >/dev/null" > /etc/crontabs/root
    # 修改 crond 的启动参数，去掉 -d 8，添加 -l 2
    exec crond -f -l 2
else
    echo "No CRON setting, container will exit after manual sync" | tee -a "$LOG_FILE"
    exec tail -f /dev/null  # Keep container running (can be replaced with other persistent process as needed)
fi
