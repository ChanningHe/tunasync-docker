# TunaSync Docker

本项目是基于Docker的镜像同步工具，支持自定义用户权限、定时任务和并行下载功能。

## 项目来源

- Docker脚本等部分来自 [ustcmirror-images](https://github.com/ustclug/ustcmirror-images)
- apt-sync.py主要代码部分来自 [tunasync-scripts](https://github.com/tuna/tunasync-scripts)
- 根据原项目要求，本项目遵守GPLv3协议

## 主要特性

- 支持自定义UID/GID，解决权限问题
- 集成cron功能，单容器即可实现定时自动同步
- 使用aria2c替换默认下载器，支持多线程并行下载，显著提升同步速度
- 保持与原项目的兼容性，可直接替换使用

## 使用方法

### Docker Compose示例

```yaml
services:
  aptsync-proxmox:
    environment:
      - PUID=[CHANGE-UID]
      - PGID=[CHANGE-GID]
      - APTSYNC_UNLINK=1
      - APTSYNC_URL=http://download.proxmox.com/debian/pve
      - APTSYNC_DISTS=bookworm|pve-no-subscription|amd64|
      - CRON=20 23,6,12,20 * * *
      - TO=/data
    volumes:
      - ./mirrors-data/proxmox:/data
      - ./mirrors-logs/aptsync-proxmox:/log
      - /etc/timezone:/etc/timezone:ro
      - /etc/localtime:/etc/localtime:ro
    image: channinghe/apt-sync:aria2c

  aptsync-tailscale:
    environment:
      - PUID=[CHANGE-UID]
      - PGID=[CHANGE-GID]
      - APTSYNC_UNLINK=1
      - APTSYNC_URL=https://pkgs.tailscale.com/stable/debian
      - APTSYNC_DISTS=bookworm|main|arm64 amd64|:bullseye|main|arm64 amd64|
      - CRON=20 23,6,12,18 * * *
      - TO=/data
    volumes:
      - ./mirrors-data/tailscale:/data
      - ./mirrors-logs/tailscale:/log
      - /etc/timezone:/etc/timezone:ro
      - /etc/localtime:/etc/localtime:ro
    image: channinghe/apt-sync:aria2c

  aptsync-debian:
    environment:
      - PUID=[CHANGE-UID]
      - PGID=[CHANGE-GID]
      - APTSYNC_UNLINK=1
      - APTSYNC_URL=http://deb.debian.org/debian
      - APTSYNC_DISTS=bookworm|main contrib non-free non-free-firmware|amd64
        arm64|:bullseye|main contrib non-free|amd64 arm64|
      - CRON="0 0,6,12,18 * * *"
      - TO=/data
      # 下载线程数
      - PARALLEL_DOWNLOADS=8
    volumes:
      - ./mirrors-data/debian:/data
      - ./mirrors-logs/aptsync-debian:/log
    image: channinghe/apt-sync:aria2c
networks: {}
```

## 环境变量说明

### 通用环境变量

| 变量名 | 描述 |
|-------|------|
| PUID | 运行进程的用户ID |
| PGID | 运行进程的组ID |
| CRON | 定时任务表达式，例如：`20 23,6,12,20 * * *` |
| TO | 数据存储路径，通常设置为 `/data` |

### APT同步专用环境变量

| 变量名 | 描述 |
|-------|------|
| APTSYNC_URL | 上游镜像URL |
| APTSYNC_DISTS | 需要同步的发行版配置，格式为：`发行版\|组件\|架构\|下载路径` |
| APTSYNC_UNLINK | 是否先删除目标文件，设置为1开启 |

PS: 关于APTSYNC_DISTS，你可以访问[APT源转TunaSync-Docker环境变量转换工具](https://www.homelabproject.cc/tools/apt-converter/) ，来在线批量转换
## 挂载点说明

| 路径 | 描述 |
|-----|------|
| /data | 镜像数据存储目录 |
| /log | 日志存储目录 |
| /etc/timezone | 容器时区配置（只读） |
| /etc/localtime | 容器本地时间配置（只读） |

## 镜像标签

当前可用镜像：
- `channinghe/apt-sync:aria2c` - 使用aria2c加速的APT同步镜像
- `channinghe/apt-sync:latest` - 无aria2c，只支持CRON和PUID/GUID的版本
