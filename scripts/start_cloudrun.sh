#!/bin/bash
set -e

# If TAILSCALE_AUTHKEY or TS_AUTHKEY is provided, start userspace Tailscale
AUTHKEY="${TS_AUTHKEY:-$TAILSCALE_AUTHKEY}"

if [ -n "$AUTHKEY" ]; then
    echo "Starting Tailscale daemon in userspace networking mode..."
    mkdir -p /var/run/tailscale /var/cache/tailscale /var/lib/tailscale
    /usr/local/bin/tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --outbound-http-proxy-listen=localhost:1055 &
    
    echo "Authenticating Tailscale node..."
    /usr/local/bin/tailscale up --authkey="${AUTHKEY}" --hostname="${TS_HOSTNAME:-budai-cloud}" --accept-dns=true
    echo "Tailscale node successfully connected to tailnet!"

    # Export HTTP and SOCKS proxy for tailnet traffic while bypassing public cloud databases and services
    export HTTP_PROXY="http://localhost:1055"
    export HTTPS_PROXY="http://localhost:1055"
    export ALL_PROXY="socks5://localhost:1055"
    export NO_PROXY="neon.tech,aws.neon.tech,upstash.io,googleapis.com,google.com,localhost,127.0.0.1"
else
    echo "No TS_AUTHKEY found. Starting backend in direct networking mode."
fi

echo "Starting BudAI backend..."
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec python main.py
fi
