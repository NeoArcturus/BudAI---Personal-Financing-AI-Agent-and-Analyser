#!/bin/bash
set -e

# If TAILSCALE_AUTHKEY or TS_AUTHKEY is provided, start userspace Tailscale
AUTHKEY="${TS_AUTHKEY:-${TAILSCALE_AUTHKEY:-"tskey-auth-ksKKTZqfPi11CNTRL-Srpy7jiCjqZiMLXPp4grqZbctSCZ7Z7v"}}"

if [ -n "$AUTHKEY" ]; then
    echo "Starting Tailscale daemon in userspace networking mode..."
    mkdir -p /var/run/tailscale /var/cache/tailscale /var/lib/tailscale
    /usr/local/bin/tailscaled --state=mem: --tun=userspace-networking --socks5-server=localhost:1055 --outbound-http-proxy-listen=localhost:1055 &
    
    echo "Authenticating Tailscale node..."
    /usr/local/bin/tailscale up --authkey="${AUTHKEY}" --hostname="${TS_HOSTNAME:-budai-cloud}" --accept-dns=true
    echo "Tailscale node successfully connected to tailnet!"
    /usr/local/bin/tailscale status || true

    # Export HTTP proxy for tailnet traffic while bypassing public cloud databases and services
    export HTTP_PROXY="http://localhost:1055"
    export HTTPS_PROXY="http://localhost:1055"
    export NO_PROXY="neon.tech,.neon.tech,upstash.io,.upstash.io,amazonaws.com,.amazonaws.com,googleapis.com,.googleapis.com,google.com,.google.com,localhost,127.0.0.1"
else
    echo "No TS_AUTHKEY found. Starting backend in direct networking mode."
fi

echo "Starting BudAI backend..."
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec python main.py
fi
