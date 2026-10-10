#!/usr/bin/env python3
"""
Serve a RadClaw trace directory over HTTP.

Usage:
    python scripts/serve_report.py output/traces_demo/sample_13
    python scripts/serve_report.py output/traces_demo/sample_13 --port 8899
    python scripts/serve_report.py output/traces_demo/sample_13 --regenerate

Opens the report in the browser automatically (or prints the URL if running remotely).
"""

import argparse
import http.server
import os
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def find_free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    return start


def main():
    parser = argparse.ArgumentParser(description="Serve a RadClaw trace directory")
    parser.add_argument("trace_dir", help="Trace directory (e.g. output/traces_demo/sample_13)")
    parser.add_argument("--port", type=int, default=8899, help="HTTP port (default: 8899)")
    parser.add_argument("--regenerate", action="store_true", help="Re-generate report.html before serving")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).resolve()
    if not trace_dir.is_dir():
        sys.exit(f"ERROR: {trace_dir} is not a directory")

    report_html = trace_dir / "report.html"

    # Optionally regenerate
    if args.regenerate or not report_html.exists():
        print("Generating report.html ...")
        gen_script = Path(__file__).resolve().parent / "generate_report_html.py"
        result = subprocess.run(
            [sys.executable, str(gen_script), str(trace_dir)],
            capture_output=True, text=True,
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

    if not report_html.exists():
        sys.exit(f"ERROR: {report_html} not found. Run generate_report_html.py first.")

    port = find_free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} in use, using {port} instead.")

    # Change to trace_dir so relative image paths resolve correctly
    os.chdir(trace_dir)

    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *a: None  # suppress request logs

    server = http.server.HTTPServer(("0.0.0.0", port), handler)

    local_ip = get_local_ip()
    url_local = f"http://localhost:{port}/report.html"
    url_remote = f"http://{local_ip}:{port}/report.html"

    print(f"\nServing: {trace_dir}")
    print(f"  Local URL : {url_local}")
    print(f"  Remote URL: {url_remote}")
    print(f"\nSSH tunnel (run on your local machine):")
    print(f"  ssh -L {port}:localhost:{port} ${{SSH_CONNECTION%% *}}")
    print(f"Then open: {url_local}")
    print("\nPress Ctrl+C to stop.\n")

    # Try to open browser (only works if running locally)
    def _try_open():
        webbrowser.open(url_local)
    threading.Timer(0.5, _try_open).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()