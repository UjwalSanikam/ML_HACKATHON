"""Serve the evaluator UI (Laptop 2).

    python ui/serve.py --port 8080 --api http://<laptop1-ip>:8000

Then open http://localhost:8080 . The UI contains no model, no road network and no labels;
everything comes from the API.
"""
import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--api", default=None, help="API base URL, e.g. http://192.168.1.20:8000")
ap.add_argument("--no-browser", action="store_true")
args = ap.parse_args()

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(Path(__file__).parent))
url = f"http://localhost:{args.port}/" + (f"?api={args.api}" if args.api else "")
# Threaded so one slow or stuck browser connection cannot block the whole UI.
socketserver.ThreadingTCPServer.allow_reuse_address = True
socketserver.ThreadingTCPServer.daemon_threads = True
with socketserver.ThreadingTCPServer(("0.0.0.0", args.port), handler) as httpd:
    print(f"PathSnap UI on {url}")
    if not args.no_browser:
        webbrowser.open(url)
    httpd.serve_forever()
