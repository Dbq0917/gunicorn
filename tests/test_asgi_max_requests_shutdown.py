#
# This file is part of gunicorn released under the MIT license.
# See the NOTICE for more information.

"""An ASGI worker that stops (max_requests) lets an in-flight request
finish, up to graceful_timeout: it keeps notifying the master while it
drains, so the master does not kill it as hung once ``timeout`` expires.
"""

import socket
import subprocess
import sys
import threading
import time

import pytest

APP = '''
import asyncio

async def app(scope, receive, send):
    if scope["path"] == "/slow":
        await asyncio.sleep(6)
    body = scope["path"].encode()
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-length", str(len(body)))]})
    await send({"type": "http.response.body", "body": body})
'''


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    (tmp_path / "app.py").write_text(APP)
    port = _free_port()
    log = tmp_path / "gunicorn.log"
    # timeout < the /slow duration < graceful_timeout: the drain only
    # survives the master's timeout check if the worker keeps notifying.
    proc = subprocess.Popen(
        [sys.executable, "-m", "gunicorn", "app:app",
         "--bind", f"127.0.0.1:{port}", "--workers", "1",
         "--worker-class", "asgi",
         "--max-requests", "2", "--timeout", "3",
         "--graceful-timeout", "30", "--log-level", "info"],
        cwd=str(tmp_path), stdout=log.open("w"), stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.3):
                break
        except OSError:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail(f"gunicorn did not start:\n{log.read_text()}")
    yield port, log
    proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _request(port, path):
    sock = socket.create_connection(("127.0.0.1", port), 30)
    sock.sendall(b"GET " + path + b" HTTP/1.1\r\nHost: localhost\r\n"
                 b"Connection: close\r\n\r\n")
    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    sock.close()
    return data


def test_in_flight_request_finishes_when_worker_stops(server):
    port, log = server
    slow = {}

    def run_slow():
        slow["response"] = _request(port, b"/slow")

    thread = threading.Thread(target=run_slow)
    thread.start()
    time.sleep(0.5)

    # max_requests counts completed requests, so two fast ones reach it
    # while /slow is running: the worker stops and drains for longer
    # than --timeout.
    response = _request(port, b"/fast")
    assert response.startswith(b"HTTP/1.1 200"), response
    response = _request(port, b"/fast")
    assert response.startswith(b"HTTP/1.1 200"), response

    thread.join(30)
    assert slow["response"].startswith(b"HTTP/1.1 200"), slow
    assert slow["response"].endswith(b"/slow")
    assert "WORKER TIMEOUT" not in log.read_text()
