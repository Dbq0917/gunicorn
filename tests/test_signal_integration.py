#
# This file is part of gunicorn released under the MIT license.
# See the NOTICE for more information.
"""
Integration tests for arbiter signal handling.

These tests start a real gunicorn process and verify signal handling
works correctly with actual requests and signals.
"""

import os
import signal
import socket
import subprocess
import sys
import time

import pytest


# Timeout for CI environments (VMs can be slow, PyPy needs more time)
CI_TIMEOUT = 90


# Simple WSGI app inline
SIMPLE_APP = '''
def application(environ, start_response):
    """Basic hello world response."""
    status = '200 OK'
    body = b'Hello, World!'
    headers = [
        ('Content-Type', 'text/plain'),
        ('Content-Length', str(len(body))),
    ]
    start_response(status, headers)
    return [body]
'''


def find_free_port():
    """Find a free port to bind to."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_for_server(host, port, timeout=CI_TIMEOUT):
    """Wait until server is accepting connections."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.1)
    return False


def make_request(host, port, path='/'):
    """Make a simple HTTP request and return the response body."""
    with socket.create_connection((host, port), timeout=5) as sock:
        request = f'GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n'
        sock.sendall(request.encode())
        response = b''
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
        return response


def worker_pids(master_pid):
    """Return the direct children of a gunicorn master."""
    output = subprocess.check_output(
        ['ps', '-o', 'pid=', '--ppid', str(master_pid)], text=True)
    return {int(pid) for pid in output.split()}


def wait_for_worker_count(master_pid, count, timeout=CI_TIMEOUT,
                          previous=None, samples=4):
    """Wait until exactly ``count`` new workers hold steady."""
    deadline = time.monotonic() + timeout
    last, streak = None, 0
    while time.monotonic() < deadline:
        pids = worker_pids(master_pid)
        if (len(pids) == count
                and (previous is None or not pids & previous)):
            streak = streak + 1 if pids == last else 1
            last = pids
            if streak >= samples:
                return pids
        else:
            last, streak = None, 0
        time.sleep(0.1)
    return worker_pids(master_pid)


@pytest.fixture
def app_module(tmp_path):
    """Create a temporary app module."""
    app_file = tmp_path / "app.py"
    app_file.write_text(SIMPLE_APP)
    return str(app_file.parent), "app:application"


@pytest.fixture
def gunicorn_server(app_module):
    """Start and stop a gunicorn server."""
    app_dir, app_name = app_module
    port = find_free_port()

    # Start gunicorn
    cmd = [
        sys.executable, '-m', 'gunicorn',
        '--bind', f'127.0.0.1:{port}',
        '--workers', '2',
        '--worker-class', 'sync',
        '--access-logfile', '-',
        '--error-logfile', '-',
        '--log-level', 'info',
        '--timeout', '30',
        '--graceful-timeout', '30',
        app_name
    ]

    # Use setsid to create new process group for proper signal handling
    proc = subprocess.Popen(
        cmd,
        cwd=app_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, 'PYTHONPATH': app_dir},
        preexec_fn=os.setsid
    )

    # Wait for server to start
    if not wait_for_server('127.0.0.1', port):
        proc.terminate()
        proc.wait()
        stdout, stderr = proc.communicate()
        pytest.fail(f"Gunicorn failed to start:\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}")

    yield proc, port

    # Cleanup - use process group kill for better cleanup
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait()


class TestSignalHandlingIntegration:
    """Integration tests for signal handling."""

    def test_basic_request(self, gunicorn_server):
        """Verify the server responds to basic requests."""
        proc, port = gunicorn_server

        response = make_request('127.0.0.1', port)
        assert b'Hello, World!' in response

    def test_graceful_shutdown_sigterm(self, gunicorn_server):
        """Verify SIGTERM causes graceful shutdown."""
        proc, port = gunicorn_server

        # Verify server is working
        response = make_request('127.0.0.1', port)
        assert b'Hello, World!' in response

        # Send SIGTERM to the process group for reliable signal delivery
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            proc.send_signal(signal.SIGTERM)

        # Wait for process to exit
        try:
            exit_code = proc.wait(timeout=CI_TIMEOUT)
            assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Gunicorn did not exit within timeout after SIGTERM")

    @pytest.mark.skipif(
        hasattr(sys, 'pypy_version_info'),
        reason="SIGINT handling differs on PyPy, use SIGTERM test instead"
    )
    def test_graceful_shutdown_sigint(self, gunicorn_server):
        """Verify SIGINT causes graceful shutdown."""
        proc, port = gunicorn_server

        # Verify server is working
        response = make_request('127.0.0.1', port)
        assert b'Hello, World!' in response

        # Send SIGINT to the process group for reliable signal delivery
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            proc.send_signal(signal.SIGINT)

        # Wait for process to exit
        try:
            exit_code = proc.wait(timeout=CI_TIMEOUT)
            assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Gunicorn did not exit within timeout after SIGINT")

    def test_sighup_reload(self, gunicorn_server):
        """Verify SIGHUP triggers reload."""
        proc, port = gunicorn_server

        # Verify server is working
        response = make_request('127.0.0.1', port)
        assert b'Hello, World!' in response

        # Send SIGHUP to the master process (not process group - only master handles reload)
        proc.send_signal(signal.SIGHUP)

        # Wait a moment for reload
        time.sleep(2)

        # Verify server still works after reload
        assert proc.poll() is None, "Server died after SIGHUP"
        response = make_request('127.0.0.1', port)
        assert b'Hello, World!' in response

    def test_three_sighup_reloads_keep_the_worker_count(self, gunicorn_server):
        """Each reload hands its listeners to the new workers before retiring
        the old ones, so repeated rolling restarts must not shrink the pool."""
        proc, port = gunicorn_server
        expected = 2
        assert len(wait_for_worker_count(proc.pid, expected)) == expected

        for _ in range(3):
            before = worker_pids(proc.pid)
            proc.send_signal(signal.SIGHUP)
            after = wait_for_worker_count(proc.pid, expected, previous=before)

            assert proc.poll() is None, "Server died after SIGHUP"
            assert len(after) == expected
            assert after != before, "reload did not replace the workers"
            response = make_request('127.0.0.1', port)
            assert b'Hello, World!' in response

    def test_multiple_requests_under_load(self, gunicorn_server):
        """Verify server handles multiple concurrent requests."""
        proc, port = gunicorn_server

        # Make several requests in sequence
        for _ in range(10):
            response = make_request('127.0.0.1', port)
            assert b'Hello, World!' in response

        # Verify server is still running
        assert proc.poll() is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
