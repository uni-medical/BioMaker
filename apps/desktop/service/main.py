"""Portable Python entry point. Standard output is a JSON readiness channel."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading

import uvicorn

from desktop_service import DesktopConfig, create_desktop_app


class DesktopServer(uvicorn.Server):
    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        if self.started and sockets:
            print(
                json.dumps({"type": "ready", "port": sockets[0].getsockname()[1]}),
                flush=True,
            )


def main() -> int:
    if os.name == "posix":
        # This process holds API/session credentials. Do not persist its memory
        # in a core file after a native-extension or interpreter crash.
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ImportError, AttributeError, OSError, ValueError):
            pass
    try:
        config = DesktopConfig.from_environment()
        application = create_desktop_app(config)
    except Exception as exc:
        # Startup errors contain filesystem/config names, never the session token.
        print(
            f"BioMaker desktop could not start: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        server = DesktopServer(
            uvicorn.Config(
                application,
                host="127.0.0.1",
                port=0,
                log_level="warning",
                access_log=False,
            )
        )
        finished = threading.Event()
        parent = os.environ.get("BIOMAKER_DESKTOP_PARENT_PID")
        if parent:
            try:
                parent_pid = int(parent)
                if parent_pid <= 1 or os.getppid() != parent_pid:
                    raise ValueError(
                        "The desktop parent process is no longer available"
                    )
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1

            def watch_parent():
                while not finished.wait(1):
                    if os.getppid() != parent_pid:
                        server.should_exit = True
                        return

            threading.Thread(
                target=watch_parent, daemon=True, name="biomaker-desktop-parent"
            ).start()
        try:
            server.run(sockets=[listener])
        finally:
            finished.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
