import atexit
import os
import signal
import socket
import sys
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path


DEFAULT_NTFY_TOPIC = "yaolong-stabl-runs"


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


class RunNotifier:
    def __init__(
        self,
        run_name,
        log_path=None,
        topic=None,
        script_path=None,
        notify=True,
    ):
        self.run_name = run_name
        self.script_path = Path(script_path or sys.argv[0]).resolve()
        self.log_path = Path(log_path or self.script_path.with_suffix(".log"))
        self.topic = topic or os.environ.get("STABL_NTFY_TOPIC", DEFAULT_NTFY_TOPIC)
        self.notify = notify and os.environ.get("STABL_DISABLE_NTFY") != "1"
        self.start_time = datetime.now()
        self.hostname = socket.gethostname()
        self.failed = False
        self.exit_reason = "completed"
        self._log_file = None
        self._previous_excepthook = sys.excepthook
        self._previous_stdout = sys.stdout
        self._previous_stderr = sys.stderr
        self._finished = False

    def install(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, self._log_file)
        sys.stderr = _Tee(sys.__stderr__, self._log_file)

        print(f"===== {self.run_name} started at {self.start_time} on {self.hostname} =====")
        print(f"===== Log file: {self.log_path} =====")

        self.send(
            title=f"{self.run_name} started",
            message=(
                f"{self.script_path.name} started on {self.hostname}. "
                f"Output is being written to {self.log_path.name}."
            ),
            priority="default",
            tags="rocket",
        )

        sys.excepthook = self._handle_exception
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        atexit.register(self.finish)
        return self

    def send(self, title, message, priority="high", tags="bell"):
        if not self.notify:
            return

        req = urllib.request.Request(
            f"https://ntfy.sh/{self.topic}",
            data=message.encode("utf-8"),
            method="POST",
            headers={
                "Title": title,
                "Priority": priority,
                "Tags": tags,
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                response.read()
        except Exception as error:
            print(f"[ntfy failed] {error}", file=sys.__stderr__)

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        self.failed = True
        self.exit_reason = f"exception: {exc_type.__name__}"
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    def _handle_signal(self, signum, frame):
        self.failed = True
        self.exit_reason = f"terminated by signal {signum}"
        print(f"\n===== {self.run_name} interrupted by signal {signum} =====", flush=True)
        raise SystemExit(128 + signum)

    def finish(self):
        if self._finished:
            return
        self._finished = True

        end_time = datetime.now()
        elapsed = end_time - self.start_time

        print()
        print(f"===== {self.run_name} ended at {end_time} =====")
        print(f"===== Elapsed time: {elapsed} =====")
        print(f"===== Status: {'FAILED' if self.failed else 'SUCCESS'} =====")
        print(f"===== Reason: {self.exit_reason} =====")
        print(f"===== Log file: {self.log_path} =====", flush=True)

        if self.failed:
            self.send(
                title=f"{self.run_name} failed",
                message=(
                    f"{self.script_path.name} did not finish successfully. "
                    f"Reason: {self.exit_reason}. Check {self.log_path.name}. "
                    f"Elapsed: {elapsed}"
                ),
                priority="urgent",
                tags="warning",
            )
        else:
            self.send(
                title=f"{self.run_name} finished",
                message=(
                    f"{self.script_path.name} finished on {self.hostname}. "
                    f"Log file: {self.log_path.name}. Elapsed: {elapsed}"
                ),
                priority="high",
                tags="white_check_mark",
            )

        if self._log_file is not None:
            try:
                sys.stdout = self._previous_stdout
                sys.stderr = self._previous_stderr
                sys.excepthook = self._previous_excepthook
                self._log_file.close()
            except Exception:
                pass


def install_run_notifier(run_name, log_name=None, topic=None, notify=True):
    script_path = Path(sys.argv[0]).resolve()
    log_path = script_path.with_name(log_name) if log_name else None
    return RunNotifier(
        run_name=run_name,
        log_path=log_path,
        topic=topic,
        script_path=script_path,
        notify=notify,
    ).install()
