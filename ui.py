from __future__ import annotations

import threading
from typing import Optional


class UI:
    STATIC_LANGUAGE: str = "do something useful"

    def __init__(self, headless: bool = True, direct_start: bool = False):
        self.language = UI.STATIC_LANGUAGE

        self.client = None          # the model client (latency to be displayed)
        self.connection = None      # the "robot" (some stats perhaps to be displayed)
        self.camera_set = None      # a set of CameraConnection objects (to be displayed)

        self.headless = headless

        self._start_event = threading.Event()

        if direct_start:
            self.start()

    def start(self) -> None:
        # This is the trigger for someone else waiting at ui.start().
        self._start_event.set()

    def wait_for_start(self, timeout: Optional[float] = None) -> bool:
        """Block until start() has been called (or timeout elapses)."""
        return self._start_event.wait(timeout)

    @property
    def started(self) -> bool:
        return self._start_event.is_set()