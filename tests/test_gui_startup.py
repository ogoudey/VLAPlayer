import threading
import unittest
from unittest.mock import MagicMock

from clients.client import Client
from gui import GUI
from player import Player


class GuiStartupTests(unittest.TestCase):
    def test_start_waits_for_slow_camera_and_returns_while_inference_runs(self):
        ui = GUI(headless=True)
        ui.report_loop_event = MagicMock()
        camera_initializing = threading.Event()
        release_camera = threading.Event()
        inference_started = threading.Event()
        release_inference = threading.Event()
        inference_finished = threading.Event()

        def initialize_camera():
            camera_initializing.set()
            if not release_camera.wait(2):
                raise TimeoutError("test did not release camera")

        def inference_loop():
            inference_started.set()
            release_inference.wait(2)
            inference_finished.set()

        client = Client()
        client.server = MagicMock()
        client.start_inference_loop = inference_loop
        connection = MagicMock()
        cameras = MagicMock()
        cameras.awake.side_effect = initialize_camera
        player = Player(client, connection, cameras, ui)
        ui.bind_player(player)
        routes = {route.path: route.endpoint for route in ui._build_http_app().routes}

        try:
            ui.awake()
            self.assertTrue(camera_initializing.wait(1))
            # Reproduce clicking Start while the onboard camera is waking.
            self.assertEqual(routes["/api/toggle"]({"prompt": "pick up cube"}).status_code, 200)
            ui.start()  # repeated Start must not launch a second worker
            self.assertIsNone(client.connection)
            self.assertFalse(inference_started.is_set())
            self.assertFalse(ui.wait_for_start(timeout=0))
            self.assertFalse(ui._status["awake"])
            self.assertTrue(ui._status["initializing"])
            self.assertTrue(ui._status["starting"])
            self.assertFalse(ui._status["started"])
            release_camera.set()
            self.assertTrue(inference_started.wait(1))
            self.assertIs(client.connection, connection)
            self.assertIs(client.camera_set, cameras)
            self.assertTrue(ui.wait_for_start(timeout=0))
            self.assertTrue(ui._status["awake"])
            self.assertTrue(ui._status["started"])
            self.assertFalse(ui._status["initializing"])
            self.assertFalse(ui._status["starting"])
            self.assertEqual(client.language, "pick up cube")
            connection.awake.assert_called_once()
            cameras.awake.assert_called_once()
            cameras.start_recording.assert_called_once()
            ui.report_loop_event.assert_not_called()
        finally:
            release_camera.set()
            release_inference.set()
            inference_finished.wait(1)

    def test_initialization_failure_does_not_start_and_can_be_retried(self):
        ui = GUI(headless=True)
        failed = threading.Event()
        started = threading.Event()

        def report(message, **kwargs):
            if message.startswith("Player start failed:"):
                failed.set()

        ui.report_loop_event = MagicMock(side_effect=report)
        player = MagicMock()
        player.awake.side_effect = [RuntimeError("camera unavailable"), None]
        player.start.side_effect = started.set
        ui.bind_player(player)
        ui.start()
        self.assertTrue(failed.wait(1))
        player.start.assert_not_called()
        self.assertFalse(ui.wait_for_start(timeout=0))
        self.assertFalse(ui._status["awake"])
        self.assertFalse(ui._status["started"])
        self.assertFalse(ui._status["initializing"])
        self.assertFalse(ui._status["starting"])

        ui.start()
        self.assertTrue(started.wait(1))
        self.assertEqual(player.awake.call_count, 2)
        player.start.assert_called_once()
        self.assertTrue(ui.wait_for_start(timeout=0))
        self.assertTrue(ui._status["awake"])
        self.assertTrue(ui._status["started"])


if __name__ == "__main__":
    unittest.main()
