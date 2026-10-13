import json
from types import SimpleNamespace
import unittest

from gui import GUI


class GuiDtTests(unittest.TestCase):
    def setUp(self):
        self.ui = GUI(headless=True)
        self.connection = SimpleNamespace(control_period_s=1.0)
        self.ui.bind_player(SimpleNamespace(connection=self.connection))
        self.routes = {route.path: route.endpoint for route in self.ui._build_http_app().routes}

    def test_dt_starts_from_connection_and_updates_before_awake(self):
        self.assertEqual(json.loads(self.routes["/api/status"]().body)["dt"], 1.0)
        response = self.routes["/api/dt"]({"dt": .5})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.connection.control_period_s, .5)
        self.assertEqual(json.loads(self.routes["/api/status"]().body)["dt"], .5)
        page = self.ui._render_page()
        self.assertLess(page.index('id="gain"'), page.index('id="dt"'))

    def test_invalid_dt_does_not_change_period(self):
        for payload in ({}, {"dt": None}, {"dt": "bad"}, {"dt": 0},
                        {"dt": -1}, {"dt": "nan"}, {"dt": "inf"}):
            with self.subTest(payload=payload):
                self.assertEqual(self.routes["/api/dt"](payload).status_code, 400)
                self.assertEqual(self.connection.control_period_s, 1.0)

    def test_unbound_connection(self):
        self.ui._player = None
        self.assertIsNone(json.loads(self.routes["/api/status"]().body)["dt"])
        self.assertEqual(self.routes["/api/dt"]({"dt": 1}).status_code, 409)

    def test_execution_horizon_status_update_and_validation(self):
        server = SimpleNamespace(execution_horizon=8)
        self.ui._player.client = SimpleNamespace(server=server)
        self.assertEqual(json.loads(self.routes["/api/status"]().body)["execution_horizon"], 8)
        endpoint = self.routes["/api/execution-horizon"]
        self.assertEqual(endpoint({"execution_horizon": 16}).status_code, 200)
        self.assertEqual(server.execution_horizon, 16)
        for value in (0, -1, 1.5, "8", True, None):
            self.assertEqual(endpoint({"execution_horizon": value}).status_code, 400)
            self.assertEqual(server.execution_horizon, 16)
        self.ui._player.client = None
        self.assertEqual(endpoint({"execution_horizon": 8}).status_code, 409)
