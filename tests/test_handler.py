import os
import sys
import types
import unittest
from unittest.mock import patch

voxcpm_stub = types.ModuleType("voxcpm")


class DummyVoxCPM:
    pass


voxcpm_stub.VoxCPM = DummyVoxCPM
sys.modules.setdefault("voxcpm", voxcpm_stub)

from config import get_config
from handler import build_error_response, health_check, validate_request


class HandlerTests(unittest.TestCase):
    def test_rejects_missing_text(self):
        request, error = validate_request({"text": "   "})
        self.assertIsNone(request)
        self.assertIn("text", error.lower())

    def test_rejects_invalid_base64(self):
        request, error = validate_request({"text": "hi", "speakerAudio": "not-base64"})
        self.assertIsNone(request)
        self.assertIn("base64", error.lower())

    def test_builds_error_response(self):
        payload = build_error_response("boom")
        self.assertIn("error", payload)
        self.assertEqual(payload["error"], "boom")

    def test_health_check_shape(self):
        payload = health_check()
        self.assertIn("status", payload)
        self.assertIn("worker", payload)
        self.assertIn("gpu", payload)

    def test_defaults_to_lazy_startup(self):
        with patch.dict(os.environ, {}, clear=True):
            config = get_config()
            self.assertFalse(config["preload_model_on_startup"])


if __name__ == "__main__":
    unittest.main()
