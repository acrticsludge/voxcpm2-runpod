import sys
import types
import unittest

voxcpm_stub = types.ModuleType("voxcpm")


class DummyVoxCPM:
    pass


voxcpm_stub.VoxCPM = DummyVoxCPM
sys.modules.setdefault("voxcpm", voxcpm_stub)

from pod_entrypoint import handle_pod_payload


class PodEntryPointTests(unittest.TestCase):
    def test_health_payload_returns_status_shape(self):
        payload = handle_pod_payload({"input": {"health": True}})
        self.assertIn("status", payload)
        self.assertIn("worker", payload)

    def test_text_payload_returns_error_for_missing_text(self):
        payload = handle_pod_payload({"input": {"text": "   "}})
        self.assertIn("error", payload)


if __name__ == "__main__":
    unittest.main()
