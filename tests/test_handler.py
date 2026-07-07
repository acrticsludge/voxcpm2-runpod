import unittest

from handler import build_error_response, validate_request


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


if __name__ == "__main__":
    unittest.main()
