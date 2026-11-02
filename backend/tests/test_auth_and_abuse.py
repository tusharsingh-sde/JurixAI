import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath('backend'))
import backend
from backend import app, limiter

class TestAuthAndAbuse(unittest.TestCase):
    def setUp(self):
        limiter.reset()
        self.client = TestClient(app)

    def test_signup_short_password_returns_422(self):
        """Signup with password under 8 characters must return 422."""
        resp = self.client.post("/signup", json={
            "email": "valid@example.com",
            "password": "short"
        })
        self.assertEqual(resp.status_code, 422)

    def test_signup_invalid_email_returns_422(self):
        """Signup with malformed email must return 422."""
        resp = self.client.post("/signup", json={
            "email": "not-an-email",
            "password": "ValidPassword123"
        })
        self.assertEqual(resp.status_code, 422)

    def test_login_wrong_password_returns_401_without_leaks(self):
        """Invalid credentials must return 401 with generic error message."""
        with patch("backend.supabase_auth.auth.sign_in_with_password") as mock_signin:
            mock_signin.side_effect = Exception("Supabase Auth API internal error stack trace")
            resp = self.client.post("/login", json={
                "email": "user@example.com",
                "password": "WrongPassword123"
            })
            self.assertEqual(resp.status_code, 401)
            self.assertEqual(resp.json()["detail"], "Invalid email or password.")
            self.assertNotIn("stack trace", resp.text.lower())

    def test_signup_rate_limit_and_anti_spoofing(self):
        """Signup enforces 5/hour per IP; spoofed X-Forwarded-For must NOT bypass the limit."""
        with patch("backend.supabase_admin.auth.admin.create_user") as mock_create:
            mock_create.return_value = MagicMock()

            # First 5 signups should succeed
            for i in range(5):
                resp = self.client.post(
                    "/signup",
                    json={"email": f"user{i}@example.com", "password": "SecurePassword123!"},
                    headers={"X-Forwarded-For": f"198.51.100.{i}"}  # Spoofed headers
                )
                self.assertEqual(resp.status_code, 200, f"Attempt {i+1} failed: {resp.text}")

            # 6th attempt must be rejected with 429 Too Many Requests
            resp6 = self.client.post(
                "/signup",
                json={"email": "user5@example.com", "password": "SecurePassword123!"},
                headers={"X-Forwarded-For": "198.51.100.99"}
            )
            self.assertEqual(resp6.status_code, 429, "6th signup attempt must return 429 Too Many Requests")

if __name__ == "__main__":
    unittest.main()

