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
        backend._user_rate_buckets.clear()
        self.client = TestClient(app)

    def test_signup_short_password_returns_422(self):
        """Signup with password under 8 characters must return 422."""
        resp = self.client.post("/signup", json={
            "email": "valid@example.com",
            "password": "short"
        })
        self.assertEqual(resp.status_code, 422)

    def test_signup_overlong_password_returns_422(self):
        """Signup with password over 128 characters must return 422 to prevent DoS."""
        resp = self.client.post("/signup", json={
            "email": "valid@example.com",
            "password": "A" * 129
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
        mock_auth_client = MagicMock()
        mock_auth_client.auth.sign_in_with_password.side_effect = Exception("Supabase Auth API internal error stack trace")
        with patch("backend._new_auth_client", return_value=mock_auth_client):
            resp = self.client.post("/login", json={
                "email": "user@example.com",
                "password": "WrongPassword123"
            })
            self.assertEqual(resp.status_code, 401)
            self.assertEqual(resp.json()["detail"], "Invalid email or password.")
            self.assertNotIn("stack trace", resp.text.lower())

    def test_signup_rate_limit_and_anti_spoofing(self):
        """Signup enforces 5/hour per IP; uses standard signup (no auto-confirm) and prevents spoofing."""
        mock_auth_client = MagicMock()
        mock_auth_client.auth.sign_up.return_value = MagicMock(user=MagicMock(id="user-123"))

        with patch("backend._new_auth_client", return_value=mock_auth_client), \
             patch("backend.supabase_admin.auth.admin.create_user") as mock_admin_create:

            # First 5 signups should succeed
            for i in range(5):
                resp = self.client.post(
                    "/signup",
                    json={"email": f"user{i}@example.com", "password": "SecurePassword123!"},
                    headers={"X-Forwarded-For": f"198.51.100.{i}"}  # Spoofed headers
                )
                self.assertEqual(resp.status_code, 200, f"Attempt {i+1} failed: {resp.text}")
                self.assertIn("verify your account", resp.json()["message"].lower())

            # Verify admin create_user (email_confirm=True) was NEVER called (pre-hijacking defense)
            mock_admin_create.assert_not_called()

            # 6th attempt must be rejected with 429 Too Many Requests
            resp6 = self.client.post(
                "/signup",
                json={"email": "user5@example.com", "password": "SecurePassword123!"},
                headers={"X-Forwarded-For": "198.51.100.99"}
            )
            self.assertEqual(resp6.status_code, 429, "6th signup attempt must return 429 Too Many Requests")

    def test_fake_token_flood_blocked_by_ip_shield(self):
        """Random fake Bearer tokens must NOT bypass rate limiting; outer IP limiter catches them at 20/min."""
        # 20 requests with different fake Bearer tokens from same IP
        for i in range(20):
            resp = self.client.post(
                "/upload",
                headers={"Authorization": f"Bearer fake-token-{i}"},
                files={"file": ("test.txt", b"hello world", "text/plain")}
            )
            self.assertEqual(resp.status_code, 401)

        # 21st attempt must be rejected by outer IP rate limiter with 429 before reaching auth
        resp21 = self.client.post(
            "/upload",
            headers={"Authorization": "Bearer fake-token-21"},
            files={"file": ("test.txt", b"hello world", "text/plain")}
        )
        self.assertEqual(resp21.status_code, 429, "21st upload attempt from same IP must be 429")

    def test_verified_user_rate_limit(self):
        """Authenticated user cannot exceed user-level rate limit (10/min for upload)."""
        mock_user = MagicMock()
        mock_user.id = "00000000-0000-0000-0000-000000000001"
        mock_user.email = "test@example.com"

        app.dependency_overrides[backend.get_current_user] = lambda: mock_user
        try:
            with patch("backend.extract_document_text_sync", return_value=("Sample extracted text", False)):
                # Send 10 valid uploads
                for i in range(10):
                    resp = self.client.post(
                        "/upload",
                        headers={"Authorization": "Bearer valid-token"},
                        files={"file": ("test.pdf", b"%PDF-1.4 legal document", "application/pdf")}
                    )
                    self.assertEqual(resp.status_code, 200, f"Upload {i+1} failed: {resp.text}")

                # 11th request must trigger user rate limit (429)
                resp11 = self.client.post(
                    "/upload",
                    headers={"Authorization": "Bearer valid-token"},
                    files={"file": ("test.pdf", b"%PDF-1.4 legal document", "application/pdf")}
                )
                self.assertEqual(resp11.status_code, 429)
                self.assertIn("Rate limit exceeded (10 requests per 60s)", resp11.json()["detail"])
        finally:
            app.dependency_overrides.pop(backend.get_current_user, None)

if __name__ == "__main__":
    unittest.main()

