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
        backend.reset_global_daily_counter()
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

    def test_guest_chat_works_without_db_writes(self):
        """POST /chat without token returns 200, session_id is None, guest is True, and 0 DB queries."""
        with patch("backend.supabase.table") as mock_table, \
             patch("backend.invoke_gemini", return_value="Here is legal guidance."):
            app.state.db = MagicMock()
            app.state.db.similarity_search.return_value = []

            resp = self.client.post("/chat", json={
                "message": "What is the penalty for simple theft under BNS?"
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIsNone(data["session_id"])
            self.assertTrue(data["guest"])
            self.assertEqual(data["reply"], "Here is legal guidance.")
            mock_table.assert_not_called()

    def test_guest_followup_and_doc_context_injected_safely(self):
        """Guest history and document text are injected into prompt and HTML/XML tags are escaped."""
        captured_prompts = []
        def fake_invoke(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return "Analysis response"

        with patch("backend.supabase.table") as mock_table, \
             patch("backend.invoke_gemini", side_effect=fake_invoke):
            app.state.db = MagicMock()
            app.state.db.similarity_search.return_value = []

            resp = self.client.post("/chat", json={
                "message": "Can they arrest me <script>alert(1)</script>?",
                "document_text": "FIR Details: Section 302 <breakout>",
                "document_name": "FIR <1>.pdf",
                "guest_history": [
                    {"role": "User", "content": "I got a notice <tag>"},
                    {"role": "JurixAI", "content": "Notice under section 41A <tag>"}
                ]
            })
            self.assertEqual(resp.status_code, 200)
            mock_table.assert_not_called()

            prompt = captured_prompts[0]
            # Verify history was injected
            self.assertIn("User: I got a notice &lt;tag&gt;", prompt)
            self.assertIn("JurixAI: Notice under section 41A &lt;tag&gt;", prompt)
            # Verify document was injected
            self.assertIn("USER UPLOADED DOCUMENT (FIR &lt;1&gt;.pdf):", prompt)
            self.assertIn("FIR Details: Section 302 &lt;breakout&gt;", prompt)
            # Verify user message was escaped
            self.assertIn("Can they arrest me &lt;script&gt;alert(1)&lt;/script&gt;?", prompt)

    def test_invalid_or_stale_token_strictly_returns_401(self):
        """Invalid or stale token must return 401 on /chat and /upload, never falling back to guest."""
        # /chat with bad token
        resp_chat = self.client.post(
            "/chat",
            headers={"Authorization": "Bearer bad-expired-token"},
            json={"message": "Help me"}
        )
        self.assertEqual(resp_chat.status_code, 401)
        self.assertIn("Invalid or expired token", resp_chat.json()["detail"])

        # /upload with bad token
        resp_upload = self.client.post(
            "/upload",
            headers={"Authorization": "Bearer bad-expired-token"},
            files={"file": ("test.pdf", b"%PDF-1.4 sample", "application/pdf")}
        )
        self.assertEqual(resp_upload.status_code, 401)
        self.assertIn("Invalid or expired token", resp_upload.json()["detail"])

    def test_tighter_guest_rate_limits(self):
        """Unauthenticated guests have 3 req/hour cap on /upload and 5 req/10min cap on /chat."""
        # /upload guest cap: 3 allowed, 4th rejected with 429
        with patch("backend.extract_document_text_sync", return_value=("Doc text", False)):
            for i in range(3):
                resp = self.client.post(
                    "/upload",
                    files={"file": ("test.pdf", b"%PDF-1.4 sample", "application/pdf")}
                )
                self.assertEqual(resp.status_code, 200, f"Guest upload {i+1} failed: {resp.text}")

            resp_upload_4 = self.client.post(
                "/upload",
                files={"file": ("test.pdf", b"%PDF-1.4 sample", "application/pdf")}
            )
            self.assertEqual(resp_upload_4.status_code, 429)
            self.assertIn("Guest limit reached", resp_upload_4.json()["detail"])

        # Reset rate limiter for chat test
        limiter.reset()
        backend._user_rate_buckets.clear()
        backend.reset_global_daily_counter()

        # /chat guest cap: 5 allowed (GUEST_CHAT_PER_10MIN), 6th rejected with 429
        with patch("backend.invoke_gemini", return_value="Reply"):
            app.state.db = MagicMock()
            app.state.db.similarity_search.return_value = []
            for i in range(5):
                resp = self.client.post(
                    "/chat",
                    json={"message": f"Question {i+1}"}
                )
                self.assertEqual(resp.status_code, 200, f"Guest chat {i+1} failed: {resp.text}")

            resp_chat_6 = self.client.post(
                "/chat",
                json={"message": "Question 6"}
            )
            self.assertEqual(resp_chat_6.status_code, 429)
            self.assertIn("Guest limit reached. Please sign in for higher limits or try again later.", resp_chat_6.json()["detail"])

    def test_global_daily_kill_switch_blocks_chat_without_invoking_gemini(self):
        """When MAX_GLOBAL_DAILY_REQUESTS is reached, /chat returns 429 and invoke_gemini is never called."""
        with patch("backend.MAX_GLOBAL_DAILY_REQUESTS", 2), \
             patch("backend.invoke_gemini", return_value="AI Reply") as mock_gemini:
            app.state.db = MagicMock()
            app.state.db.similarity_search.return_value = []

            # 1st request succeeds
            r1 = self.client.post("/chat", json={"message": "Question 1"})
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(mock_gemini.call_count, 1)

            # 2nd request succeeds
            r2 = self.client.post("/chat", json={"message": "Question 2"})
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(mock_gemini.call_count, 2)

            # 3rd request hits global limit (2) and must immediately return 429
            r3 = self.client.post("/chat", json={"message": "Question 3"})
            self.assertEqual(r3.status_code, 429)
            self.assertIn(
                "JurixAI has reached its free daily usage limit due to high traffic. Please try again tomorrow.",
                r3.json()["detail"],
            )
            # Crucial verification: invoke_gemini was NOT called for the blocked request
            self.assertEqual(mock_gemini.call_count, 2)

    def test_global_daily_counter_resets_on_utc_date_rollover(self):
        """Global daily counter automatically resets when the UTC date advances."""
        from datetime import timedelta
        yesterday = backend._global_daily_date - timedelta(days=1)
        backend.reset_global_daily_counter(count=500, date=yesterday)

        with patch("backend.MAX_GLOBAL_DAILY_REQUESTS", 250), \
             patch("backend.invoke_gemini", return_value="AI Reply"):
            app.state.db = MagicMock()
            app.state.db.similarity_search.return_value = []

            resp = self.client.post("/chat", json={"message": "Question on new day"})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(backend._global_daily_count, 1)

    def test_guest_message_length_limit(self):
        """Guest messages > 1500 chars are rejected with 400, while authenticated users can send up to 4000."""
        with patch("backend.invoke_gemini", return_value="AI Reply"):
            app.state.db = MagicMock()
            app.state.db.similarity_search.return_value = []

            # Guest sending exactly 1500 characters succeeds
            resp_ok = self.client.post("/chat", json={"message": "A" * 1500})
            self.assertEqual(resp_ok.status_code, 200)

            # Guest sending 1501 characters is rejected with 400
            resp_too_long = self.client.post("/chat", json={"message": "A" * 1501})
            self.assertIn(resp_too_long.status_code, (400, 422))
            self.assertIn("Guest messages cannot exceed 1500 characters", resp_too_long.json()["detail"])

        # Authenticated user can send message > 1500 chars (up to MAX_MESSAGE_CHARS=4000)
        mock_user = MagicMock()
        mock_user.id = "00000000-0000-0000-0000-000000000001"
        mock_user.email = "authed@example.com"
        app.dependency_overrides[backend.get_current_user] = lambda: mock_user
        try:
            with patch("backend.supabase.table") as mock_table, \
                 patch("backend.invoke_gemini", return_value="AI Reply"):
                mock_table.return_value.insert.return_value.execute.return_value.data = [{"id": "new-sess-1"}]
                mock_table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = []
                app.state.db = MagicMock()
                app.state.db.similarity_search.return_value = []

                resp_auth = self.client.post(
                    "/chat",
                    headers={"Authorization": "Bearer valid-token"},
                    json={"message": "A" * 2000}
                )
                self.assertEqual(resp_auth.status_code, 200)
        finally:
            app.dependency_overrides.pop(backend.get_current_user, None)

    def test_gemini_model_max_output_tokens(self):
        """get_gemini_model configures max_output_tokens from GEMINI_MAX_OUTPUT_TOKENS."""
        model = backend.get_gemini_model("gemini-2.5-flash")
        self.assertEqual(model.max_output_tokens, backend.GEMINI_MAX_OUTPUT_TOKENS)

    def test_sessions_strictly_protected_for_guests(self):
        """GET /sessions without token returns 401/403, guest cannot view sessions."""
        resp = self.client.get("/sessions")
        self.assertIn(resp.status_code, (401, 403))

if __name__ == "__main__":
    unittest.main()

