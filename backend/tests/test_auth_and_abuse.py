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

    def test_is_small_talk_classification(self):
        """is_small_talk returns True for greetings/acknowledgements and False for legal queries."""
        # True cases: short greetings, closings, acknowledgements under 5 words
        self.assertTrue(backend.is_small_talk("hi"))
        self.assertTrue(backend.is_small_talk("Hello JurixAI"))
        self.assertTrue(backend.is_small_talk("Namaste, kaise ho?"))
        self.assertTrue(backend.is_small_talk("thank you so much"))
        self.assertTrue(backend.is_small_talk("ok got it"))
        self.assertTrue(backend.is_small_talk("bye tc"))
        self.assertTrue(backend.is_small_talk("theek hai shukriya"))

        # False cases: contains legal keywords even if short
        self.assertFalse(backend.is_small_talk("hi what is bail"))
        self.assertFalse(backend.is_small_talk("FIR kaise darj karein?"))
        self.assertFalse(backend.is_small_talk("Section 138 notice"))
        self.assertFalse(backend.is_small_talk("police arrested my friend"))
        self.assertFalse(backend.is_small_talk("IPC 420 kya hai?"))
        self.assertFalse(backend.is_small_talk("consumer complaint process"))

        # False cases: longer than 5 words
        self.assertFalse(backend.is_small_talk("hello how are you doing today my friend"))
        # False cases: empty or whitespace
        self.assertFalse(backend.is_small_talk(""))
        self.assertFalse(backend.is_small_talk("   "))

    def test_chat_skips_rag_on_small_talk(self):
        """Small talk messages skip ChromaDB similarity_search completely and use minimal prompt."""
        captured_prompts = []
        with patch("backend.invoke_gemini", side_effect=lambda p: (captured_prompts.append(p), "Hello! How can I help?")[1]):
            mock_db = MagicMock()
            app.state.db = mock_db

            resp = self.client.post("/chat", json={"message": "hello jurixai"})
            self.assertEqual(resp.status_code, 200)

            # Verification: Chroma DB was NEVER queried for small talk
            mock_db.similarity_search.assert_not_called()

            # Verification: Minimal prompt was used (< 600 characters)
            self.assertEqual(len(captured_prompts), 1)
            self.assertLess(len(captured_prompts[0]), 600)
            self.assertIn("The user is greeting you", captured_prompts[0])

    def test_chat_performs_rag_on_legal_query(self):
        """Legal queries trigger ChromaDB similarity_search with RAG_TOP_K and full grounding prompt."""
        captured_prompts = []
        with patch("backend.invoke_gemini", side_effect=lambda p: (captured_prompts.append(p), "Here is legal guidance")[1]):
            mock_db = MagicMock()
            doc_mock = MagicMock(page_content="Section 436 CrPC deals with bailable offences.")
            mock_db.similarity_search.return_value = [doc_mock]
            app.state.db = mock_db

            resp = self.client.post("/chat", json={"message": "Can I get bail for bailable offence?"})
            self.assertEqual(resp.status_code, 200)

            # Verification: Chroma DB WAS queried with k=RAG_TOP_K (default: 3)
            mock_db.similarity_search.assert_called_once_with("Can I get bail for bailable offence?", k=backend.RAG_TOP_K)
            self.assertEqual(len(captured_prompts), 1)
            self.assertIn("LEGAL KNOWLEDGE BASE:", captured_prompts[0])
            self.assertIn("Section 436 CrPC deals with bailable offences.", captured_prompts[0])

    def test_chat_does_not_skip_rag_when_document_present(self):
        """When an uploaded document is present, even a greeting does NOT skip RAG/document pipeline."""
        captured_prompts = []
        with patch("backend.invoke_gemini", side_effect=lambda p: (captured_prompts.append(p), "Document review response")[1]):
            mock_db = MagicMock()
            mock_db.similarity_search.return_value = []
            app.state.db = mock_db

            resp = self.client.post("/chat", json={
                "message": "hello",
                "document_text": "This is a rental deed between landlord and tenant.",
                "document_name": "Rental_Agreement.pdf"
            })
            self.assertEqual(resp.status_code, 200)

            # Verification: Chroma similarity_search WAS called
            mock_db.similarity_search.assert_called_once()
            self.assertEqual(len(captured_prompts), 1)
            self.assertIn("<uploaded_document>", captured_prompts[0])
            self.assertIn("Rental_Agreement.pdf", captured_prompts[0])

    def test_guest_history_and_context_budget_truncation(self):
        """Guest history is capped to MAX_HISTORY_MESSAGES=4, 700 chars per message, doc text capped at 6000."""
        captured_prompts = []
        with patch("backend.invoke_gemini", side_effect=lambda p: (captured_prompts.append(p), "AI reply")[1]):
            mock_db = MagicMock()
            mock_db.similarity_search.return_value = []
            app.state.db = mock_db

            # Create 7 guest history items each 1000 chars long
            history_input = [
                {"role": "User", "content": f"Turn {i} " + ("X" * 1000)}
                for i in range(7)
            ]
            long_doc = "D" * 10000

            resp = self.client.post("/chat", json={
                "message": "Explain my legal notice",
                "guest_history": history_input,
                "document_text": long_doc,
                "document_name": "Notice.pdf"
            })
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(len(captured_prompts), 1)
            prompt = captured_prompts[0]

            # Only the last 4 turns (3, 4, 5, 6) should be present; older turns (0, 1, 2) omitted
            self.assertNotIn("Turn 0", prompt)
            self.assertNotIn("Turn 1", prompt)
            self.assertNotIn("Turn 2", prompt)
            self.assertIn("Turn 3", prompt)
            self.assertIn("Turn 6", prompt)

            # Each turn should be truncated to MAX_HISTORY_MSG_CHARS (700)
            # Thus "X" * 1000 should never appear intact
            self.assertNotIn("X" * 750, prompt)

            # Injected document text should be truncated to MAX_DOC_CONTEXT_CHARS (6000)
            self.assertNotIn("D" * 6500, prompt)
            self.assertIn("D" * 6000, prompt)

    def test_rag_context_capped_at_budget(self):
        """Concatenated ChromaDB context is capped at MAX_RAG_CONTEXT_CHARS=3500."""
        captured_prompts = []
        with patch("backend.invoke_gemini", side_effect=lambda p: (captured_prompts.append(p), "AI reply")[1]):
            mock_db = MagicMock()
            # Return 3 massive docs totaling 9000 chars
            mock_db.similarity_search.return_value = [
                MagicMock(page_content="C" * 3000),
                MagicMock(page_content="C" * 3000),
                MagicMock(page_content="C" * 3000),
            ]
            app.state.db = mock_db

            resp = self.client.post("/chat", json={"message": "Legal question on statute"})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(len(captured_prompts), 1)
            prompt = captured_prompts[0]

            # Context between <legal_context> tags must be exactly capped to MAX_RAG_CONTEXT_CHARS (3500)
            context_inside = prompt.split("<legal_context>\n")[1].split("\n</legal_context>")[0]
            self.assertEqual(len(context_inside), backend.MAX_RAG_CONTEXT_CHARS)
            self.assertNotIn("C" * 3600, prompt)
            self.assertIn("C" * 3000, prompt)

if __name__ == "__main__":
    unittest.main()

