import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath('backend'))
import backend
from backend import app, get_current_user, get_history_for_prompt, _get_session_lock, _session_locks, MAX_CACHED_SESSION_LOCKS

class TestDataLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_user = MagicMock()
        cls.mock_user.id = "00000000-0000-0000-0000-000000000001"
        cls.mock_user.email = "test@example.com"
        app.dependency_overrides[get_current_user] = lambda: cls.mock_user
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.clear()

    def test_delete_session_404_when_missing(self):
        """delete_session returns 404 when no session matched."""
        with patch("backend.supabase.table") as mock_table:
            mock_table.return_value.delete.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
            resp = self.client.delete("/sessions/00000000-0000-0000-0000-000000000099")
            self.assertEqual(resp.status_code, 404)
            self.assertEqual(resp.json()["detail"], "Session not found.")

    def test_delete_session_200_when_present(self):
        """delete_session returns 200 when session is deleted."""
        with patch("backend.supabase.table") as mock_table:
            mock_table.return_value.delete.return_value.eq.return_value.eq.return_value.execute.return_value.data = [{"id": "s1"}]
            resp = self.client.delete("/sessions/s1")
            self.assertEqual(resp.status_code, 200)

    def test_get_session_messages_orders_by_seq(self):
        """get_session_messages orders by seq ascending."""
        with patch("backend.supabase.table") as mock_table:
            mock_table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [{"id": "s1"}]
            mock_table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [
                {"id": "m1", "role": "User", "content": "Hi", "created_at": "2026-11-03T00:00:00Z", "seq": 1},
                {"id": "m2", "role": "JurixAI", "content": "Hello", "created_at": "2026-11-03T00:00:01Z", "seq": 2}
            ]
            resp = self.client.get("/sessions/s1/messages")
            self.assertEqual(resp.status_code, 200)
            msgs = resp.json()["messages"]
            self.assertEqual(msgs[0]["seq"], 1)
            self.assertEqual(msgs[1]["seq"], 2)

    def test_history_summary_reuse(self):
        """History reuses stored summary when under threshold (< 10 new old messages)."""
        session_id = "test-session-reuse"
        with patch("backend.supabase.table") as mock_table, \
             patch("backend.invoke_gemini") as mock_invoke:

            mock_session = MagicMock()
            mock_session.data = [{"summary": "Previous legal summary.", "summary_upto": 10}]
            recent = [{"role": "User", "content": f"msg {i}", "seq": i} for i in range(24, 14, -1)]
            unsummarized = [{"role": "User", "content": f"old msg {i}", "seq": i} for i in range(11, 15)]

            def table_dispatch(table_name):
                t_mock = MagicMock()
                if table_name == "sessions":
                    t_mock.select.return_value.eq.return_value.execute.return_value = mock_session
                elif table_name == "messages":
                    m_recent = MagicMock()
                    m_recent.data = recent
                    t_mock.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = m_recent

                    m_unsum = MagicMock()
                    m_unsum.data = unsummarized
                    t_mock.select.return_value.eq.return_value.gt.return_value.lt.return_value.order.return_value.execute.return_value = m_unsum
                return t_mock

            mock_table.side_effect = table_dispatch
            result = get_history_for_prompt(session_id)

            mock_invoke.assert_not_called()
            self.assertIn("[EARLIER SUMMARY]:\nPrevious legal summary.", result)

    def test_bounded_session_lock_cache(self):
        """Session lock cache must remain bounded and evict unlocked items when reaching limit."""
        _session_locks.clear()
        # Fill up to max capacity
        for i in range(MAX_CACHED_SESSION_LOCKS + 10):
            _get_session_lock(f"session-{i}")
        # Verify length does not exceed limit
        self.assertLessEqual(len(_session_locks), MAX_CACHED_SESSION_LOCKS + 1)

    def test_document_persistence_in_chat(self):
        """Document text is saved on first message and loaded automatically on follow-ups."""
        captured_prompts = []

        def fake_invoke(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return "AI response"

        session_store = {
            "00000000-0000-0000-0000-000000000020": {
                "id": "00000000-0000-0000-0000-000000000020",
                "title": "Document Session",
                "document_text": None,
                "document_name": None
            }
        }

        def table_dispatch(table_name):
            t_mock = MagicMock()
            if table_name == "sessions":
                def mock_select(fields):
                    s_mock = MagicMock()
                    def mock_eq1(f1, v1):
                        eq1_mock = MagicMock()
                        def mock_eq2(f2, v2):
                            eq2_mock = MagicMock()
                            session = session_store.get(v1)
                            eq2_mock.execute.return_value.data = [session] if session else []
                            return eq2_mock
                        eq1_mock.eq.side_effect = mock_eq2
                        eq1_mock.execute.return_value.data = [session_store.get(v1)] if session_store.get(v1) else []
                        return eq1_mock
                    s_mock.eq.side_effect = mock_eq1
                    return s_mock
                t_mock.select.side_effect = mock_select

                def mock_update(payload):
                    u_mock = MagicMock()
                    def mock_update_eq(f, v):
                        if v in session_store:
                            session_store[v].update(payload)
                        return MagicMock()
                    u_mock.eq.side_effect = mock_update_eq
                    return u_mock
                t_mock.update.side_effect = mock_update

            elif table_name == "messages":
                t_mock.insert.return_value.execute.return_value.data = [{"id": 1}]
            return t_mock

        with patch("backend.supabase.table", side_effect=table_dispatch), \
             patch("backend.invoke_gemini", side_effect=fake_invoke), \
             patch("backend.get_history_for_prompt", return_value="History"):

            app.state.db = MagicMock()
            app.state.db.similarity_search.return_value = []

            # Turn 1: with document_text
            self.client.post("/chat", json={
                "session_id": "00000000-0000-0000-0000-000000000020",
                "message": "Analyze this contract",
                "document_text": "CONTRACT TEXT: Section 5 Arbitration clause"
            })
            self.assertIn("Arbitration clause", captured_prompts[0])

            # Turn 2: without document_text (follow-up)
            self.client.post("/chat", json={
                "session_id": "00000000-0000-0000-0000-000000000020",
                "message": "Where will the arbitration take place?"
            })
            self.assertIn("Arbitration clause", captured_prompts[1])

if __name__ == "__main__":
    unittest.main()

