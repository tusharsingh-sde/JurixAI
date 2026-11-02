-- =============================================================================
-- Migration: 001_jurix.sql
-- Description: Schema hardening for JurixAI production deployment
-- 
-- ORDER OF OPERATIONS:
-- 1. Apply this migration file in the Supabase SQL Editor BEFORE deploying
--    the Phase 6 backend code.
-- 2. Once applied, deploy the updated backend code.
--
-- BACKWARD COMPATIBILITY / DEPLOYMENT SAFETY:
-- - The Phase 6 backend code queries order by `messages.seq` and reads/writes
--   `sessions.summary`, `sessions.summary_upto`, and `sessions.updated_at`.
-- - If the backend code is deployed BEFORE running this migration, Supabase
--   PostgREST queries referencing `seq`, `summary`, or `summary_upto` would fail
--   with a missing column error (HTTP 400).
-- - Applying this migration first is completely safe: existing code does not
--   break when new nullable/defaulted columns and indexes are added, ensuring
--   zero downtime.
--
-- ROW LEVEL SECURITY (RLS) DEFENSE-IN-DEPTH NOTE:
-- - Row Level Security (RLS) should be enabled on both `sessions` and `messages`
--   tables as defense-in-depth.
-- - The backend connects using the Supabase Service Key (`SUPABASE_SECRET_KEY`),
--   which bypasses RLS policies. Enabling RLS protects data if anon/public keys
--   ever attempt direct table access from client devices.
-- - As per project guidelines, RLS is NOT automatically enabled here. If desired,
--   run:
--     ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
--     ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
-- =============================================================================

-- 1. Add sequential message ordering to messages
-- Generated identity column guarantees monotonically increasing sequence numbers
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'messages' AND column_name = 'seq'
    ) THEN
        ALTER TABLE messages ADD COLUMN seq bigint GENERATED ALWAYS AS IDENTITY;
    END IF;
END $$;

-- 2. Add rolling conversation summary caching to sessions
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary text;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary_upto bigint DEFAULT 0;

-- 3. Ensure updated_at column exists on sessions
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();

-- 4. Create performance indexes
-- Composite index for fast session message retrieval ordered by sequence number
CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages(session_id, seq);

-- Index for fast user session listing ordered by most recent activity
CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC);
