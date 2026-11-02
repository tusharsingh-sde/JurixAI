-- =============================================================================
-- Migration: 001_jurix.sql
-- Description: Schema hardening, sequential ordering, and document persistence
--
-- ORDER OF OPERATIONS:
-- 1. Apply this migration file in the Supabase SQL Editor BEFORE deploying
--    the Phase 6b backend code.
-- 2. Once applied, deploy the updated backend code.
--
-- BACKWARD COMPATIBILITY / DEPLOYMENT SAFETY:
-- - The backend code actively queries `.order("seq")` and selects/updates
--   `sessions.summary`, `sessions.summary_upto`, `sessions.document_text`,
--   and `sessions.updated_at`.
-- - Applying this migration first is safe and idempotent: existing code ignores
--   new nullable columns and indexes.
-- =============================================================================

BEGIN;

-- 1. Sequential message ordering with historical backfill
-- Add seq column as nullable bigint if not present
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'messages' AND column_name = 'seq'
    ) THEN
        ALTER TABLE messages ADD COLUMN seq bigint;
    END IF;
END $$;

-- Backfill seq for existing rows ordered by created_at, id
WITH ranked AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY created_at ASC, id ASC) AS rn
    FROM messages
)
UPDATE messages m
SET seq = r.rn
FROM ranked r
WHERE m.id = r.id AND m.seq IS NULL;

-- Create sequence if not exists, attach to column, and set default nextval
DO $$
DECLARE
    max_val bigint;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = 'messages_seq_seq' AND relkind = 'S'
    ) THEN
        CREATE SEQUENCE messages_seq_seq;
    END IF;

    ALTER SEQUENCE messages_seq_seq OWNED BY messages.seq;

    SELECT COALESCE(MAX(seq), 0) + 1 INTO max_val FROM messages;
    PERFORM setval('messages_seq_seq', max_val, false);

    ALTER TABLE messages ALTER COLUMN seq SET DEFAULT nextval('messages_seq_seq');
    ALTER TABLE messages ALTER COLUMN seq SET NOT NULL;
END $$;

-- 2. Rolling conversation summary caching on sessions
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary text;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary_upto bigint DEFAULT 0;

-- 3. Document persistence across conversation follow-ups
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS document_text text;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS document_name text;

-- 4. Activity timestamp on sessions
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();

-- 5. Performance indexes
CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC);

-- 6. Defense-in-depth: Enable Row Level Security (RLS)
-- The backend uses the Supabase service role key, which bypasses RLS.
-- Frontend clients talk strictly through the FastAPI backend, never directly to Supabase.
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

COMMIT;

-- =============================================================================
-- ROLLBACK SCRIPT (Execute in Supabase SQL editor if rollback is needed)
-- =============================================================================
/*
BEGIN;
DROP INDEX IF EXISTS idx_messages_session_seq;
DROP INDEX IF EXISTS idx_sessions_user_updated;
ALTER TABLE messages ALTER COLUMN seq DROP DEFAULT;
DROP SEQUENCE IF EXISTS messages_seq_seq CASCADE;
ALTER TABLE messages DROP COLUMN IF EXISTS seq;
ALTER TABLE sessions DROP COLUMN IF EXISTS summary;
ALTER TABLE sessions DROP COLUMN IF EXISTS summary_upto;
ALTER TABLE sessions DROP COLUMN IF EXISTS document_text;
ALTER TABLE sessions DROP COLUMN IF EXISTS document_name;
ALTER TABLE messages DISABLE ROW LEVEL SECURITY;
ALTER TABLE sessions DISABLE ROW LEVEL SECURITY;
COMMIT;
*/
