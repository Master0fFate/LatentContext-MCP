import { v4 as uuidv4 } from "uuid";
import { getConfig } from "./config.js";
import {
    insertSession,
    endSessionRecord,
    transitionSessionRecord,
    getActiveSession,
    getRecentSessions as dbGetRecentSessions,
    type SessionRow,
} from "./database.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SessionInfo {
    sessionId: string;
    startedAt: string;
    isActive: boolean;
}

export interface SessionStartResult {
    sessionId: string;
    startedAt: string;
    previousSessionArchived: boolean;
    previousSessionId: string | null;
    archiveSummary: string | null;
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

let _currentSessionId: string | null = null;
let _sessionStartedAt: string | null = null;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Start a new session. If a session is already active, archives it first.
 * Called automatically on server startup and can be called manually via
 * the session_start tool.
 *
 * @param archiveCallback — optional async callback that compresses/archives
 *        the current working memory before switching sessions.
 *        Receives the old session ID so it can tag the archive properly.
 */
export async function startSession(
    archiveCallback?: (oldSessionId: string) => Promise<string | null>
): Promise<SessionStartResult> {
    const result: SessionStartResult = {
        sessionId: "",
        startedAt: "",
        previousSessionArchived: false,
        previousSessionId: null,
        archiveSummary: null,
    };

    // A process can disappear before it marks its session ended. Treat that
    // database record exactly like an in-process predecessor, rather than
    // leaving an orphaned active lifecycle behind on the next boot.
    const previousSessionId = _currentSessionId ?? getActiveSession()?.id ?? null;
    if (previousSessionId) {
        result.previousSessionId = previousSessionId;

        // Archival is part of the acknowledged transition. Do not hide an
        // error and then clear/close the old lifecycle: its durable Tier 0
        // checkpoint remains recoverable and the caller can retry.
        if (archiveCallback) {
            result.archiveSummary = await archiveCallback(previousSessionId);
            result.previousSessionArchived = result.archiveSummary !== null;
        }
    }

    // Generate locally first; publish it to module state only after the
    // database transition has reached its durable checkpoint.
    const timestamp = Date.now();
    const nextSessionId = `${timestamp}-${uuidv4()}`;
    const nextStartedAt = new Date(timestamp).toISOString();
    const nextSession: SessionRow = {
        id: nextSessionId,
        started_at: nextStartedAt,
        ended_at: null,
        metadata: JSON.stringify({ previousSessionId }),
    };

    if (previousSessionId) {
        transitionSessionRecord(previousSessionId, nextSession);
    } else {
        insertSession(nextSession);
    }

    _currentSessionId = nextSessionId;
    _sessionStartedAt = nextStartedAt;
    result.sessionId = nextSessionId;
    result.startedAt = nextStartedAt;

    return result;
}

/**
 * End the current session. Marks it as ended in the database.
 */
export function endCurrentSession(): void {
    if (_currentSessionId) {
        endSessionRecord(_currentSessionId);
        _currentSessionId = null;
        _sessionStartedAt = null;
    }
}

/**
 * Archive the current durable Tier 0 checkpoint before ending it. This is the
 * shutdown path; failures propagate so callers can log and surface them.
 */
export async function endCurrentSessionWithArchive(
    archiveCallback: (sessionId: string) => Promise<string | null>
): Promise<string | null> {
    if (!_currentSessionId) return null;

    const sessionId = _currentSessionId;
    const archiveSummary = await archiveCallback(sessionId);
    endSessionRecord(sessionId);
    _currentSessionId = null;
    _sessionStartedAt = null;
    return archiveSummary;
}

/**
 * Get the current session ID. Throws if no session is active.
 */
export function getCurrentSessionId(): string {
    if (!_currentSessionId) {
        throw new Error("No active session. Call startSession() first.");
    }
    return _currentSessionId;
}

/**
 * Get the current session ID or null if no session is active.
 */
export function getCurrentSessionIdOrNull(): string | null {
    return _currentSessionId;
}

/**
 * Get the current session start time.
 */
export function getSessionStartTime(): string | null {
    return _sessionStartedAt;
}

/**
 * Check if a session is currently active.
 */
export function isSessionActive(): boolean {
    return _currentSessionId !== null;
}

/**
 * Get current session info.
 */
export function getSessionInfo(): SessionInfo | null {
    if (!_currentSessionId || !_sessionStartedAt) return null;
    return {
        sessionId: _currentSessionId,
        startedAt: _sessionStartedAt,
        isActive: true,
    };
}

/**
 * Get recent session history from the database.
 */
export function getRecentSessions(limit: number = 10): SessionRow[] {
    return dbGetRecentSessions(limit);
}
