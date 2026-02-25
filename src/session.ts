import { v4 as uuidv4 } from "uuid";
import { getConfig } from "./config.js";
import {
    insertSession,
    endSessionRecord,
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

    // Archive previous session if one exists
    if (_currentSessionId) {
        result.previousSessionId = _currentSessionId;

        // Call the archive callback (compresses working memory → Tier 1)
        if (archiveCallback) {
            try {
                result.archiveSummary = await archiveCallback(_currentSessionId);
                result.previousSessionArchived = result.archiveSummary !== null;
            } catch {
                // Archive failure is non-fatal — we still start the new session
            }
        }

        // Mark old session as ended in the database
        endSessionRecord(_currentSessionId);
    }

    // Generate new session
    _currentSessionId = uuidv4();
    _sessionStartedAt = new Date().toISOString();

    result.sessionId = _currentSessionId;
    result.startedAt = _sessionStartedAt;

    // Persist session to database
    insertSession({
        id: _currentSessionId,
        started_at: _sessionStartedAt,
        ended_at: null,
        metadata: JSON.stringify({
            previousSessionId: result.previousSessionId,
        }),
    });

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
