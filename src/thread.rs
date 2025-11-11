//! Thread and chat domain model for multi-tenant, audit-safe, provider-agnostic conversations.
//!
//! This module defines the core data structures and contracts for:
//! - Multi-tenant isolation (TenantId-scoped threads, messages, and idempotency records)
//! - Versioned thread configuration snapshots (append-only, per-message association)
//! - Audit-grade metadata (provider-agnostic usage summaries and logical model identifiers)
//! - History packing strategies (pure contracts, deterministic, tenant-safe)
//! - ThreadStore + idempotency interfaces (no concrete storage, no HTTP, no secrets)
//!
//! Multi-tenant Guarantees
//! =======================
//! - All domain entities are explicitly scoped by `TenantId`.
//! - `(tenant_id, thread_id)` together form the composite identity for a thread.
//! - Every persisted `Message` is scoped by:
//!     - `tenant_id: TenantId`
//!     - `thread_id: ThreadId`
//! - The `ThreadStore` trait:
//!     - Accepts `&TenantId` on all read/write operations that load or mutate tenant data.
//!     - MUST enforce that lookups and mutations never cross tenants.
//!     - MUST ensure that `Message` and `Thread` records cannot be observed or modified
//!       with a mismatched `TenantId`.
//!
//! Logging & Secrets
//! =================
//! - Implementations MAY log tenant/thread identifiers and logical model IDs for observability.
//! - Implementations MUST NOT log secrets, raw API keys, or full provider responses.
//! - Provider-specific HTTP details and secret handling live exclusively in
//!   [`src/utils.rs`](src/utils.rs:440).
//!
//! Config Versioning
//! =================
//! - Threads use append-only configuration versioning:
//!     - Each configuration is represented by a `ThreadConfigVersion` + `ThreadConfigSnapshot`.
//!     - `Thread.config_history` is append-only; later entries represent newer versions.
//!     - `Thread.current_config_version` MUST match the latest entry in `config_history`.
//! - Each `Message` includes `config_version_ref: ThreadConfigVersion`:
//!     - This is immutable once set and captures which config governed that message.
//! - `update_thread_config_append_only` (in `ThreadStore`) MUST:
//!     - Create a new version only (no in-place mutation of old snapshots).
//!     - Preserve previous versions for audit and replay.
//!
//! Audit Metadata
//! ==============
//! - `MessageAudit` captures provider-agnostic metadata:
//!     - Logical model identifier
//!     - Packing strategy used
//!     - Truncation/partial flags
//!     - High-level usage/cost estimates (`UsageSummary`)
//! - Audit metadata:
//!     - MUST NOT contain secrets or raw provider responses.
//!     - Is designed to align with usage tracking (e.g. OpenRouterUsage) while remaining
//!       independent of any concrete provider.
//!
//! History Packing Contracts
//! =========================
//! - `HistoryPackingStrategy` enumerates high-level packing behaviors.
//! - `HistoryPacker` defines a deterministic strategy interface:
//!     - `pack(&self, thread, messages) -> PackedHistory`
//! - Contracts:
//!     - Packing MUST be deterministic for a given (thread, messages, strategy).
//!     - MUST respect tenant and thread scoping (single-tenant, single-thread inputs).
//!     - MUST respect redaction/truncation flags and avoid leaking redacted content.
//!     - Implementations will be added in later steps; this module defines only contracts.
//!
//! ThreadStore & Idempotency
//! =========================
//! - `ThreadStore` is a pluggable abstraction (e.g., Postgres, DynamoDB, in-memory):
//!     - No HTTP, no provider logic, no secrets.
//!     - All methods return `Result<_, ThreadStoreError>`.
//!     - Idempotency is modeled explicitly via `IdempotencyRecord`.
//! - `ThreadStoreError`:
//!     - Avoids exposing secrets or provider-specific internals.
//!     - Provides portable error categories suitable for HTTP and flow mapping upstream.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::SystemTime;

/// Strongly typed tenant identifier (multi-tenant isolation boundary).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub String);

impl TenantId {
    /// Construct from any string-like source.
    pub fn new<S: Into<String>>(s: S) -> Self {
        Self(s.into())
    }
}

impl From<&str> for TenantId {
    fn from(s: &str) -> Self {
        TenantId(s.to_owned())
    }
}

impl From<String> for TenantId {
    fn from(s: String) -> Self {
        TenantId(s)
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Strongly typed thread identifier (scoped within a TenantId).
///
/// Note: Together with `TenantId`, forms the composite identity for a thread.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ThreadId(pub String);

impl ThreadId {
    pub fn new<S: Into<String>>(s: S) -> Self {
        Self(s.into())
    }
}

impl From<&str> for ThreadId {
    fn from(s: &str) -> Self {
        ThreadId(s.to_owned())
    }
}

impl From<String> for ThreadId {
    fn from(s: String) -> Self {
        ThreadId(s)
    }
}

impl fmt::Display for ThreadId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Strongly typed message identifier (scoped within a TenantId + ThreadId).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(pub String);

impl MessageId {
    pub fn new<S: Into<String>>(s: S) -> Self {
        Self(s.into())
    }
}

impl From<&str> for MessageId {
    fn from(s: &str) -> Self {
        MessageId(s.to_owned())
    }
}

impl From<String> for MessageId {
    fn from(s: String) -> Self {
        MessageId(s)
    }
}

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Logical model identifier used in audit metadata.
///
/// Provider-agnostic wrapper around model IDs such as
/// "openai/gpt-4o", "anthropic/claude-3-sonnet", etc.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LogicalModelId(pub String);

impl LogicalModelId {
    /// Construct from any string-like source.
    pub fn new<S: Into<String>>(s: S) -> Self {
        Self(s.into())
    }
}

impl From<&str> for LogicalModelId {
    fn from(s: &str) -> Self {
        LogicalModelId(s.to_owned())
    }
}

impl From<String> for LogicalModelId {
    fn from(s: String) -> Self {
        LogicalModelId(s)
    }
}

impl fmt::Display for LogicalModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Monotonic, append-only configuration version for a thread.
///
/// Invariants:
/// - Versions start from an implementation-defined positive value (commonly 1).
/// - New configurations MUST always use a strictly greater version.
/// - Old versions MUST NOT be mutated or re-used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ThreadConfigVersion(pub u64);

impl ThreadConfigVersion {
    /// Construct from a raw version number.
    pub fn new(v: u64) -> Self {
        ThreadConfigVersion(v)
    }
}

impl fmt::Display for ThreadConfigVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// High-level strategy labels for conversation history packing.
///
/// These are intentionally abstract so they can be mapped to specific algorithms
/// without changing the domain contracts.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HistoryPackingStrategy {
    /// Take messages from the tail (most recent first) up to context limits.
    Tail,
    /// Combine head (early system/user/setup) and tail (recent turns).
    HeadTail,
    /// Implementation-defined "smart" packing based on heuristics.
    Smart,
}

/// Compact view of a message used when building provider prompts.
///
/// This intentionally includes only:
/// - `role`
/// - `content`
///
/// It excludes tenant IDs, thread IDs, and other metadata to prevent leakage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageForPrompt {
    pub role: MessageRole,
    pub content: String,
}

/// Result of applying a history packing strategy for a given thread.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedHistory {
    /// Provider-agnostic, prompt-ready messages.
    pub messages: Vec<MessageForPrompt>,
    /// Whether any truncation occurred while constructing this view.
    pub truncated: bool,
    /// Strategy that produced this packed history.
    pub strategy: HistoryPackingStrategy,
}

/// Trait defining deterministic history packing strategies.
///
/// Contracts:
/// - MUST be deterministic for the same `(thread, messages)` input and internal
///   configuration (no hidden RNG without fixed seed).
/// - MUST treat all `messages` as belonging to the same `(tenant_id, thread_id)`.
///   Callers and implementations MUST NOT mix messages from multiple threads.
/// - MUST respect redaction / truncation flags on `Message`.
/// - MUST NOT leak redacted content or secrets into `MessageForPrompt`.
pub trait HistoryPacker {
    fn pack(&self, thread: &Thread, messages: &[Message]) -> PackedHistory;
}

/// Minimal reference implementation to exercise the packing contracts.
///
/// This is intentionally simple and used only to validate that the core
/// types form a coherent API surface. It packs the most recent messages
/// up to a configurable soft limit using the `Tail` strategy.
pub struct DefaultHistoryPacker {
    max_messages: usize,
}

impl DefaultHistoryPacker {
    pub fn new(max_messages: usize) -> Self {
        Self { max_messages }
    }
}

impl HistoryPacker for DefaultHistoryPacker {
    fn pack(&self, thread: &Thread, messages: &[Message]) -> PackedHistory {
        let strategy = HistoryPackingStrategy::Tail;

        let truncated = messages.len() > self.max_messages;
        let start = messages.len().saturating_sub(self.max_messages);
        let slice = &messages[start..];

        let packed_messages = slice
            .iter()
            .map(|m| MessageForPrompt {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();

        // Touch thread identity to keep the contract explicit and used.
        let _ = (&thread.tenant_id, &thread.thread_id);

        PackedHistory {
            messages: packed_messages,
            truncated,
            strategy,
        }
    }
}

/// Snapshot of thread configuration at a specific version.
///
/// Provider-agnostic fields only:
/// - No raw secrets or API keys.
/// - No direct HTTP details.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadConfigSnapshot {
    /// Logical model identifier (e.g., "openai/gpt-4o").
    pub model_id: Option<LogicalModelId>,
    /// Raw model identifier string for flexibility / migration.
    pub model: Option<String>,
    /// Temperature for sampling, if applicable.
    pub temperature: Option<f32>,
    /// Max tokens for completion, if applicable.
    pub max_tokens: Option<u32>,
    /// System prompt or instruction text applied at this version.
    pub system_prompt: Option<String>,
    /// History packing strategy name associated with this config.
    /// Stored as enum to keep it type-safe and auditable.
    pub history_packing_strategy: Option<HistoryPackingStrategy>,
    /// Arbitrary, provider-agnostic label or metadata (e.g., "default", "safe", "experimenta").
    pub label: Option<String>,
}

/// Thread-level metadata.
///
/// Invariants:
/// - `tenant_id` + `thread_id` form the composite key.
/// - `config_history` is append-only; latest entry defines `current_config_version`.
/// - `current_config_version` MUST reference an entry present in `config_history`.
/// - `created_at` is set on creation and MUST NOT change.
/// - `updated_at` MUST be updated on mutations like config append or archival.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Thread {
    pub tenant_id: TenantId,
    pub thread_id: ThreadId,
    /// Creation time; implementations may choose to persist an opaque `SystemTime`.
    pub created_at: SystemTime,
    /// Last update time (e.g., message appended, config changed).
    pub updated_at: SystemTime,
    /// The currently active configuration version.
    pub current_config_version: ThreadConfigVersion,
    /// Append-only history of configurations.
    ///
    /// The last element's version MUST equal `current_config_version`.
    pub config_history: Vec<(ThreadConfigVersion, ThreadConfigSnapshot)>,
    /// Whether this thread is archived (no new messages by default).
    pub archived: bool,
    /// Optional human-readable name for UI / debugging.
    pub name: Option<String>,
}

/// Role of a message in a conversation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Provider-agnostic usage summary for a single message or turn.
///
/// Designed for alignment with higher-level usage tracking without embedding
/// provider-specific response payloads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UsageSummary {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub estimated_cost_usd: Option<f64>,
}

/// Audit metadata attached to a message.
///
/// MUST NOT contain:
/// - Raw provider responses.
/// - Unredacted secrets or API keys.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessageAudit {
    /// Logical model identifier actually used (if known).
    pub model_id: Option<LogicalModelId>,
    /// Packing strategy actually applied when constructing the prompt.
    pub packing_strategy: Option<HistoryPackingStrategy>,
    /// Whether the message content was truncated relative to the original intent.
    pub truncated: bool,
    /// Whether this message represents a partial / streaming response.
    pub partial: bool,
    /// High-level, non-sensitive error description if something went wrong.
    pub error_summary: Option<String>,
    /// Usage summary (tokens, cost) if available.
    pub usage: Option<UsageSummary>,
}

/// Single message within a thread.
///
/// Invariants:
/// - `tenant_id` + `thread_id` MUST match the owning Thread.
/// - `sequence` is monotonically increasing per thread (no gaps requirement enforced here).
/// - `config_version_ref` is immutable once assigned.
/// - Redaction/truncation flags control what may be surfaced externally.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub tenant_id: TenantId,
    pub thread_id: ThreadId,
    pub message_id: MessageId,
    pub role: MessageRole,
    pub created_at: SystemTime,
    /// Monotonic sequence number per thread to enforce ordering.
    pub sequence: u64,
    /// User-visible content after any redaction / truncation.
    pub content: String,
    /// Configuration version snapshot used when this message was created.
    pub config_version_ref: ThreadConfigVersion,
    /// Whether this message's content has been redacted (e.g., PII removal).
    pub redacted: bool,
    /// Whether this message's content has been truncated for length/context reasons.
    pub truncated: bool,
    /// Optional human-readable reason or code for redaction/truncation.
    pub redaction_reason: Option<String>,
    /// Optional audit metadata (model id, usage, packing, errors).
    pub audit: Option<MessageAudit>,
}

/// Idempotency record for write operations.
///
/// This captures enough information to:
/// - Detect duplicate submissions for the same (tenant, endpoint, key).
/// - Return a stable, consistent response for retried idempotent requests.
/// - WITHOUT storing secrets or full provider responses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdempotencyRecord {
    /// Tenant that owns the idempotent operation.
    pub tenant_id: TenantId,
    /// Optional thread this operation targets.
    pub thread_id: Option<ThreadId>,
    /// Optional message this operation produced (e.g., assistant reply).
    pub message_id: Option<MessageId>,
    /// Snapshot of the configuration version in effect.
    pub config_version: Option<ThreadConfigVersion>,
    /// Stable summary or hash of the logical response to return on retries.
    /// Implementations SHOULD choose a deterministic, non-sensitive representation.
    pub response_fingerprint: Option<String>,
    /// Timestamp when this record was created.
    pub created_at: SystemTime,
    /// Optional expiry hint (implementations may enforce cleanup based on this).
    pub expires_at: Option<SystemTime>,
}

/// Error type for `ThreadStore` operations.
///
/// Contracts:
/// - MUST NOT include secrets or full upstream error messages.
/// - SHOULD be mappable to HTTP or flow-level errors without leaking internals.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ThreadStoreError {
    /// Resource not found (thread, message, idempotency record, etc.).
    NotFound,
    /// Conflict in state (e.g., duplicate IDs, sequence violation, stale config).
    Conflict,
    /// Tenant or caller is not allowed to access the resource.
    PermissionDenied,
    /// Store is temporarily unavailable (e.g., network/infra issues).
    Unavailable,
    /// Generic internal error; implementations SHOULD log details internally only.
    Internal,
    /// Catch-all with a non-sensitive message.
    Other(String),
}

/// Pluggable storage abstraction for threads, messages, and idempotency.
///
/// Notes:
/// - Implementations MAY be in-memory, SQL, NoSQL, or any other backend.
/// - This trait is intentionally synchronous for now; later steps MAY introduce
///   async variants or use `async_trait`.
/// - All methods:
///     - MUST enforce tenant scoping.
///     - MUST adhere to config versioning and append-only invariants.
///     - MUST respect idempotency contracts.
pub trait ThreadStore: Send + Sync {
    /// Create a new thread for a tenant with an initial configuration snapshot.
    ///
    /// Implementations:
    /// - MUST assign an initial `ThreadConfigVersion` (e.g., 1).
    /// - MUST store `(version, snapshot)` in `config_history`.
    /// - MUST set `current_config_version` to that version.
    fn create_thread(
        &self,
        tenant: &TenantId,
        initial_config: ThreadConfigSnapshot,
    ) -> Result<Thread, ThreadStoreError>;

    /// Fetch a thread by `(tenant, thread_id)`.
    fn get_thread(
        &self,
        tenant: &TenantId,
        thread: &ThreadId,
    ) -> Result<Option<Thread>, ThreadStoreError>;

    /// List all threads for a given tenant.
    fn list_threads(&self, tenant: &TenantId) -> Result<Vec<Thread>, ThreadStoreError>;

    /// Append a message to its owning thread.
    ///
    /// Implementations:
    /// - MUST verify `message.tenant_id` and `message.thread_id` are consistent
    ///   with the targeted thread and provided tenant context.
    /// - MUST ensure `sequence` ordering per thread.
    fn append_message(&self, message: Message) -> Result<(), ThreadStoreError>;

    /// Get all messages for a given `(tenant, thread_id)`.
    fn get_messages(
        &self,
        tenant: &TenantId,
        thread: &ThreadId,
    ) -> Result<Vec<Message>, ThreadStoreError>;

    /// Append a new configuration snapshot as a new version for a thread.
    ///
    /// Contracts:
    /// - MUST be append-only: previous snapshots remain intact.
    /// - MUST assign a strictly increasing `ThreadConfigVersion`.
    /// - MUST update `current_config_version` to the new version.
    fn update_thread_config_append_only(
        &self,
        tenant: &TenantId,
        thread: &ThreadId,
        new_config: ThreadConfigSnapshot,
    ) -> Result<Thread, ThreadStoreError>;

    /// Record an idempotency key for a given `(tenant, endpoint, key)`.
    ///
    /// Contracts:
    /// - MUST be safe for concurrent calls (no duplicate divergent records).
    /// - MUST associate the record with the tenant to prevent cross-tenant leaks.
    fn record_idempotency_key(
        &self,
        tenant: &TenantId,
        endpoint: &str,
        key: &str,
        value: IdempotencyRecord,
    ) -> Result<(), ThreadStoreError>;

    /// Retrieve a previously stored idempotency record.
    fn get_idempotency_record(
        &self,
        tenant: &TenantId,
        endpoint: &str,
        key: &str,
    ) -> Result<Option<IdempotencyRecord>, ThreadStoreError>;
}
