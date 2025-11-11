use serde::{Deserialize, Serialize};

use crate::thread::{
    HistoryPacker, HistoryPackingStrategy, IdempotencyRecord, Message, MessageAudit, MessageId,
    MessageRole, PackedHistory, TenantId, Thread, ThreadConfigSnapshot, ThreadConfigVersion,
    ThreadId, ThreadStore, ThreadStoreError,
};

use std::time::SystemTime;

/// Public DTOs and handlers for framework-agnostic HTTP-style operations over [`ThreadStore`].
///
/// This module:
/// - Defines serde-friendly request/response/view types.
/// - Exposes pure, synchronous functions that wrap a `ThreadStore`.
/// - Does NOT perform any real HTTP or IO; mapping to concrete frameworks is left to callers.

/// Thread view exposed over HTTP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadView {
    pub tenant_id: TenantId,
    pub thread_id: ThreadId,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub current_config_version: ThreadConfigVersion,
    pub archived: bool,
    pub name: Option<String>,
}

/// Thread configuration snapshot with its version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadConfigVersionedSnapshotView {
    pub version: ThreadConfigVersion,
    pub snapshot: ThreadConfigSnapshot,
}

/// Message view exposed over HTTP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageView {
    pub tenant_id: TenantId,
    pub thread_id: ThreadId,
    pub message_id: MessageId,
    pub role: MessageRole,
    pub created_at: SystemTime,
    pub sequence: u64,
    pub content: String,
    pub config_version_ref: ThreadConfigVersion,
    pub redacted: bool,
    pub truncated: bool,
    pub redaction_reason: Option<String>,
    pub audit: Option<MessageAudit>,
}

/// Create-thread request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateThreadRequest {
    pub name: Option<String>,
    pub initial_config: Option<ThreadConfigSnapshot>,
}

/// Create-thread response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateThreadResponse {
    pub thread: ThreadView,
}

/// Get-thread response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetThreadResponse {
    pub thread: ThreadView,
}

/// List-threads response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListThreadsResponse {
    pub threads: Vec<ThreadView>,
}

/// Append-only configuration update request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateThreadConfigAppendOnlyRequest {
    pub thread_id: ThreadId,
    pub new_snapshot: ThreadConfigSnapshot,
}

/// Append-only configuration update response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateThreadConfigAppendOnlyResponse {
    pub thread: ThreadView,
}

/// Append-message request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendMessageRequest {
    pub thread_id: ThreadId,
    pub role: MessageRole,
    pub content: String,
    pub config_version_ref: ThreadConfigVersion,
    pub idempotency_key: Option<String>,
    pub idempotency_scope_key: Option<String>,
    pub idempotency_endpoint: Option<String>,
}

/// Append-message response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendMessageResponse {
    pub message: MessageView,
}

/// Get-messages response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetMessagesResponse {
    pub messages: Vec<MessageView>,
}

/// Idempotency: record request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordIdempotencyRequest {
    pub endpoint: String,
    pub key: String,
    pub scope_key: Option<String>,
    pub thread_id: Option<ThreadId>,
    pub message_id: Option<MessageId>,
    pub config_version: Option<ThreadConfigVersion>,
    pub response_fingerprint: Option<String>,
    pub expires_at: Option<SystemTime>,
}

/// Idempotency: record response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordIdempotencyResponse {
    pub recorded: bool,
}

/// Idempotency: get request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetIdempotencyRecordRequest {
    pub endpoint: String,
    pub key: String,
    pub scope_key: Option<String>,
}

/// Idempotency record view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdempotencyRecordView {
    pub tenant_id: TenantId,
    pub thread_id: Option<ThreadId>,
    pub message_id: Option<MessageId>,
    pub config_version: Option<ThreadConfigVersion>,
    pub response_fingerprint: Option<String>,
    pub created_at: SystemTime,
    pub expires_at: Option<SystemTime>,
}

/// Idempotency: get response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetIdempotencyRecordResponse {
    pub record: Option<IdempotencyRecordView>,
}

/// Packed history: get request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPackedHistoryRequest {
    pub thread_id: ThreadId,
    pub strategy: Option<HistoryPackingStrategy>,
    pub max_messages: Option<usize>,
}

/// Packed history: get response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPackedHistoryResponse {
    pub packed: PackedHistory,
}

/// HTTP-like status codes mapped to domain errors.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HttpStatusLike {
    Ok,
    Created,
    NoContent,
    BadRequest,
    Unauthorized,
    Forbidden,
    NotFound,
    Conflict,
    Unavailable,
    Internal,
    UnprocessableEntity,
}

/// Canonical error body for HTTP-like responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

/// Map `ThreadStoreError` into HTTP-like status + body.
pub fn map_thread_store_error_to_http(err: ThreadStoreError) -> (HttpStatusLike, ErrorBody) {
    match err {
        ThreadStoreError::NotFound => (
            HttpStatusLike::NotFound,
            ErrorBody {
                code: "not_found".into(),
                message: "Resource not found".into(),
            },
        ),
        ThreadStoreError::Conflict => (
            HttpStatusLike::Conflict,
            ErrorBody {
                code: "conflict".into(),
                message: "Conflict with current resource state".into(),
            },
        ),
        ThreadStoreError::PermissionDenied => (
            HttpStatusLike::Forbidden,
            ErrorBody {
                code: "forbidden".into(),
                message: "Permission denied".into(),
            },
        ),
        ThreadStoreError::Unavailable => (
            HttpStatusLike::Unavailable,
            ErrorBody {
                code: "unavailable".into(),
                message: "Service temporarily unavailable".into(),
            },
        ),
        ThreadStoreError::Internal => (
            HttpStatusLike::Internal,
            ErrorBody {
                code: "internal".into(),
                message: "Internal server error".into(),
            },
        ),
        ThreadStoreError::Other(msg) => (
            HttpStatusLike::BadRequest,
            ErrorBody {
                code: "other".into(),
                message: msg,
            },
        ),
    }
}

//
// Internal mappers
//

fn thread_to_view(thread: &Thread) -> ThreadView {
    ThreadView {
        tenant_id: thread.tenant_id.clone(),
        thread_id: thread.thread_id.clone(),
        created_at: thread.created_at,
        updated_at: thread.updated_at,
        current_config_version: thread.current_config_version,
        archived: thread.archived,
        name: thread.name.clone(),
    }
}

fn message_to_view(message: &Message) -> MessageView {
    MessageView {
        tenant_id: message.tenant_id.clone(),
        thread_id: message.thread_id.clone(),
        message_id: message.message_id.clone(),
        role: message.role.clone(),
        created_at: message.created_at,
        sequence: message.sequence,
        content: message.content.clone(),
        config_version_ref: message.config_version_ref,
        redacted: message.redacted,
        truncated: message.truncated,
        redaction_reason: message.redaction_reason.clone(),
        audit: message.audit.clone(),
    }
}

fn idempotency_record_to_view(record: &IdempotencyRecord) -> IdempotencyRecordView {
    IdempotencyRecordView {
        tenant_id: record.tenant_id.clone(),
        thread_id: record.thread_id.clone(),
        message_id: record.message_id.clone(),
        config_version: record.config_version,
        response_fingerprint: record.response_fingerprint.clone(),
        created_at: record.created_at,
        expires_at: record.expires_at,
    }
}

//
// Handlers
//

/// Create a new thread for the given tenant.
///
/// - Uses `req.initial_config` if provided, otherwise a default/empty snapshot.
/// - Delegates to `ThreadStore::create_thread`.
pub fn create_thread<T: ThreadStore>(
    store: &T,
    tenant_id: TenantId,
    req: CreateThreadRequest,
) -> Result<CreateThreadResponse, ThreadStoreError> {
    let initial_config = req
        .initial_config
        .unwrap_or(ThreadConfigSnapshot {
            model_id: None,
            model: None,
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            history_packing_strategy: None,
            label: None,
        });

    let mut thread = store.create_thread(&tenant_id, initial_config)?;
    // Apply optional name from request if present and not already set.
    if thread.name.is_none() {
        thread.name = req.name;
    }

    Ok(CreateThreadResponse {
        thread: thread_to_view(&thread),
    })
}

/// Get a thread by id for a tenant.
pub fn get_thread<T: ThreadStore>(
    store: &T,
    tenant_id: TenantId,
    thread_id: ThreadId,
) -> Result<Option<GetThreadResponse>, ThreadStoreError> {
    match store.get_thread(&tenant_id, &thread_id)? {
        Some(thread) => Ok(Some(GetThreadResponse {
            thread: thread_to_view(&thread),
        })),
        None => Ok(None),
    }
}

/// List threads for a tenant.
pub fn list_threads<T: ThreadStore>(
    store: &T,
    tenant_id: TenantId,
) -> Result<ListThreadsResponse, ThreadStoreError> {
    let threads = store
        .list_threads(&tenant_id)?
        .into_iter()
        .map(|t| thread_to_view(&t))
        .collect();

    Ok(ListThreadsResponse { threads })
}

/// Append a new configuration snapshot for a thread (append-only).
pub fn update_thread_config_append_only<T: ThreadStore>(
    store: &T,
    tenant_id: TenantId,
    req: UpdateThreadConfigAppendOnlyRequest,
) -> Result<UpdateThreadConfigAppendOnlyResponse, ThreadStoreError> {
    let updated_thread =
        store.update_thread_config_append_only(&tenant_id, &req.thread_id, req.new_snapshot)?;

    Ok(UpdateThreadConfigAppendOnlyResponse {
        thread: thread_to_view(&updated_thread),
    })
}

/// Append a message to a thread.
///
/// Notes:
/// - `ThreadStore::append_message` owns sequencing and invariants.
/// - This handler constructs a `Message` with placeholder identity/ordering;
///   the store implementation is responsible for normalizing as needed.
/// - After append, messages are fetched and the last one is returned as view.
pub fn append_message<T: ThreadStore>(
    store: &T,
    tenant_id: TenantId,
    req: AppendMessageRequest,
) -> Result<AppendMessageResponse, ThreadStoreError> {
    // Build a provisional message. Concrete stores are responsible for:
    // - ensuring message_id uniqueness
    // - assigning correct sequence
    // - validating config_version_ref
    let provisional = Message {
        tenant_id: tenant_id.clone(),
        thread_id: req.thread_id.clone(),
        // placeholder; store may ignore/override
        message_id: MessageId::new(""),
        role: req.role,
        created_at: SystemTime::now(),
        sequence: 0,
        content: req.content,
        config_version_ref: req.config_version_ref,
        redacted: false,
        truncated: false,
        redaction_reason: None,
        audit: None,
    };

    store.append_message(provisional)?;

    // Fetch messages and treat the last as the appended one.
    let messages = store.get_messages(&tenant_id, &req.thread_id)?;
    let appended = messages
        .last()
        .ok_or(ThreadStoreError::Internal)?; // should not happen if append succeeded

    Ok(AppendMessageResponse {
        message: message_to_view(appended),
    })
}

/// Get all messages for a thread.
pub fn get_messages<T: ThreadStore>(
    store: &T,
    tenant_id: TenantId,
    thread_id: ThreadId,
) -> Result<GetMessagesResponse, ThreadStoreError> {
    let messages = store
        .get_messages(&tenant_id, &thread_id)?
        .into_iter()
        .map(|m| message_to_view(&m))
        .collect();

    Ok(GetMessagesResponse { messages })
}

/// Record an idempotency key.
///
/// - On success returns `recorded: true`.
/// - On conflict or other error, propagates the `ThreadStoreError`.
pub fn record_idempotency<T: ThreadStore>(
    store: &T,
    tenant_id: TenantId,
    req: RecordIdempotencyRequest,
) -> Result<RecordIdempotencyResponse, ThreadStoreError> {
    let record = IdempotencyRecord {
        tenant_id: tenant_id.clone(),
        thread_id: req.thread_id,
        message_id: req.message_id,
        config_version: req.config_version,
        response_fingerprint: req.response_fingerprint,
        created_at: SystemTime::now(),
        expires_at: req.expires_at,
    };

    store.record_idempotency_key(&tenant_id, &req.endpoint, &req.key, record)?;

    Ok(RecordIdempotencyResponse { recorded: true })
}

/// Get an idempotency record.
///
/// - Returns `record: None` if not found.
pub fn get_idempotency_record<T: ThreadStore>(
    store: &T,
    tenant_id: TenantId,
    req: GetIdempotencyRecordRequest,
) -> Result<GetIdempotencyRecordResponse, ThreadStoreError> {
    let record_opt =
        store.get_idempotency_record(&tenant_id, &req.endpoint, &req.key)?;

    let view_opt = record_opt.as_ref().map(idempotency_record_to_view);

    Ok(GetIdempotencyRecordResponse { record: view_opt })
}

/// Get packed history for a thread using a provided `HistoryPacker`.
///
/// - Fails with `ThreadStoreError::NotFound` if the thread does not exist.
/// - Ignores `strategy`/`max_messages` here; the `HistoryPacker` implementation may
///   incorporate them internally based on its configuration.
pub fn get_packed_history<TStore, TPacker>(
    store: &TStore,
    packer: &TPacker,
    tenant_id: TenantId,
    req: GetPackedHistoryRequest,
) -> Result<GetPackedHistoryResponse, ThreadStoreError>
where
    TStore: ThreadStore,
    TPacker: HistoryPacker,
{
    let thread = match store.get_thread(&tenant_id, &req.thread_id)? {
        Some(t) => t,
        None => return Err(ThreadStoreError::NotFound),
    };

    let messages = store.get_messages(&tenant_id, &req.thread_id)?;

    let packed = packer.pack(&thread, &messages);

    Ok(GetPackedHistoryResponse { packed })
}