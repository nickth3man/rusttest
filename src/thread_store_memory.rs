//! In-memory implementation of the [`ThreadStore`](src/thread.rs:370) contract.
//!
//! This module provides a minimal, single-process, non-persistent store for:
//! - Threads (scoped by `(TenantId, ThreadId)`)
//! - Messages (scoped by `(TenantId, ThreadId)`)
//! - Idempotency records (scoped by `(TenantId, endpoint, key)`)
//!
//! Design notes:
//! - Intended only for tests, local examples, and documentation snippets.
//! - NOT suitable for production use.
//! - No external dependencies; uses only `std` and `crate::thread`.
//! - Synchronous API, matching the `ThreadStore` trait.
//!
//! Contract adherence highlights:
//! - All operations are tenant-scoped; lookups and mutations never cross tenants.
//! - `create_thread`:
//!   - Generates a `ThreadId` if needed using a deterministic, local, non-secret scheme.
//!   - Initializes `config_history` with version 1 and provided `ThreadConfigSnapshot`.
//!   - Sets `current_config_version` to 1.
//! - `append_message`:
//!   - Validates `(tenant_id, thread_id)` exist and match an existing thread.
//!   - Enforces monotonically increasing `sequence` per thread.
//! - `update_thread_config_append_only`:
//!   - Appends a new strictly increasing `ThreadConfigVersion` and updates
//!     `current_config_version`.
//! - Idempotency:
//!   - Stores and retrieves `IdempotencyRecord` by `(tenant, endpoint, key)`
//!     without expiration logic.
//!
//! Error mapping:
//! - Missing resources      -> `ThreadStoreError::NotFound`
//! - Sequence / version etc -> `ThreadStoreError::Conflict`
//! - Cross-tenant misuse    -> `ThreadStoreError::PermissionDenied`
//! - Lock poisoning / bugs  -> `ThreadStoreError::Internal`

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use crate::thread::{
    IdempotencyRecord, Message, TenantId, Thread, ThreadConfigSnapshot, ThreadConfigVersion,
    ThreadId, ThreadStore, ThreadStoreError,
};

/// Internal composite key for threads and their messages.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TenantThreadKey {
    tenant_id: String,
    thread_id: String,
}

impl TenantThreadKey {
    fn new(tenant: &TenantId, thread: &ThreadId) -> Self {
        Self {
            tenant_id: tenant.0.clone(),
            thread_id: thread.0.clone(),
        }
    }

    fn from_thread(thread: &Thread) -> Self {
        Self {
            tenant_id: thread.tenant_id.0.clone(),
            thread_id: thread.thread_id.0.clone(),
        }
    }
}

/// Internal key for idempotency records.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct IdempotencyKey {
    tenant_id: String,
    endpoint: String,
    key: String,
}

impl IdempotencyKey {
    fn new(tenant: &TenantId, endpoint: &str, key: &str) -> Self {
        Self {
            tenant_id: tenant.0.clone(),
            endpoint: endpoint.to_string(),
            key: key.to_string(),
        }
    }
}

/// Shared in-memory state.
#[derive(Default)]
struct Inner {
    /// Threads keyed by `(tenant_id, thread_id)`.
    threads: HashMap<TenantThreadKey, Thread>,
    /// Messages keyed by `(tenant_id, thread_id)`.
    messages: HashMap<TenantThreadKey, Vec<Message>>,
    /// Idempotency records keyed by `(tenant_id, endpoint, key)`.
    idempotency: HashMap<IdempotencyKey, IdempotencyRecord>,
    /// Monotonic counter used to synthesize unique thread IDs when not provided.
    thread_counter: u64,
}

/// In-memory, non-persistent [`ThreadStore`](src/thread.rs:370) implementation.
///
/// Characteristics:
/// - Append-only semantics for thread configs are enforced in-memory.
/// - Message sequence ordering is validated per thread.
/// - Idempotency records are stored as-is with no expiry logic.
/// - Suitable for unit tests and examples; NOT for production.
pub struct InMemoryThreadStore {
    inner: Arc<RwLock<Inner>>,
}

impl InMemoryThreadStore {
    /// Create a new empty in-memory store.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner::default())),
        }
    }

    /// Generate a new `ThreadId` string in a deterministic, local manner.
    ///
    /// Uses a monotonically increasing counter plus a coarse timestamp component
    /// to reduce collision risk while avoiding external dependencies.
    fn next_thread_id(inner: &mut Inner, tenant: &TenantId) -> ThreadId {
        inner.thread_counter = inner.thread_counter.saturating_add(1);
        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        ThreadId(format!("{}-thread-{}-{}", tenant.0, ts, inner.thread_counter))
    }

    /// Helper to get a mutable reference to a thread by tenant+id with scoping.
    fn get_thread_mut_scoped<'a>(
        inner: &'a mut Inner,
        tenant: &TenantId,
        thread_id: &ThreadId,
    ) -> Result<&'a mut Thread, ThreadStoreError> {
        let key = TenantThreadKey::new(tenant, thread_id);
        match inner.threads.get_mut(&key) {
            Some(thread) => Ok(thread),
            None => Err(ThreadStoreError::NotFound),
        }
    }

    /// Helper to validate that a message's `(tenant_id, thread_id)` corresponds
    /// to an existing thread and respects tenant scoping.
    fn ensure_message_thread_exists(
        inner: &Inner,
        message: &Message,
    ) -> Result<(), ThreadStoreError> {
        let key = TenantThreadKey::new(&message.tenant_id, &message.thread_id);
        if !inner.threads.contains_key(&key) {
            return Err(ThreadStoreError::NotFound);
        }
        Ok(())
    }

    /// Helper to compute the next allowed sequence and validate monotonicity.
    fn validate_sequence_and_insert_message(
        inner: &mut Inner,
        message: Message,
    ) -> Result<(), ThreadStoreError> {
        let key = TenantThreadKey::new(&message.tenant_id, &message.thread_id);
        let entry = inner.messages.entry(key).or_default();

        if let Some(last) = entry.last() {
            if message.sequence <= last.sequence {
                return Err(ThreadStoreError::Conflict);
            }
        }

        entry.push(message);
        Ok(())
    }
}

impl ThreadStore for InMemoryThreadStore {
    fn create_thread(
        &self,
        tenant: &TenantId,
        initial_config: ThreadConfigSnapshot,
    ) -> Result<Thread, ThreadStoreError> {
        let mut inner = self.inner.write().map_err(|_| ThreadStoreError::Internal)?;

        // Determine new ThreadId; always generate to keep this implementation simple and safe.
        let thread_id = Self::next_thread_id(&mut inner, tenant);

        let now = SystemTime::now();
        let version = ThreadConfigVersion(1);

        let thread = Thread {
            tenant_id: tenant.clone(),
            thread_id: thread_id.clone(),
            created_at: now,
            updated_at: now,
            current_config_version: version,
            config_history: vec![(version, initial_config)],
            archived: false,
            name: None,
        };

        let key = TenantThreadKey::from_thread(&thread);

        // Enforce uniqueness; if already exists, report conflict.
        if inner.threads.contains_key(&key) {
            return Err(ThreadStoreError::Conflict);
        }

        inner.threads.insert(key.clone(), thread.clone());
        inner.messages.entry(key).or_insert_with(Vec::new);

        Ok(thread)
    }

    fn get_thread(
        &self,
        tenant: &TenantId,
        thread: &ThreadId,
    ) -> Result<Option<Thread>, ThreadStoreError> {
        let inner = self.inner.read().map_err(|_| ThreadStoreError::Internal)?;
        let key = TenantThreadKey::new(tenant, thread);
        Ok(inner.threads.get(&key).cloned())
    }

    fn list_threads(&self, tenant: &TenantId) -> Result<Vec<Thread>, ThreadStoreError> {
        let inner = self.inner.read().map_err(|_| ThreadStoreError::Internal)?;
        let mut out = Vec::new();
        for (key, thread) in inner.threads.iter() {
            if key.tenant_id == tenant.0 {
                out.push(thread.clone());
            }
        }
        Ok(out)
    }

    fn append_message(&self, message: Message) -> Result<(), ThreadStoreError> {
        let mut inner = self.inner.write().map_err(|_| ThreadStoreError::Internal)?;

        // Ensure owning thread exists and tenant scoping is respected.
        Self::ensure_message_thread_exists(&inner, &message)?;

        // Enforce monotonically increasing sequence per thread.
        Self::validate_sequence_and_insert_message(&mut inner, message)?;
        Ok(())
    }

    fn get_messages(
        &self,
        tenant: &TenantId,
        thread: &ThreadId,
    ) -> Result<Vec<Message>, ThreadStoreError> {
        let inner = self.inner.read().map_err(|_| ThreadStoreError::Internal)?;
        let key = TenantThreadKey::new(tenant, thread);

        match inner.messages.get(&key) {
            Some(messages) => Ok(messages.clone()),
            None => {
                // Distinguish between missing thread vs. empty messages.
                let has_thread = inner.threads.contains_key(&key);
                if has_thread {
                    Ok(Vec::new())
                } else {
                    Err(ThreadStoreError::NotFound)
                }
            }
        }
    }

    fn update_thread_config_append_only(
        &self,
        tenant: &TenantId,
        thread: &ThreadId,
        new_config: ThreadConfigSnapshot,
    ) -> Result<Thread, ThreadStoreError> {
        let mut inner = self.inner.write().map_err(|_| ThreadStoreError::Internal)?;

        let t = Self::get_thread_mut_scoped(&mut inner, tenant, thread)?;

        // Determine next strictly increasing version.
        let next_version = match t.config_history.last() {
            Some((v, _)) => ThreadConfigVersion(v.0.saturating_add(1)),
            None => ThreadConfigVersion(1),
        };

        t.config_history.push((next_version, new_config));
        t.current_config_version = next_version;
        t.updated_at = SystemTime::now();

        Ok(t.clone())
    }

    fn record_idempotency_key(
        &self,
        tenant: &TenantId,
        endpoint: &str,
        key: &str,
        value: IdempotencyRecord,
    ) -> Result<(), ThreadStoreError> {
        // Enforce tenant scoping: record must belong to the provided tenant.
        if value.tenant_id != *tenant {
            return Err(ThreadStoreError::PermissionDenied);
        }

        let mut inner = self.inner.write().map_err(|_| ThreadStoreError::Internal)?;
        let id_key = IdempotencyKey::new(tenant, endpoint, key);

        if inner.idempotency.contains_key(&id_key) {
            // Existing record for the same (tenant, endpoint, key) -> conflict.
            return Err(ThreadStoreError::Conflict);
        }

        inner.idempotency.insert(id_key, value);
        Ok(())
    }

    fn get_idempotency_record(
        &self,
        tenant: &TenantId,
        endpoint: &str,
        key: &str,
    ) -> Result<Option<IdempotencyRecord>, ThreadStoreError> {
        let inner = self.inner.read().map_err(|_| ThreadStoreError::Internal)?;
        let id_key = IdempotencyKey::new(tenant, endpoint, key);
        Ok(inner.idempotency.get(&id_key).cloned())
    }
}