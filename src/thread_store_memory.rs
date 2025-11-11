 //! In-memory implementation of the [`ThreadStore`](src/thread.rs:483-564) contract.
 //!
 //! This module provides a single-process, non-persistent store for:
 //! - Threads (scoped by `(TenantId, ThreadId)`)
 //! - Messages (scoped by `(TenantId, ThreadId)`)
 //! - Idempotency records (scoped by `(TenantId, endpoint, key)`)
 //!
 //! Design notes:
 //! - Intended for tests, examples, and local/dev usage; not a durable backend.
 //! - Pure in-memory with `RwLock` + `HashMap`, no external IO or dependencies.
 //! - Deterministic given call order (aside from `SystemTime::now()`).
 //! - Strictly follows contracts in [`ThreadStore`](src/thread.rs:483-564) and is
 //!   aligned with handlers in [`src/http/thread_handlers.rs`](src/http/thread_handlers.rs:1).
 //!
 //! Contract adherence highlights:
 //! - Multi-tenant:
 //!   - All maps keyed by tenant; no cross-tenant reads/writes.
 //! - Threads:
 //!   - `create_thread` assigns initial `ThreadConfigVersion(1)`.
 //!   - Seeds `config_history` and `current_config_version` consistently.
 //!   - Generates deterministic `ThreadId` per-tenant via an internal counter.
 //! - Config:
 //!   - `update_thread_config_append_only`:
 //!     - Strictly append-only.
 //!     - Versions monotonically increasing.
 //!     - Updates `current_config_version` to latest entry.
 //! - Messages:
 //!   - `append_message`:
 //!     - Validates owning thread exists.
 //!     - Overwrites placeholder `message_id`/`sequence` from handler with
 //!       monotonic per-thread values.
 //!     - Validates `config_version_ref` exists in the thread history.
 //!   - `get_messages`:
 //!     - Returns messages sorted by `(sequence, created_at, message_id)` to
 //!       provide deterministic order.
 //! - Idempotency:
 //!   - Keys are `(tenant, endpoint, key)` as defined by `ThreadStore` trait.
 //!   - Reinserting the exact same logical record is allowed (idempotent).
 //!   - Divergent record for same key yields `ThreadStoreError::Conflict`.
 //!
 //! Error mapping:
 //! - Missing resources      -> `ThreadStoreError::NotFound`
 //! - Sequence/version/etc   -> `ThreadStoreError::Conflict`
 //! - Tenant mismatch        -> `ThreadStoreError::PermissionDenied`
 //! - Lock poisoning / bugs  -> `ThreadStoreError::Internal`
 
 use std::collections::HashMap;
 use std::sync::{Arc, RwLock};
 use std::time::SystemTime;
 
 use crate::thread::{
     IdempotencyRecord, Message, MessageId, TenantId, Thread, ThreadConfigSnapshot,
     ThreadConfigVersion, ThreadId, ThreadStore, ThreadStoreError,
 };
 
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
         Self::new(&thread.tenant_id, &thread.thread_id)
     }
 }
 
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
             endpoint: endpoint.to_owned(),
             key: key.to_owned(),
         }
     }
 }
 
 /// Internal representation of a thread plus its messages and per-thread counters.
 struct StoredThread {
     thread: Thread,
     /// Ordered messages for this thread.
     messages: Vec<Message>,
     /// Next sequence to assign (monotonic, >= 1).
     next_sequence: u64,
     /// Next numeric suffix for message id generation (monotonic, >= 1).
     next_message_idx: u64,
 }
 
 impl StoredThread {
     fn new(thread: Thread) -> Self {
         StoredThread {
             thread,
             messages: Vec::new(),
             next_sequence: 1,
             next_message_idx: 1,
         }
     }
 
     fn to_thread(&self) -> Thread {
         self.thread.clone()
     }
 
     fn ensure_config_version_exists(
         &self,
         version: &ThreadConfigVersion,
     ) -> Result<(), ThreadStoreError> {
         if self
             .thread
             .config_history
             .iter()
             .any(|(v, _)| v == version)
         {
             Ok(())
         } else {
             Err(ThreadStoreError::Conflict)
         }
     }
 
     /// Append a message, assigning sequence and message_id if needed.
     fn append_message(&mut self, mut message: Message) -> Result<(), ThreadStoreError> {
         // Validate tenant/thread match exactly.
         if message.tenant_id != self.thread.tenant_id
             || message.thread_id != self.thread.thread_id
         {
             return Err(ThreadStoreError::PermissionDenied);
         }
 
         // Validate config_version_ref exists.
         self.ensure_config_version_exists(&message.config_version_ref)?;
 
         // Assign deterministic sequence if placeholder/zero.
         // Contract: monotonically increasing; we own it.
         if message.sequence == 0 {
             message.sequence = self.next_sequence;
         } else if message.sequence != self.next_sequence {
             // Any externally given non-zero that doesn't match expected is a conflict.
             return Err(ThreadStoreError::Conflict);
         }
         self.next_sequence = self
             .next_sequence
             .checked_add(1)
             .ok_or(ThreadStoreError::Internal)?;
 
         // Assign message_id if empty.
         if message.message_id.0.is_empty() {
             let mid = format!(
                 "{}:{}:msg-{}",
                 self.thread.tenant_id.0, self.thread.thread_id.0, self.next_message_idx
             );
             message.message_id = MessageId::new(mid);
         }
         self.next_message_idx = self
             .next_message_idx
             .checked_add(1)
             .ok_or(ThreadStoreError::Internal)?;
 
         // Ensure created_at is set (handler sets now; keep as-is if provided).
         // No extra validation required for determinism beyond ordering below.
 
         self.messages.push(message);
         // Keep messages sorted by (sequence, created_at, message_id) for determinism.
         self.messages.sort_by(|a, b| {
             a.sequence
                 .cmp(&b.sequence)
                 .then_with(|| a.created_at.cmp(&b.created_at))
                 .then_with(|| a.message_id.0.cmp(&b.message_id.0))
         });
 
         // Update thread.updated_at to now (mutation).
         self.thread.updated_at = SystemTime::now();
 
         Ok(())
     }
 
     fn messages_sorted(&self) -> Vec<Message> {
         // messages already maintained sorted; clone to avoid exposing internal mutability.
         self.messages.clone()
     }
 }
 
 /// Global in-memory state.
 #[derive(Default)]
 struct Inner {
     /// Threads keyed by `(tenant_id, thread_id)`.
     threads: HashMap<TenantThreadKey, StoredThread>,
     /// Next per-tenant thread index for deterministic ThreadId generation.
     next_thread_idx: HashMap<String, u64>,
     /// Idempotency records keyed by `(tenant_id, endpoint, key)`.
     idempotency: HashMap<IdempotencyKey, IdempotencyRecord>,
 }
 
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
 
     fn with_inner_mut<F, R>(&self, f: F) -> Result<R, ThreadStoreError>
     where
         F: FnOnce(&mut Inner) -> Result<R, ThreadStoreError>,
     {
         let mut guard = self.inner.write().map_err(|_| ThreadStoreError::Internal)?;
         f(&mut guard)
     }
 
     fn with_inner<F, R>(&self, f: F) -> Result<R, ThreadStoreError>
     where
         F: FnOnce(&Inner) -> Result<R, ThreadStoreError>,
     {
         let guard = self.inner.read().map_err(|_| ThreadStoreError::Internal)?;
         f(&guard)
     }
 }
 
 impl Default for InMemoryThreadStore {
     fn default() -> Self {
         Self::new()
     }
 }
 
 impl ThreadStore for InMemoryThreadStore {
     fn create_thread(
         &self,
         tenant: &TenantId,
         initial_config: ThreadConfigSnapshot,
     ) -> Result<Thread, ThreadStoreError> {
         self.with_inner_mut(|inner| {
             // Deterministic per-tenant ThreadId: "t:{tenant}:{n}"
             let next_idx = inner
                 .next_thread_idx
                 .entry(tenant.0.clone())
                 .and_modify(|v| *v = v.saturating_add(1))
                 .or_insert(1);
 
             let thread_id = ThreadId::new(format!("t:{}:{}", tenant.0, next_idx));
 
             if inner
                 .threads
                 .contains_key(&TenantThreadKey::new(tenant, &thread_id))
             {
                 // Extremely unlikely with monotonic counter, but respect contract.
                 return Err(ThreadStoreError::Conflict);
             }
 
             let version = ThreadConfigVersion::new(1);
             let now = SystemTime::now();
 
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
             inner.threads.insert(key, StoredThread::new(thread.clone()));
 
             Ok(thread)
         })
     }
 
     fn get_thread(
         &self,
         tenant: &TenantId,
         thread: &ThreadId,
     ) -> Result<Option<Thread>, ThreadStoreError> {
         self.with_inner(|inner| {
             let key = TenantThreadKey::new(tenant, thread);
             Ok(inner.threads.get(&key).map(|st| st.to_thread()))
         })
     }
 
     fn list_threads(&self, tenant: &TenantId) -> Result<Vec<Thread>, ThreadStoreError> {
         self.with_inner(|inner| {
             let mut threads: Vec<Thread> = inner
                 .threads
                 .iter()
                 .filter_map(|(key, st)| {
                     if key.tenant_id == tenant.0 {
                         Some(st.to_thread())
                     } else {
                         None
                     }
                 })
                 .collect();
 
             threads.sort_by(|a, b| {
                 a.created_at
                     .cmp(&b.created_at)
                     .then_with(|| a.thread_id.0.cmp(&b.thread_id.0))
             });
 
             Ok(threads)
         })
     }
 
     fn append_message(&self, message: Message) -> Result<(), ThreadStoreError> {
         self.with_inner_mut(|inner| {
             let key = TenantThreadKey::new(&message.tenant_id, &message.thread_id);
             let stored = inner
                 .threads
                 .get_mut(&key)
                 .ok_or(ThreadStoreError::NotFound)?;
 
             stored.append_message(message)
         })
     }
 
     fn get_messages(
         &self,
         tenant: &TenantId,
         thread: &ThreadId,
     ) -> Result<Vec<Message>, ThreadStoreError> {
         self.with_inner(|inner| {
             let key = TenantThreadKey::new(tenant, thread);
             match inner.threads.get(&key) {
                 Some(stored) => Ok(stored.messages_sorted()),
                 None => Err(ThreadStoreError::NotFound),
             }
         })
     }
 
     fn update_thread_config_append_only(
         &self,
         tenant: &TenantId,
         thread: &ThreadId,
         new_config: ThreadConfigSnapshot,
     ) -> Result<Thread, ThreadStoreError> {
         self.with_inner_mut(|inner| {
             let key = TenantThreadKey::new(tenant, thread);
             let stored = inner
                 .threads
                 .get_mut(&key)
                 .ok_or(ThreadStoreError::NotFound)?;
 
             // Determine next strictly increasing version.
             let next_version = stored
                 .thread
                 .config_history
                 .last()
                 .map(|(v, _)| ThreadConfigVersion(v.0 + 1))
                 .unwrap_or_else(|| ThreadConfigVersion::new(1));
 
             // Append-only: no mutation of existing entries.
             stored
                 .thread
                 .config_history
                 .push((next_version, new_config));
             stored.thread.current_config_version = next_version;
             stored.thread.updated_at = SystemTime::now();
 
             Ok(stored.to_thread())
         })
     }
 
     fn record_idempotency_key(
         &self,
         tenant: &TenantId,
         endpoint: &str,
         key: &str,
         value: IdempotencyRecord,
     ) -> Result<(), ThreadStoreError> {
         // Enforce tenant scoping.
         if value.tenant_id != *tenant {
             return Err(ThreadStoreError::PermissionDenied);
         }
 
         self.with_inner_mut(|inner| {
             let id_key = IdempotencyKey::new(tenant, endpoint, key);
 
             if let Some(existing) = inner.idempotency.get(&id_key) {
                 // Semantic idempotency: allow exact-match replays.
                 if existing.thread_id == value.thread_id
                     && existing.message_id == value.message_id
                     && existing.config_version == value.config_version
                     && existing.response_fingerprint == value.response_fingerprint
                 {
                     return Ok(());
                 }
                 return Err(ThreadStoreError::Conflict);
             }
 
             inner.idempotency.insert(id_key, value);
             Ok(())
         })
     }
 
     fn get_idempotency_record(
         &self,
         tenant: &TenantId,
         endpoint: &str,
         key: &str,
     ) -> Result<Option<IdempotencyRecord>, ThreadStoreError> {
         self.with_inner(|inner| {
             let id_key = IdempotencyKey::new(tenant, endpoint, key);
             Ok(inner.idempotency.get(&id_key).cloned())
         })
     }
 }
 
 #[cfg(test)]
 mod tests {
     use super::*;
 
     fn empty_config() -> ThreadConfigSnapshot {
         ThreadConfigSnapshot {
             model_id: None,
             model: None,
             temperature: None,
             max_tokens: None,
             system_prompt: None,
             history_packing_strategy: None,
             label: None,
         }
     }
 
     #[test]
     fn multi_tenant_isolation_threads_and_messages() {
         let store = InMemoryThreadStore::new();
         let t1 = TenantId::new("tenant1");
         let t2 = TenantId::new("tenant2");
 
         let thread1 = store.create_thread(&t1, empty_config()).unwrap();
         let thread2 = store.create_thread(&t2, empty_config()).unwrap();
 
         // Ensure different tenants, different IDs.
         assert_ne!(thread1.tenant_id, thread2.tenant_id);
         assert_ne!(thread1.thread_id, thread2.thread_id);
 
         // get_thread scoped by tenant.
         assert!(store
             .get_thread(&t1, &thread1.thread_id)
             .unwrap()
             .is_some());
         assert!(store
             .get_thread(&t1, &thread2.thread_id)
             .unwrap()
             .is_none());
 
         // list_threads scoped.
         let list_t1 = store.list_threads(&t1).unwrap();
         let list_t2 = store.list_threads(&t2).unwrap();
         assert_eq!(list_t1.len(), 1);
         assert_eq!(list_t1[0].tenant_id, t1);
         assert_eq!(list_t2.len(), 1);
         assert_eq!(list_t2[0].tenant_id, t2);
 
         // Messages do not cross tenants.
         let msg = Message {
             tenant_id: t1.clone(),
             thread_id: thread1.thread_id.clone(),
             message_id: MessageId::new(""),
             role: crate::thread::MessageRole::User,
             created_at: SystemTime::now(),
             sequence: 0,
             content: "hello".into(),
             config_version_ref: thread1.current_config_version,
             redacted: false,
             truncated: false,
             redaction_reason: None,
             audit: None,
         };
         store.append_message(msg).unwrap();
 
         // tenant1 can see one message
         let msgs_t1 = store
             .get_messages(&t1, &thread1.thread_id)
             .unwrap();
         assert_eq!(msgs_t1.len(), 1);
 
         // tenant2 cannot see messages of tenant1
         let res = store.get_messages(&t2, &thread1.thread_id);
         assert!(matches!(res, Err(ThreadStoreError::NotFound)));
     }
 
     #[test]
     fn append_only_config_history_and_versions() {
         let store = InMemoryThreadStore::new();
         let tenant = TenantId::new("t");
         let thread = store.create_thread(&tenant, empty_config()).unwrap();
 
         // Initial invariants.
         let fetched = store
             .get_thread(&tenant, &thread.thread_id)
             .unwrap()
             .unwrap();
         assert_eq!(fetched.current_config_version, ThreadConfigVersion(1));
         assert_eq!(fetched.config_history.len(), 1);
 
         // Append two new configs.
         let t2 = store
             .update_thread_config_append_only(&tenant, &thread.thread_id, empty_config())
             .unwrap();
         assert_eq!(t2.config_history.len(), 2);
         assert_eq!(t2.current_config_version, ThreadConfigVersion(2));
 
         let t3 = store
             .update_thread_config_append_only(&tenant, &thread.thread_id, empty_config())
             .unwrap();
         assert_eq!(t3.config_history.len(), 3);
         assert_eq!(t3.current_config_version, ThreadConfigVersion(3));
 
         // Ensure strictly increasing and append-only.
         let versions: Vec<u64> = t3.config_history.iter().map(|(v, _)| v.0).collect();
         assert_eq!(versions, vec![1, 2, 3]);
     }
 
     #[test]
     fn message_lifecycle_and_ordering() {
         let store = InMemoryThreadStore::new();
         let tenant = TenantId::new("tenant-msg");
         let thread = store.create_thread(&tenant, empty_config()).unwrap();
 
         // Append multiple messages with placeholder identity/sequence.
         for i in 0..5 {
             let msg = Message {
                 tenant_id: tenant.clone(),
                 thread_id: thread.thread_id.clone(),
                 message_id: MessageId::new(""),
                 role: crate::thread::MessageRole::User,
                 created_at: SystemTime::now(),
                 sequence: 0,
                 content: format!("m{}", i),
                 config_version_ref: thread.current_config_version,
                 redacted: false,
                 truncated: false,
                 redaction_reason: None,
                 audit: None,
             };
             store.append_message(msg).unwrap();
         }
 
         let msgs = store
             .get_messages(&tenant, &thread.thread_id)
             .unwrap();
         assert_eq!(msgs.len(), 5);
 
         // Sequences must be strictly increasing from 1..=5.
         for (i, m) in msgs.iter().enumerate() {
             assert_eq!(m.sequence, (i + 1) as u64);
             assert!(!m.message_id.0.is_empty());
         }
 
         // Tenant mismatch should be NotFound (no leakage).
         let other_tenant = TenantId::new("other");
         let res = store.get_messages(&other_tenant, &thread.thread_id);
         assert!(matches!(res, Err(ThreadStoreError::NotFound)));
     }
 
     #[test]
     fn idempotency_behavior_semantic_conflict() {
         let store = InMemoryThreadStore::new();
         let tenant = TenantId::new("tenant- idem");
         let endpoint = "POST /v1/test";
         let key = "k1";
 
         let base_record = IdempotencyRecord {
             tenant_id: tenant.clone(),
             thread_id: Some(ThreadId::new("th1")),
             message_id: Some(MessageId::new("m1")),
             config_version: Some(ThreadConfigVersion(1)),
             response_fingerprint: Some("fp".into()),
             created_at: SystemTime::now(),
             expires_at: None,
         };
 
         // First insert ok.
         store
             .record_idempotency_key(&tenant, endpoint, key, base_record.clone())
             .unwrap();
 
         // Identical second insert ok.
         store
             .record_idempotency_key(&tenant, endpoint, key, base_record.clone())
             .unwrap();
 
         // Different fingerprint => conflict.
         let conflicting = IdempotencyRecord {
             response_fingerprint: Some("different".into()),
             ..base_record.clone()
         };
         let err = store
             .record_idempotency_key(&tenant, endpoint, key, conflicting)
             .unwrap_err();
         assert!(matches!(err, ThreadStoreError::Conflict));
 
         // Retrieval works.
         let fetched = store
             .get_idempotency_record(&tenant, endpoint, key)
             .unwrap()
             .unwrap();
         assert_eq!(fetched.tenant_id, tenant);
         assert_eq!(fetched.response_fingerprint, base_record.response_fingerprint);
 
         // Cross-tenant isolation.
         let other = TenantId::new("other");
         let none = store
             .get_idempotency_record(&other, endpoint, key)
             .unwrap();
         assert!(none.is_none());
     }
 }