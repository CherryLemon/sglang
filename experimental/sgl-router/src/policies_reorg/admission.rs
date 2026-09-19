// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::state::load_monitor::engine_load::EngineLoadSnapshot;
use crate::workers::Worker;

use super::{PickError, PickRequest, Stage};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Reject(String),
}

/// Quantities admission can bound. Their sources and request increments are
/// defined once by AdmissionContext, independently of named policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionMetric {
    RunningRequests,
    KvTokens,
    InFlightRequests,
    WaitingRequests,
    PendingPrefillTokens,
}

impl AdmissionMetric {
    fn rejection_reason(self) -> &'static str {
        match self {
            Self::RunningRequests => "running_capacity",
            Self::KvTokens => "kv_capacity",
            Self::InFlightRequests => "in_flight_limit",
            Self::WaitingRequests => "queue_limit",
            Self::PendingPrefillTokens => "pending_prefill_limit",
        }
    }
}

/// An absolute upper bound on one quantity, including the incoming request.
#[derive(Debug, Clone, Copy)]
struct AdmissionLimit {
    metric: AdmissionMetric,
    max: u64,
}

impl AdmissionLimit {
    fn new(metric: AdmissionMetric, max: u64) -> Self {
        Self { metric, max }
    }
}

/// Selected engine, request facts, and the snapshot already used by selection.
/// The checker reads only the measurements required by its configured limits.
/// Router-local in-flight load is read live; telemetry stays on this snapshot.
#[derive(Debug)]
pub struct AdmissionContext<'a> {
    pub engine: &'a Worker,
    pub request: &'a PickRequest<'a>,
    load: &'a EngineLoadSnapshot,
    uncached_tokens: Option<u64>,
}

impl<'a> AdmissionContext<'a> {
    pub fn new(
        engine: &'a Worker,
        request: &'a PickRequest<'a>,
        load: &'a EngineLoadSnapshot,
    ) -> Self {
        Self {
            engine,
            request,
            load,
            uncached_tokens: None,
        }
    }

    /// Candidate-specific uncached work, supplied by cache-aware selection.
    /// Without a cache observation, pending-prefill admission uses full input.
    pub fn with_uncached_tokens(mut self, tokens: u64) -> Self {
        self.uncached_tokens = Some(tokens);
        self
    }

    /// Missing, stale, or incomplete telemetry is unknown, never zero.
    pub fn current(&self, metric: AdmissionMetric) -> Option<u64> {
        match metric {
            AdmissionMetric::RunningRequests => self
                .load
                .fresh_load_for_url(&self.engine.url)
                .map(|load| load.num_running_reqs),
            AdmissionMetric::KvTokens => self
                .load
                .fresh_native_cache_load_for_url(&self.engine.url)
                .map(|load| load.num_total_tokens),
            AdmissionMetric::InFlightRequests => Some(self.engine.active_load() as u64),
            AdmissionMetric::WaitingRequests => self
                .load
                .fresh_load_for_url(&self.engine.url)
                .map(|load| load.num_waiting_reqs),
            AdmissionMetric::PendingPrefillTokens => self
                .load
                .fresh_native_cache_load_for_url(&self.engine.url)
                .map(|load| load.num_waiting_uncached_tokens),
        }
    }

    fn incoming(&self, metric: AdmissionMetric) -> Result<u64, PickError> {
        match metric {
            AdmissionMetric::RunningRequests
            | AdmissionMetric::InFlightRequests
            | AdmissionMetric::WaitingRequests => Ok(1),
            AdmissionMetric::KvTokens => Ok(self.request.kv_tokens()),
            AdmissionMetric::PendingPrefillTokens => {
                if self.request.stage == Stage::Decode {
                    return Err(PickError::InvalidConfiguration(
                        "pending_prefill admission requires plain or prefill routing".into(),
                    ));
                }
                let tokens = self.uncached_tokens.unwrap_or(self.request.input_tokens);
                if tokens > self.request.input_tokens {
                    return Err(PickError::InvalidSignal(
                        "uncached tokens exceed request input tokens".into(),
                    ));
                }
                Ok(tokens)
            }
        }
    }
}

/// Checks one selected engine. Selection owns when to check and how to handle
/// rejection; admission does not select replacements or reserve capacity.
pub trait EngineAdmission: Send + Sync + Debug {
    fn check(&self, context: &AdmissionContext<'_>) -> Result<Decision, PickError>;
}

/// Each name carries only its relevant, required absolute limits.
/// Named policies map to metric/limit pairs evaluated by the same checker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdmissionConfig {
    AllowAll {},
    RunningPlusKvCapacity {
        max_running_requests: u64,
        max_kv_tokens: u64,
    },
    InFlightLimit {
        max_in_flight: u64,
    },
    QueueLimit {
        max_waiting_requests: u64,
    },
    PendingPrefill {
        max_pending_prefill_tokens: u64,
    },
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self::AllowAll {}
    }
}

impl EngineAdmission for AdmissionConfig {
    fn check(&self, context: &AdmissionContext<'_>) -> Result<Decision, PickError> {
        use AdmissionMetric::*;

        // A new named policy only needs its config fields and this mapping.
        // Stack-backed slices allow any number of limits without allocating.
        let limits: &[AdmissionLimit] = match *self {
            Self::AllowAll {} => &[],
            Self::RunningPlusKvCapacity {
                max_running_requests,
                max_kv_tokens,
            } => &[
                AdmissionLimit::new(RunningRequests, max_running_requests),
                AdmissionLimit::new(KvTokens, max_kv_tokens),
            ],
            Self::InFlightLimit { max_in_flight } => {
                &[AdmissionLimit::new(InFlightRequests, max_in_flight)]
            }
            Self::QueueLimit {
                max_waiting_requests,
            } => &[AdmissionLimit::new(WaitingRequests, max_waiting_requests)],
            Self::PendingPrefill {
                max_pending_prefill_tokens,
            } => &[AdmissionLimit::new(
                PendingPrefillTokens,
                max_pending_prefill_tokens,
            )],
        };

        for limit in limits {
            let incoming = context.incoming(limit.metric)?;
            // Unknown telemetry fails open. Overflow is known not to fit.
            if context.current(limit.metric).is_some_and(|current| {
                current
                    .checked_add(incoming)
                    .is_none_or(|projected| projected > limit.max)
            }) {
                return Ok(Decision::Reject(limit.metric.rejection_reason().into()));
            }
        }
        Ok(Decision::Allow)
    }
}
