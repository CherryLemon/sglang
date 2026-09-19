// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use sgl_router::discovery::{ModelId, WorkerId, WorkerSpec};
use sgl_router::policies_reorg::admission::{
    AdmissionConfig, AdmissionContext, AdmissionMetric, Decision, EngineAdmission,
};
use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
use sgl_router::policies_reorg::{PickError, PickRequest, Policy, Stage};
use sgl_router::state::load_monitor::engine_load::{
    EngineLoadSnapshot, EngineLoadTable, LoadStat, NativeCacheRankLoad, NativeCacheWorkerLoad,
};
use sgl_router::workers::Worker;

fn engine() -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("w".into()),
        url: "http://w".into(),
        mode: Stage::Plain,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }))
}

fn snapshot(running: u64, waiting: u64, total_kv: u64, pending: u64) -> EngineLoadSnapshot {
    EngineLoadSnapshot::from_native_cache_workers(
        1,
        [(
            "http://w".into(),
            NativeCacheWorkerLoad {
                num_running_reqs: running,
                num_waiting_reqs: waiting,
                num_waiting_uncached_tokens: pending,
                num_used_tokens: 0,
                num_total_tokens: total_kv,
                max_total_num_tokens: 1000,
                max_running_requests: 100,
                prefill_throughput_tokens_per_s: None,
                estimated_prefill_queue_ms: None,
                captured_at: Instant::now(),
            },
        )]
        .into(),
    )
}

#[test]
fn config_requires_exactly_the_limits_for_its_name() {
    for value in [
        json!({"name": "allow_all"}),
        json!({"name": "running_plus_kv_capacity", "max_running_requests": 4, "max_kv_tokens": 100}),
        json!({"name": "in_flight_limit", "max_in_flight": 4}),
        json!({"name": "queue_limit", "max_waiting_requests": 2}),
        json!({"name": "pending_prefill", "max_pending_prefill_tokens": 100}),
    ] {
        let config: AdmissionConfig = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(config).unwrap(), value);
        let mut extra = value.clone();
        extra["unrelated_limit"] = json!(10);
        assert!(serde_json::from_value::<AdmissionConfig>(extra).is_err());
        for field in value
            .as_object()
            .unwrap()
            .keys()
            .filter(|key| *key != "name")
        {
            let mut missing = value.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<AdmissionConfig>(missing).is_err());
        }
    }
    for invalid in [
        json!({"name": "unknown"}),
        json!({"name": "allow_all", "max_running_requests": 4}),
        json!({"name": "in_flight_limit", "max_in_flight": -1}),
    ] {
        assert!(
            serde_json::from_value::<AdmissionConfig>(invalid.clone()).is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn capacity_checks_projected_usage_and_overflow() {
    let engine = engine();
    let model = ModelId("m".into());
    let config = AdmissionConfig::RunningPlusKvCapacity {
        max_running_requests: 4,
        max_kv_tokens: 100,
    };
    for (running, kv, peak, rejection) in [
        (3, 80, Some(20), None),
        (4, 80, Some(20), Some("running_capacity")),
        (3, 80, Some(21), Some("kv_capacity")),
        (3, 90, None, None),
        (3, 91, None, Some("kv_capacity")),
        (u64::MAX, 0, None, Some("running_capacity")),
        (0, 1, Some(u64::MAX), Some("kv_capacity")),
    ] {
        let mut request = PickRequest::new(&model, Stage::Plain, 10);
        request.expected_peak_tokens = peak;
        // Unrelated queue and prefill pressure must not affect capacity.
        let load = snapshot(running, u64::MAX, kv, u64::MAX);
        let context = AdmissionContext::new(&engine, &request, &load);
        assert_eq!(
            config.check(&context).unwrap(),
            rejection.map_or(Decision::Allow, |reason| Decision::Reject(reason.into()))
        );
        assert_eq!(
            AdmissionConfig::default().check(&context).unwrap(),
            Decision::Allow
        );
    }
    let request = PickRequest::new(&model, Stage::Plain, 1);
    let unlimited = AdmissionConfig::RunningPlusKvCapacity {
        max_running_requests: u64::MAX,
        max_kv_tokens: u64::MAX,
    };
    for (running, kv, reason) in [
        (u64::MAX, 0, "running_capacity"),
        (0, u64::MAX, "kv_capacity"),
    ] {
        let load = snapshot(running, 0, kv, 0);
        assert_eq!(
            unlimited
                .check(&AdmissionContext::new(&engine, &request, &load))
                .unwrap(),
            Decision::Reject(reason.into())
        );
    }
}

#[test]
fn inflight_reads_live_local_load_without_requiring_reports() {
    let engine = engine();
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    let empty = EngineLoadSnapshot::default();
    // Engine-reported running requests are a different quantity.
    let full = snapshot(u64::MAX, u64::MAX, u64::MAX, u64::MAX);
    for load in [&empty, &full] {
        let context = AdmissionContext::new(&engine, &request, load);
        let config: AdmissionConfig = serde_json::from_value(json!({
            "name": "in_flight_limit", "max_in_flight": 1
        }))
        .unwrap();
        assert_eq!(config.check(&context).unwrap(), Decision::Allow);
        let guard = engine.load_guard();
        assert_eq!(
            config.check(&context).unwrap(),
            Decision::Reject("in_flight_limit".into())
        );
        drop(guard);
        assert_eq!(config.check(&context).unwrap(), Decision::Allow);
        assert_eq!(
            AdmissionConfig::InFlightLimit { max_in_flight: 0 }
                .check(&context)
                .unwrap(),
            Decision::Reject("in_flight_limit".into())
        );
    }
}

#[test]
fn pending_prefill_projects_uncached_work_and_validates_its_scope() {
    let engine = engine();
    let model = ModelId("m".into());
    let config = AdmissionConfig::PendingPrefill {
        max_pending_prefill_tokens: 100,
    };
    // Other overloaded measurements do not affect this rule.
    let load = snapshot(u64::MAX, u64::MAX, u64::MAX, 80);
    for stage in [Stage::Plain, Stage::Prefill] {
        let request = PickRequest::new(&model, stage, 100);
        let context = AdmissionContext::new(&engine, &request, &load);
        assert_eq!(
            config.check(&context).unwrap(),
            Decision::Reject("pending_prefill_limit".into())
        );
        for (uncached, expected) in [
            (20, Decision::Allow),
            (21, Decision::Reject("pending_prefill_limit".into())),
            (0, Decision::Allow),
        ] {
            let context =
                AdmissionContext::new(&engine, &request, &load).with_uncached_tokens(uncached);
            assert_eq!(config.check(&context).unwrap(), expected);
        }
        assert!(matches!(
            config.check(&context.with_uncached_tokens(101)),
            Err(PickError::InvalidSignal(_))
        ));
    }
    let decode = PickRequest::new(&model, Stage::Decode, 100);
    assert!(matches!(
        config.check(&AdmissionContext::new(&engine, &decode, &load)),
        Err(PickError::InvalidConfiguration(_))
    ));
    let request = PickRequest::new(&model, Stage::Prefill, 1);
    let load = snapshot(0, 0, 0, u64::MAX);
    assert_eq!(
        AdmissionConfig::PendingPrefill {
            max_pending_prefill_tokens: u64::MAX
        }
        .check(&AdmissionContext::new(&engine, &request, &load))
        .unwrap(),
        Decision::Reject("pending_prefill_limit".into())
    );
}

#[test]
fn queue_checks_reported_waiting_count_and_retains_the_snapshot() {
    let engine = engine();
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Plain, 10);
    let table = EngineLoadTable::new();
    let sample = |waiting| LoadStat {
        num_running_reqs: 100,
        num_waiting_reqs: waiting,
        num_tokens: 1000,
        max_total_num_tokens: 1000,
        native_cache: None,
    };
    table.set(&engine.url, 0, sample(2), Instant::now());
    let load = table.capture_snapshot(Instant::now());
    let context = AdmissionContext::new(&engine, &request, &load);
    let config = AdmissionConfig::QueueLimit {
        max_waiting_requests: 2,
    };
    assert_eq!(
        config.check(&context).unwrap(),
        Decision::Reject("queue_limit".into())
    );
    table.set(&engine.url, 0, sample(1), Instant::now());
    assert_eq!(
        config.check(&context).unwrap(),
        Decision::Reject("queue_limit".into())
    );
    let newer = table.capture_snapshot(Instant::now());
    assert_eq!(
        config
            .check(&AdmissionContext::new(&engine, &request, &newer))
            .unwrap(),
        Decision::Allow
    );
    // Basic reports suffice for running and queue limits, but not total KV or uncached work.
    assert_eq!(context.current(AdmissionMetric::KvTokens), None);
    assert_eq!(context.current(AdmissionMetric::PendingPrefillTokens), None);
    assert_eq!(
        AdmissionConfig::RunningPlusKvCapacity {
            max_running_requests: 100,
            max_kv_tokens: 0
        }
        .check(&context)
        .unwrap(),
        Decision::Reject("running_capacity".into())
    );
    assert_eq!(
        AdmissionConfig::PendingPrefill {
            max_pending_prefill_tokens: 0
        }
        .check(&context)
        .unwrap(),
        Decision::Allow
    );
}

#[test]
fn missing_telemetry_fails_open_only_for_reported_metrics() {
    use std::time::Duration;
    let engine = engine();
    let model = ModelId("m".into());
    let request = PickRequest::new(&model, Stage::Prefill, 10);
    for case in ["missing", "stale", "incomplete"] {
        let table = EngineLoadTable::new();
        if case != "missing" {
            let at = if case == "stale" {
                Instant::now() - Duration::from_secs(3600)
            } else {
                Instant::now()
            };
            table.set(
                &engine.url,
                0,
                LoadStat {
                    num_running_reqs: 10,
                    num_waiting_reqs: 10,
                    num_tokens: 10,
                    max_total_num_tokens: 100,
                    native_cache: Some(NativeCacheRankLoad {
                        num_waiting_uncached_tokens: 10,
                        num_total_tokens: 10,
                        max_running_requests: 100,
                        total_prefill_uncached_tokens: 0,
                        total_prefill_busy_us: 0,
                    }),
                },
                at,
            );
            if case == "incomplete" {
                table.mark_expected_rank(&engine.url, 1);
            }
        }
        let load = table.capture_snapshot(Instant::now());
        let context = AdmissionContext::new(&engine, &request, &load);
        for config in [
            AdmissionConfig::RunningPlusKvCapacity {
                max_running_requests: 0,
                max_kv_tokens: 0,
            },
            AdmissionConfig::QueueLimit {
                max_waiting_requests: 0,
            },
            AdmissionConfig::PendingPrefill {
                max_pending_prefill_tokens: 0,
            },
        ] {
            assert_eq!(
                config.check(&context).unwrap(),
                Decision::Allow,
                "{case}: {config:?}"
            );
        }
        assert_eq!(
            AdmissionConfig::InFlightLimit { max_in_flight: 0 }
                .check(&context)
                .unwrap(),
            Decision::Reject("in_flight_limit".into()),
            "{case}"
        );
    }
}

#[tokio::test]
async fn capacity_uses_configured_limits_and_total_kv_from_selection_snapshot() {
    let engine = engine();
    let table = EngineLoadTable::new();
    let mut report = LoadStat {
        num_running_reqs: 1,
        num_waiting_reqs: 0,
        num_tokens: 10,
        max_total_num_tokens: 1000,
        native_cache: Some(NativeCacheRankLoad {
            num_waiting_uncached_tokens: 0,
            num_total_tokens: 80,
            max_running_requests: 100,
            total_prefill_uncached_tokens: 0,
            total_prefill_busy_us: 0,
        }),
    };
    table.set(&engine.url, 0, report.clone(), Instant::now());
    let snapshot = table.capture_snapshot(Instant::now());
    let config = AdmissionConfig::RunningPlusKvCapacity {
        max_running_requests: 2,
        max_kv_tokens: 100,
    };
    let mut policy = PowerOfTwoPolicy::new(table.clone());
    policy.admission = Arc::new(config.clone());
    let model = ModelId("m".into());
    let mut request = PickRequest::new(&model, Stage::Plain, 10);
    request.expected_peak_tokens = Some(21);
    let context = AdmissionContext::new(&engine, &request, &snapshot);
    assert_eq!(context.current(AdmissionMetric::KvTokens), Some(80));
    assert!(matches!(
        policy.pick(std::slice::from_ref(&engine), &request).await,
        Err(PickError::AdmissionRejected(reason)) if reason.reason == "kv_capacity"
    ));
    // A later report affects the next selection, not the retained observation.
    report.native_cache.as_mut().unwrap().num_total_tokens = 0;
    table.set(&engine.url, 0, report.clone(), Instant::now());
    assert_eq!(
        config.check(&context).unwrap(),
        Decision::Reject("kv_capacity".into())
    );
    assert!(policy
        .pick(std::slice::from_ref(&engine), &request)
        .await
        .is_ok());
    report.num_running_reqs = 2;
    table.set(&engine.url, 0, report, Instant::now());
    assert!(matches!(
        policy.pick(std::slice::from_ref(&engine), &request).await,
        Err(PickError::AdmissionRejected(reason)) if reason.reason == "running_capacity"
    ));
}
