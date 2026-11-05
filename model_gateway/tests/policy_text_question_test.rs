//! Policy paths a request walks without allocating, pinned: the registry's
//! question whether some policy wants the request's text (the answer is kept
//! where a policy is added or removed), the least-load decision, which
//! allocates its eligible list and nothing else once it has dispatched to a
//! worker, and a keyed hit on a sticky pin, which pays its namespaced key
//! under the routing-key override and its eligible list in the manual policy,
//! with the records of their metrics, and nothing besides. A separate
//! executable, so it can declare the counting allocator the pins need.

#![expect(unsafe_code, reason = "a global allocator that counts allocations")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
    sync::Arc,
};

use http::HeaderMap;
use openai_protocol::worker::HealthCheckConfig;
use smg::{
    config::types::{ManualAssignmentMode, PolicyConfig, RoutingKeyOverrideConfig},
    policies::{
        LeastLoadPolicy, LoadBalancingPolicy, ManualPolicy, PolicyRegistry, SelectWorkerInfo,
    },
    worker::{
        load_state::{one_rank_load, TestFeed},
        BasicWorkerBuilder, Worker, WorkerType,
    },
};

struct Counting;

thread_local! {
    /// The allocation calls of this thread while one of its pins measures,
    /// `None` outside a measurement. A thread counts its own calls alone, so
    /// neither the test harness nor a sibling test on another thread reaches
    /// a pin's count. Const-initialised with no destructor, so reading it
    /// inside the allocator allocates nothing and never runs a destructor.
    static COUNTED: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Counts one allocation call on this thread while it measures; a thread
/// whose local is gone (tearing down) counts nothing.
fn count() {
    let _ = COUNTED.try_with(|counted| {
        if let Some(calls) = counted.get() {
            counted.set(Some(calls + 1));
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: the layout is the caller's, handed to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocation calls `work` makes on this thread: nothing another thread
/// does in the meantime is counted.
fn allocations_during(work: impl FnOnce()) -> usize {
    COUNTED.with(|counted| counted.set(Some(0)));
    work();
    COUNTED.with(|counted| counted.replace(None)).unwrap_or(0)
}

/// A cache-aware default (text wanted, no eviction thread) and 64 models on
/// hinted cache-aware policies of their own: a request's question allocates
/// nothing, with and without a routing key and token ids, however many
/// models are registered.
#[test]
fn a_request_asks_whether_text_is_wanted_without_allocating() {
    let registry = PolicyRegistry::with_override(
        PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096),
        RoutingKeyOverrideConfig {
            enabled: true,
            ..Default::default()
        },
    );
    for model in 0..64 {
        registry.on_worker_added(&format!("model-{model}"), Some("cache_aware"));
    }
    let plain = HeaderMap::new();
    let mut keyed = HeaderMap::new();
    keyed.insert(
        "x-smg-routing-key",
        "session-A".parse().expect("a header value"),
    );

    assert!(registry.any_policy_needs_request_text(Some(&plain)));
    assert!(!registry.any_policy_needs_request_text(Some(&keyed)));
    assert!(!registry.any_policy_needs_request_text_with_tokens(Some(&plain)));

    let allocations = allocations_during(|| {
        for _ in 0..1_000 {
            black_box(registry.any_policy_needs_request_text(Some(&plain)));
            black_box(registry.any_policy_needs_request_text(Some(&keyed)));
            black_box(registry.any_policy_needs_request_text_with_tokens(Some(&plain)));
            black_box(registry.any_policy_needs_request_text_with_tokens(None));
        }
    });
    assert_eq!(
        allocations, 0,
        "the text questions allocated {allocations} times over 4,000 requests"
    );
}

/// Eight ready regular workers (no health checker behind them), each with a
/// published report and requests in flight, under a least-load policy that
/// has dispatched to every one of them: a decision allocates its eligible
/// list and nothing else. The winner's credit finds its tally by a borrowed
/// key; only a worker's first dispatch allocates the key. A report is
/// published before every decision, so the credit of the one before is
/// released and the tally holds at most one dispatch per worker.
#[test]
fn a_least_load_decision_allocates_its_eligible_list_and_nothing_else() {
    let policy = LeastLoadPolicy::new();
    let feed = TestFeed::attached_to(&policy);
    let workers: Vec<Arc<dyn Worker>> = (0..8)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("http://worker-{i}:8000"))
                    .worker_type(WorkerType::Regular)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    let reports = workers
        .iter()
        .enumerate()
        .map(|(i, worker)| {
            for _ in 0..i % 3 {
                worker.increment_load();
            }
            (
                worker.url().to_string(),
                one_rank_load(512 * i as i32, 0.1 * i as f64, 2_000.0 + 100.0 * i as f64),
            )
        })
        .collect();
    feed.publish(&reports);
    let tokens: Vec<u32> = (0..1_024).collect();
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        ..Default::default()
    };
    // Every worker has been dispatched to: the tally knows each key.
    for _ in 0..64 {
        black_box(policy.select_worker(&workers, &info));
    }
    let decisions = 1_000;
    let mut allocations = 0;
    for _ in 0..decisions {
        feed.publish(&reports);
        allocations += allocations_during(|| {
            black_box(policy.select_worker(&workers, &info));
        });
    }
    // The eligible list is collected from a filter without a size hint: four
    // indices, then eight, two blocks for this pool.
    assert_eq!(
        allocations,
        2 * decisions,
        "least-load decisions allocated {allocations} times over {decisions} decisions; the \
         eligible list (two blocks for eight workers) is the budget"
    );
}

/// What a keyed hit still pays once its key is read borrowed and the
/// eligible list is built only to assign, per hit: under the routing-key
/// override the namespaced key it formats (model, role and key in one
/// string, the map's key) and the records of its metrics (blocks in the
/// metrics layer, outside this path); in the manual policy its eligible list
/// (a filter collected without a size hint: four indices, then eight) and
/// the record of its branch metric. Recorded on the unchanged code as the
/// hit's count less the key copy and the list, then held after the change.
const OVERRIDE_HIT_BUDGET: usize = 5;

const MANUAL_HIT_BUDGET: usize = 3;

/// Eight ready regular workers (no health checker behind them) and sixteen
/// conversations pinned once. A keyed hit under the routing-key override
/// (delegating its assignments to a round-robin registry) pays its budget
/// and nothing besides: the pin is read through the borrowed key, and no
/// eligible list is built for a hit. A keyed hit in the manual policy pays
/// its budget and nothing besides: its key is read borrowed too.
#[test]
fn a_keyed_hit_pays_its_key_under_the_override_and_its_eligible_list_in_manual() {
    let workers: Vec<Arc<dyn Worker>> = (0..8)
        .map(|i| {
            Arc::new(
                BasicWorkerBuilder::new(format!("http://worker-{i}:8000"))
                    .worker_type(WorkerType::Regular)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    let model_id = workers[0].model_id().to_string();
    let registry = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        RoutingKeyOverrideConfig {
            enabled: true,
            assignment_mode: ManualAssignmentMode::Delegate,
            ..Default::default()
        },
    );
    let policy = registry.get_default_policy();
    let manual = ManualPolicy::new();
    let keys: Vec<String> = (0..16).map(|i| format!("conversation-{i:04}")).collect();
    let infos: Vec<SelectWorkerInfo> = keys
        .iter()
        .map(|key| SelectWorkerInfo {
            rid_key: Some(key),
            ..Default::default()
        })
        .collect();
    for info in &infos {
        assert!(registry
            .select_worker_for_model(&policy, &model_id, &workers, info)
            .is_some());
        assert!(manual.select_worker(&workers, info).is_some());
    }

    let hits = 1_000;
    let mut override_allocations = 0;
    let mut manual_allocations = 0;
    for hit in 0..hits {
        let info = &infos[hit % infos.len()];
        override_allocations += allocations_during(|| {
            black_box(registry.select_worker_for_model(&policy, &model_id, &workers, info));
        });
        manual_allocations += allocations_during(|| {
            black_box(manual.select_worker(&workers, info));
        });
    }
    assert!(
        override_allocations == OVERRIDE_HIT_BUDGET * hits
            && manual_allocations == MANUAL_HIT_BUDGET * hits,
        "keyed hits allocated {override_allocations} times under the override and \
         {manual_allocations} times in the manual policy over {hits} hits; the budgets are \
         {OVERRIDE_HIT_BUDGET} and {MANUAL_HIT_BUDGET} per hit"
    );
}
