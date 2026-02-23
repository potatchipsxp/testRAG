#!/usr/bin/env python3
"""
benchmark_incidents.py
----------------------
Evaluates a RAG configuration against the 25 synthetic PaaS incidents.

Two evaluation layers:
  1. RETRIEVAL — did the right log chunks come back?
     Scored by checking whether expected signal phrases appear in retrieved text.

  2. ANSWER — did the LLM correctly diagnose / recommend a fix?
     Scored by checking whether required answer keywords appear in the response.
     Graded: full_credit | partial | miss

Results are saved to benchmark_results.json and a summary is printed.

Usage:
    python benchmark_incidents.py

Configure the CONFIG block to match your build_index.py settings.
All other settings (questions, expected signals) are pre-populated.
"""

import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime

# Reuse helpers from query_index.py (must be in same directory)
try:
    from query_index import (
        make_collection_name, get_collection,
        retrieve_chunks, build_prompt, generate_answer,
    )
except ImportError:
    sys.exit("query_index.py not found in current directory.")


# ============================================================================
# CONFIG — edit to match build_index.py settings
# ============================================================================

DB_PATH           = "./chroma_db"
BASE_COLLECTION   = "logs"
COLLECTION_SUFFIX = ""

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
CHUNK_METHOD      = "record"
CHUNK_SIZE        = 1
CHUNK_OVERLAP     = 0

LLM_MODEL         = "llama3.2"
N_RESULTS         = 15   # slightly wider than default to give retrieval a fair chance
VERBOSE           = False   # set True to see LLM answers as they run

RESULTS_FILE      = "benchmark_results.json"


# ============================================================================
# BENCHMARK CASES
#
# Each case has:
#   incident_id      : for grouping in results
#   tier             : 1 / 2 / 3
#   question_type    : "retrieval_probe" | "diagnosis" | "remediation"
#   question         : the natural-language query
#   where_filter     : optional ChromaDB metadata pre-filter
#   retrieval_signals: list of strings — at least one must appear in any
#                      retrieved chunk's text for retrieval to pass.
#                      Use short, distinctive substrings from the actual
#                      log messages in the incident files.
#   answer_required  : ALL of these substrings (case-insensitive) must
#                      appear in the LLM answer for full credit.
#   answer_partial   : If answer_required fails, check these — any hit
#                      gives partial credit (shows partial understanding).
# ============================================================================

BENCHMARK_CASES = [

    # ── TIER 1 ───────────────────────────────────────────────────────────────

    {
        "incident_id": "INC-001",
        "tier": 1,
        "question_type": "retrieval_probe",
        "question": "An app is repeatedly crashing. What port is it listening on and what port does the platform expect?",
        "where_filter": None,
        "retrieval_signals": ["port 8080", "port 3000", "health check failed"],
        "answer_required": ["3000", "8080"],
        "answer_partial":  ["port", "health check"],
    },
    {
        "incident_id": "INC-001",
        "tier": 1,
        "question_type": "diagnosis",
        "question": "An app instance is entering a crash loop with repeated health check failures. What is the root cause?",
        "where_filter": {"level": "ERROR"},
        "retrieval_signals": ["health check failed", "port 8080", "Listening on 0.0.0.0:3000"],
        "answer_required": ["port", "8080"],
        "answer_partial":  ["health check", "crash"],
    },
    {
        "incident_id": "INC-001",
        "tier": 1,
        "question_type": "remediation",
        "question": "How should an app be fixed when the platform health check can't connect but the app starts successfully?",
        "where_filter": None,
        "retrieval_signals": ["port 8080", "Listening on 0.0.0.0:3000", "health check"],
        "answer_required": ["port", "$PORT"],
        "answer_partial":  ["environment variable", "8080", "binding"],
    },

    {
        "incident_id": "INC-002",
        "tier": 1,
        "question_type": "diagnosis",
        "question": "An app was killed by the platform. What resource was exhausted and what was the app writing?",
        "where_filter": None,
        "retrieval_signals": ["disk quota", "no space left on device", "app.log"],
        "answer_required": ["disk", "quota"],
        "answer_partial":  ["space", "log"],
    },
    {
        "incident_id": "INC-002",
        "tier": 1,
        "question_type": "remediation",
        "question": "What should be done to prevent a container from being killed due to disk usage growth?",
        "where_filter": None,
        "retrieval_signals": ["disk quota", "no space left", "container_disk_usage"],
        "answer_required": ["log rotation", "quota"],
        "answer_partial":  ["disk", "syslog", "limit"],
    },

    {
        "incident_id": "INC-003",
        "tier": 1,
        "question_type": "diagnosis",
        "question": "An app fails to start and exits with status 1. What error appears in stderr and what service is it trying to reach?",
        "where_filter": {"level": "ERROR"},
        "retrieval_signals": ["gaierror", "postgres-svc", "Name or service not known"],
        "answer_required": ["dns", "postgres"],
        "answer_partial":  ["hostname", "not known", "service"],
    },

    {
        "incident_id": "INC-004",
        "tier": 1,
        "question_type": "retrieval_probe",
        "question": "Show log evidence of a container being killed due to memory usage.",
        "where_filter": None,
        "retrieval_signals": ["OOM", "OutOfMemoryError", "memory limit", "oom"],
        "answer_required": ["memory", "512"],
        "answer_partial":  ["killed", "OOM", "heap"],
    },
    {
        "incident_id": "INC-004",
        "tier": 1,
        "question_type": "diagnosis",
        "question": "Memory metrics show a steady climb before an app crash. What is the likely cause and what JVM error confirms it?",
        "where_filter": None,
        "retrieval_signals": ["OutOfMemoryError", "memory_bytes", "99% quota"],
        "answer_required": ["OutOfMemoryError", "heap"],
        "answer_partial":  ["memory", "quota", "gc"],
    },

    {
        "incident_id": "INC-005",
        "tier": 1,
        "question_type": "diagnosis",
        "question": "An app suddenly cannot connect to its message queue at midnight. What security error appears and why?",
        "where_filter": None,
        "retrieval_signals": ["CERTIFICATE_VERIFY_FAILED", "certificate has expired", "amqps://"],
        "answer_required": ["certificate", "expired"],
        "answer_partial":  ["ssl", "tls", "rabbitmq"],
    },
    {
        "incident_id": "INC-005",
        "tier": 1,
        "question_type": "remediation",
        "question": "How should a TLS certificate expiration on a bound service be resolved in a PaaS environment?",
        "where_filter": None,
        "retrieval_signals": ["certificate has expired", "CERTIFICATE_VERIFY_FAILED", "amqps://"],
        "answer_required": ["bind", "certificate"],
        "answer_partial":  ["rotate", "unbind", "service", "credential"],
    },

    {
        "incident_id": "INC-006",
        "tier": 1,
        "question_type": "diagnosis",
        "question": "A Node.js app build fails with a package integrity error. What package failed and what does the error code indicate?",
        "where_filter": None,
        "retrieval_signals": ["EINTEGRITY", "sha512", "express@4.18.2"],
        "answer_required": ["integrity", "cache"],
        "answer_partial":  ["npm", "sha", "express"],
    },

    {
        "incident_id": "INC-007",
        "tier": 1,
        "question_type": "diagnosis",
        "question": "Users are getting 502 errors immediately after a deployment that completed successfully. What does the router report?",
        "where_filter": {"level": "WARN"},
        "retrieval_signals": ["No route registered", "502", "api.example.com"],
        "answer_required": ["route", "mapped"],
        "answer_partial":  ["502", "backend", "register"],
    },

    # ── TIER 2 ───────────────────────────────────────────────────────────────

    {
        "incident_id": "INC-008",
        "tier": 2,
        "question_type": "retrieval_probe",
        "question": "Find log entries showing database connection pool saturation.",
        "where_filter": None,
        "retrieval_signals": ["pool exhausted", "db_pool_active_connections=20", "connection pool exhausted"],
        "answer_required": ["pool", "20"],
        "answer_partial":  ["connection", "exhausted", "database"],
    },
    {
        "incident_id": "INC-008",
        "tier": 2,
        "question_type": "diagnosis",
        "question": "503 errors are spiking and scaling the app out did not resolve them. What is the actual bottleneck and what evidence points to it?",
        "where_filter": None,
        "retrieval_signals": ["db_pool_active_connections=20", "Slow query", "connection pool exhausted"],
        "answer_required": ["slow quer", "pool"],
        "answer_partial":  ["database", "connection", "scaling"],
    },
    {
        "incident_id": "INC-008",
        "tier": 2,
        "question_type": "remediation",
        "question": "After identifying the cause of a connection pool exhaustion incident, what are the correct steps to fix it?",
        "where_filter": None,
        "retrieval_signals": ["slow query", "pool exhausted", "db_pool"],
        "answer_required": ["index", "query"],
        "answer_partial":  ["timeout", "pool size", "optimize"],
    },

    {
        "incident_id": "INC-009",
        "tier": 2,
        "question_type": "diagnosis",
        "question": "Eight app instances crashed simultaneously on the same cell. What happened to that cell and what was the downstream effect?",
        "where_filter": None,
        "retrieval_signals": ["cell-009", "cell OOM", "Rescheduling", "mass reschedule"],
        "answer_required": ["cell", "memory", "reschedule"],
        "answer_partial":  ["OOM", "instance", "killed"],
    },

    {
        "incident_id": "INC-010",
        "tier": 2,
        "question_type": "diagnosis",
        "question": "An app is getting ECONNREFUSED errors trying to call an internal service. What does the network layer log show?",
        "where_filter": None,
        "retrieval_signals": ["DENY", "silk.daemon", "network-policy", "ECONNREFUSED"],
        "answer_required": ["network policy", "deny"],
        "answer_partial":  ["policy", "blocked", "connection refused"],
    },
    {
        "incident_id": "INC-010",
        "tier": 2,
        "question_type": "remediation",
        "question": "Service-to-service HTTP calls are being refused despite both apps running. What is the fix?",
        "where_filter": None,
        "retrieval_signals": ["DENY", "policy", "ECONNREFUSED", "inventory-svc"],
        "answer_required": ["network-policy", "allow"],
        "answer_partial":  ["policy", "port", "cf add"],
    },

    {
        "incident_id": "INC-011",
        "tier": 2,
        "question_type": "diagnosis",
        "question": "A Python app builds successfully but crashes immediately on startup with an ImportError. What mismatch caused this?",
        "where_filter": None,
        "retrieval_signals": ["numpy", "1.21", "ImportError", "buildpack default"],
        "answer_required": ["numpy", "version"],
        "answer_partial":  ["import", "buildpack", "1.24"],
    },

    {
        "incident_id": "INC-012",
        "tier": 2,
        "question_type": "diagnosis",
        "question": "Multiple deployments are failing in the staging pipeline with a 429 error from an external registry. What is happening?",
        "where_filter": None,
        "retrieval_signals": ["toomanyrequests", "rate limit", "registry-1.docker.io", "429"],
        "answer_required": ["rate limit", "docker"],
        "answer_partial":  ["429", "registry", "pull"],
    },

    {
        "incident_id": "INC-013",
        "tier": 2,
        "question_type": "diagnosis",
        "question": "A service binding operation is blocking deployment indefinitely. What timed out and what was the underlying cause?",
        "where_filter": None,
        "retrieval_signals": ["broker_timeout", "Bind timeout", "Redis", "master node unreachable"],
        "answer_required": ["broker", "redis"],
        "answer_partial":  ["timeout", "service", "bind"],
    },

    {
        "incident_id": "INC-014",
        "tier": 2,
        "question_type": "diagnosis",
        "question": "Error rates spiked during a rolling deployment but the app was not crashing. What caused the mixed response behavior?",
        "where_filter": None,
        "retrieval_signals": ["mixed pool", "v1", "v2", "idempotency_key", "autoscaler"],
        "answer_required": ["version", "mixed"],
        "answer_partial":  ["autoscaler", "deploy", "schema"],
    },

    {
        "incident_id": "INC-015",
        "tier": 2,
        "question_type": "diagnosis",
        "question": "Metrics and logs appear to be missing for some apps. What platform component is responsible and what does it report?",
        "where_filter": None,
        "retrieval_signals": ["doppler", "dropped", "backpressure", "buffer"],
        "answer_required": ["doppler", "drop"],
        "answer_partial":  ["loggregator", "buffer", "backpressure"],
    },

    {
        "incident_id": "INC-016",
        "tier": 2,
        "question_type": "diagnosis",
        "question": "The Diego scheduler stopped receiving cell heartbeats and the routing table went stale. What was the root cause?",
        "where_filter": None,
        "retrieval_signals": ["NATS", "48000", "slow consumer", "runaway", "metrics-agent"],
        "answer_required": ["nats", "message bus"],
        "answer_partial":  ["saturated", "publisher", "rate"],
    },

    {
        "incident_id": "INC-017",
        "tier": 2,
        "question_type": "diagnosis",
        "question": "Health checks are timing out for apps on cell-012 but the apps themselves appear to be running. What does the metric data show?",
        "where_filter": None,
        "retrieval_signals": ["cell-012", "cpu_usage", "94%", "noisy neighbor"],
        "answer_required": ["cpu", "cell-012"],
        "answer_partial":  ["throttle", "neighbor", "health check"],
    },

    # ── TIER 3 ───────────────────────────────────────────────────────────────

    {
        "incident_id": "INC-018",
        "tier": 3,
        "question_type": "diagnosis",
        "question": "After a blue-green deployment swap, 502 errors appeared for about 45 seconds then stopped. What caused the intermittent failures?",
        "where_filter": None,
        "retrieval_signals": ["stale backend", "cache", "blue", "deregistered", "TTL"],
        "answer_required": ["cache", "route"],
        "answer_partial":  ["blue", "stale", "ttl", "expired"],
    },
    {
        "incident_id": "INC-018",
        "tier": 3,
        "question_type": "retrieval_probe",
        "question": "Find log entries showing requests being routed to a backend that was already deregistered.",
        "where_filter": None,
        "retrieval_signals": ["stale backend", "deregistered", "connection refused", "cache"],
        "answer_required": ["stale", "deregistered"],
        "answer_partial":  ["502", "cache", "blue"],
    },

    {
        "incident_id": "INC-019",
        "tier": 3,
        "question_type": "diagnosis",
        "question": "Two services were deployed together and both crashed at exactly the same time after 90 seconds. What startup pattern caused this?",
        "where_filter": None,
        "retrieval_signals": ["waiting for gateway", "waiting for auth", "poll", "circular"],
        "answer_required": ["circular", "dependency"],
        "answer_partial":  ["deadlock", "waiting", "health"],
    },
    {
        "incident_id": "INC-019",
        "tier": 3,
        "question_type": "remediation",
        "question": "Two microservices that depend on each other's health endpoints both fail to start. How should this be resolved architecturally?",
        "where_filter": None,
        "retrieval_signals": ["waiting for", "poll", "health check timed out"],
        "answer_required": ["circuit breaker", "startup"],
        "answer_partial":  ["lazy", "dependency", "readiness"],
    },

    {
        "incident_id": "INC-020",
        "tier": 3,
        "question_type": "diagnosis",
        "question": "The autoscaler is repeatedly scaling up and then immediately scaling back down in a regular cycle. What configuration mismatch causes this?",
        "where_filter": None,
        "retrieval_signals": ["cooldown", "Scale up", "Scale down", "oscillat", "startup"],
        "answer_required": ["cooldown", "startup"],
        "answer_partial":  ["oscillat", "thrash", "scale"],
    },

    {
        "incident_id": "INC-021",
        "tier": 3,
        "question_type": "diagnosis",
        "question": "New deployments are failing and the scheduler is refusing to place instances, but running apps are unaffected. What platform component has failed?",
        "where_filter": {"level": "ERROR"},
        "retrieval_signals": ["quorum", "Locket", "BBS", "not the active master"],
        "answer_required": ["quorum", "bbs"],
        "answer_partial":  ["locket", "scheduler", "partition"],
    },
    {
        "incident_id": "INC-021",
        "tier": 3,
        "question_type": "retrieval_probe",
        "question": "Find log evidence of a distributed consensus failure preventing new instance scheduling.",
        "where_filter": None,
        "retrieval_signals": ["quorum lost", "Locket", "BBS is not the active master", "scheduling suspended"],
        "answer_required": ["quorum", "locket"],
        "answer_partial":  ["bbs", "scheduling", "master"],
    },

    {
        "incident_id": "INC-022",
        "tier": 3,
        "question_type": "diagnosis",
        "question": "An app became completely unresponsive despite no code changes. What external dependency degraded and how did it cascade to a full outage?",
        "where_filter": None,
        "retrieval_signals": ["OAuth", "thread pool", "3800ms", "RejectedExecutionException"],
        "answer_required": ["oauth", "thread pool"],
        "answer_partial":  ["latency", "exhausted", "cascade"],
    },
    {
        "incident_id": "INC-022",
        "tier": 3,
        "question_type": "remediation",
        "question": "How should an app be protected against an external authentication provider becoming slow or unresponsive?",
        "where_filter": None,
        "retrieval_signals": ["OAuth", "thread pool", "latency", "circuit"],
        "answer_required": ["circuit breaker", "timeout"],
        "answer_partial":  ["bulkhead", "async", "fallback"],
    },

    {
        "incident_id": "INC-023",
        "tier": 3,
        "question_type": "diagnosis",
        "question": "A deployment worked previously but now consistently fails with a checksum mismatch error. The build itself succeeds. What is wrong?",
        "where_filter": None,
        "retrieval_signals": ["checksum mismatch", "partial blob", "Blob key conflict", "67%"],
        "answer_required": ["partial", "blob"],
        "answer_partial":  ["checksum", "upload", "interrupted"],
    },

    {
        "incident_id": "INC-024",
        "tier": 3,
        "question_type": "diagnosis",
        "question": "A rolling deployment has been paused for several minutes. The canary instance passes its liveness check but the deploy won't proceed. Why?",
        "where_filter": None,
        "retrieval_signals": ["readiness", "503", "cache not warmed", "liveness"],
        "answer_required": ["readiness", "503"],
        "answer_partial":  ["canary", "cache", "warmed"],
    },

    # ── PRECEDING FAILURE DETECTION (Tier 3 — special class) ─────────────────

    {
        "incident_id": "INC-025",
        "tier": 3,
        "question_type": "retrieval_probe",
        "question": "Are there any certificate expiration warnings in the logs before a TLS failure occurs?",
        "where_filter": None,
        "retrieval_signals": ["CERT WARNING", "expires in", "mTLS", "6h"],
        "answer_required": ["cert", "warn"],
        "answer_partial":  ["expire", "tls", "certificate"],
    },
    {
        "incident_id": "INC-025",
        "tier": 3,
        "question_type": "diagnosis",
        "question": "All inter-service communication failed simultaneously with TLS errors. What warning appeared in the logs hours before the outage?",
        "where_filter": None,
        "retrieval_signals": ["CERT WARNING", "expires in 6h", "certificate has expired", "mTLS"],
        "answer_required": ["certificate", "warn"],
        "answer_partial":  ["expire", "6 hour", "6h"],
    },
    {
        "incident_id": "INC-025",
        "tier": 3,
        "question_type": "remediation",
        "question": "A platform-wide TLS failure took down all inter-service communication. How should this class of incident be prevented?",
        "where_filter": None,
        "retrieval_signals": ["certificate has expired", "CERT WARNING", "mTLS"],
        "answer_required": ["automat", "alert"],
        "answer_partial":  ["renew", "rotation", "monitor"],
    },
]


# ============================================================================
# SCORING
# ============================================================================

def score_retrieval(chunks: list[dict], signals: list[str]) -> dict:
    """
    Check whether any retrieved chunk contains at least one signal phrase.
    Returns hit/miss per signal and an overall pass/fail.
    """
    combined_text = " ".join(c["text"] for c in chunks).lower()
    hits = {s: s.lower() in combined_text for s in signals}
    passed = any(hits.values())
    return {
        "passed": passed,
        "signal_hits": hits,
        "hit_count": sum(hits.values()),
        "total_signals": len(signals),
    }


def score_answer(answer: str, required: list[str], partial: list[str]) -> dict:
    """
    Grade the LLM answer:
      full_credit  — all required keywords present
      partial      — required fails but at least one partial keyword present
      miss         — nothing matches
    """
    answer_lower = answer.lower()
    req_hits = {k: k.lower() in answer_lower for k in required}
    par_hits = {k: k.lower() in answer_lower for k in partial}

    if all(req_hits.values()):
        grade = "full_credit"
    elif any(par_hits.values()):
        grade = "partial"
    else:
        grade = "miss"

    return {
        "grade": grade,
        "required_hits": req_hits,
        "partial_hits": par_hits,
    }


# ============================================================================
# RUNNER
# ============================================================================

def run_benchmark(collection_name: str) -> dict:
    collection = get_collection(collection_name)
    if collection.count() == 0:
        sys.exit(f"Collection '{collection_name}' is empty.")

    results = []
    total   = len(BENCHMARK_CASES)

    print(f"\nRunning {total} benchmark cases against '{collection_name}'")
    print(f"Model: {LLM_MODEL}  |  Embeddings: {EMBEDDING_MODEL}  |  k={N_RESULTS}\n")

    for i, case in enumerate(BENCHMARK_CASES, 1):
        inc_id = case["incident_id"]
        qtype  = case["question_type"]
        tier   = case["tier"]

        print(f"[{i:02d}/{total}] {inc_id} T{tier} {qtype:<20} ", end="", flush=True)
        t_start = time.time()

        # ── Retrieval ─────────────────────────────────────────────────────
        chunks = retrieve_chunks(
            case["question"], collection,
            EMBEDDING_MODEL, N_RESULTS, case.get("where_filter")
        )
        retrieval_score = score_retrieval(chunks, case["retrieval_signals"])

        # ── Generation ────────────────────────────────────────────────────
        prompt = build_prompt(case["question"], chunks)
        answer = generate_answer(prompt, LLM_MODEL)
        answer_score = score_answer(answer, case["answer_required"], case["answer_partial"])

        elapsed = time.time() - t_start

        r_flag = "✓" if retrieval_score["passed"]           else "✗"
        a_flag = {"full_credit": "✓", "partial": "~", "miss": "✗"}[answer_score["grade"]]
        print(f"ret:{r_flag}  ans:{a_flag}  ({elapsed:.1f}s)")

        if VERBOSE:
            print(f"  Q: {case['question'][:80]}")
            print(f"  A: {answer[:120]}")

        results.append({
            "incident_id":    inc_id,
            "tier":           tier,
            "question_type":  qtype,
            "question":       case["question"],
            "where_filter":   case.get("where_filter"),
            "answer":         answer,
            "retrieval":      retrieval_score,
            "answer_score":   answer_score,
            "elapsed_s":      round(elapsed, 2),
            "config": {
                "collection":      collection_name,
                "embedding_model": EMBEDDING_MODEL,
                "llm_model":       LLM_MODEL,
                "chunk_method":    CHUNK_METHOD,
                "n_results":       N_RESULTS,
            },
        })

    return results


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results: list[dict]):
    total = len(results)

    ret_pass  = sum(1 for r in results if r["retrieval"]["passed"])
    full      = sum(1 for r in results if r["answer_score"]["grade"] == "full_credit")
    partial   = sum(1 for r in results if r["answer_score"]["grade"] == "partial")
    miss      = sum(1 for r in results if r["answer_score"]["grade"] == "miss")

    print("\n" + "═" * 60)
    print("  BENCHMARK SUMMARY")
    print("═" * 60)
    print(f"  Total cases        : {total}")
    print(f"  Retrieval pass     : {ret_pass}/{total}  ({100*ret_pass//total}%)")
    print(f"  Answer full credit : {full}/{total}  ({100*full//total}%)")
    print(f"  Answer partial     : {partial}/{total}  ({100*partial//total}%)")
    print(f"  Answer miss        : {miss}/{total}  ({100*miss//total}%)")

    # ── By tier ──────────────────────────────────────────────────────────
    print("\n  ── By tier ─────────────────────────")
    for tier in (1, 2, 3):
        t_rows = [r for r in results if r["tier"] == tier]
        t_ret  = sum(1 for r in t_rows if r["retrieval"]["passed"])
        t_full = sum(1 for r in t_rows if r["answer_score"]["grade"] == "full_credit")
        print(f"  Tier {tier} ({len(t_rows):2} cases)  "
              f"ret:{t_ret}/{len(t_rows)}  full:{t_full}/{len(t_rows)}")

    # ── By question type ─────────────────────────────────────────────────
    print("\n  ── By question type ────────────────")
    for qtype in ("retrieval_probe", "diagnosis", "remediation"):
        q_rows = [r for r in results if r["question_type"] == qtype]
        q_ret  = sum(1 for r in q_rows if r["retrieval"]["passed"])
        q_full = sum(1 for r in q_rows if r["answer_score"]["grade"] == "full_credit")
        print(f"  {qtype:<20}  "
              f"ret:{q_ret}/{len(q_rows)}  full:{q_full}/{len(q_rows)}")

    # ── Misses worth investigating ────────────────────────────────────────
    misses = [r for r in results if r["answer_score"]["grade"] == "miss"]
    if misses:
        print("\n  ── Answer misses ───────────────────")
        for r in misses:
            print(f"  {r['incident_id']} T{r['tier']} {r['question_type']}")
            print(f"    Q: {r['question'][:70]}")
            print(f"    required: {list(r['answer_score']['required_hits'].keys())}")

    ret_fails = [r for r in results if not r["retrieval"]["passed"]]
    if ret_fails:
        print("\n  ── Retrieval failures ──────────────")
        for r in ret_fails:
            print(f"  {r['incident_id']} T{r['tier']} {r['question_type']}")
            hits = r["retrieval"]["signal_hits"]
            missed = [s for s, hit in hits.items() if not hit]
            print(f"    missed signals: {missed}")

    print("═" * 60)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    collection_name = make_collection_name(
        BASE_COLLECTION, EMBEDDING_MODEL, CHUNK_METHOD,
        CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_SUFFIX,
    )

    results = run_benchmark(collection_name)

    # Add run metadata
    output = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "collection":      collection_name,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model":       LLM_MODEL,
            "chunk_method":    CHUNK_METHOD,
            "chunk_size":      CHUNK_SIZE,
            "chunk_overlap":   CHUNK_OVERLAP,
            "n_results":       N_RESULTS,
        },
        "summary": {
            "total_cases":        len(results),
            "retrieval_pass":     sum(1 for r in results if r["retrieval"]["passed"]),
            "answer_full_credit": sum(1 for r in results if r["answer_score"]["grade"] == "full_credit"),
            "answer_partial":     sum(1 for r in results if r["answer_score"]["grade"] == "partial"),
            "answer_miss":        sum(1 for r in results if r["answer_score"]["grade"] == "miss"),
        },
        "cases": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print_summary(results)
    print(f"\n  Full results saved to {RESULTS_FILE}")
