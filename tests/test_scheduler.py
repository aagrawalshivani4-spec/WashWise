"""
Automated Test Suite for WashWise Scheduling Engine
Covers hard constraints, decoder, GA operators, FCFS baseline, edge cases, and metrics.
"""

import os
import pytest
import pandas as pd
from scheduler import (
    Priority,
    LaundryRequest,
    ScheduledJob,
    ScheduleResult,
    GeneticAlgorithmConfig,
    time_to_minutes,
    minutes_to_time,
    time_to_slot,
    slot_to_time,
    compute_total_slots,
    prepare_request,
    generate_sample_requests,
    save_requests_to_csv,
    load_requests_from_csv,
    compute_schedule_metrics,
    run_fcfs_scheduler,
    run_genetic_algorithm,
    decode_chromosome,
    compare_schedules,
    Chromosome,
    RequestGene,
)


def test_time_conversions():
    assert time_to_minutes("08:00") == 480
    assert time_to_minutes("22:00") == 1320
    assert minutes_to_time(480) == "08:00"
    assert minutes_to_time(1320) == "22:00"

    assert time_to_slot("08:00", opening_time="08:00", granularity_mins=15) == 0
    assert time_to_slot("08:15", opening_time="08:00", granularity_mins=15) == 1
    assert time_to_slot("09:00", opening_time="08:00", granularity_mins=15) == 4

    assert slot_to_time(0, opening_time="08:00", granularity_mins=15) == "08:00"
    assert slot_to_time(4, opening_time="08:00", granularity_mins=15) == "09:00"
    assert slot_to_time(56, opening_time="08:00", granularity_mins=15) == "22:00"

    total = compute_total_slots("08:00", "22:00", 15)
    assert total == 56


def test_request_preparation():
    req = LaundryRequest(
        request_id="REQ-TEST",
        student_id="STU-001",
        student_name="Test Student",
        duration_mins=45,
        earliest_time_str="10:00",
        latest_time_str="12:00",
        priority=Priority.HIGH,
    )
    prepared = prepare_request(req, "08:00", "22:00", 15)
    assert prepared.earliest_slot == 8   # 120 mins from 8:00 = 8 slots
    assert prepared.latest_slot == 16   # 240 mins from 8:00 = 16 slots
    assert prepared.duration_slots == 3 # 45 mins / 15 = 3 slots


def test_no_machine_overlaps_in_fcfs():
    requests = generate_sample_requests(25, seed=42)
    result = run_fcfs_scheduler(requests, num_machines=4, total_slots=56)

    assert result.metrics["Scheduling Conflicts"] == 0

    # Explicit manual check for any overlap
    by_machine = {}
    for job in result.scheduled_jobs:
        by_machine.setdefault(job.machine_id, []).append(job)

    for m, jobs in by_machine.items():
        sorted_jobs = sorted(jobs, key=lambda j: j.start_slot)
        for i in range(len(sorted_jobs) - 1):
            assert sorted_jobs[i].end_slot <= sorted_jobs[i + 1].start_slot, (
                f"Overlap detected on machine {m}: Job {sorted_jobs[i].request.request_id} "
                f"ends at {sorted_jobs[i].end_slot} but next starts at {sorted_jobs[i+1].start_slot}"
            )


def test_no_machine_overlaps_in_ga():
    requests = generate_sample_requests(25, seed=42)
    config = GeneticAlgorithmConfig(population_size=20, generations=10, random_seed=42)
    result, history = run_genetic_algorithm(requests, num_machines=4, total_slots=56, config=config)

    assert result.metrics["Scheduling Conflicts"] == 0

    by_machine = {}
    for job in result.scheduled_jobs:
        by_machine.setdefault(job.machine_id, []).append(job)

    for m, jobs in by_machine.items():
        sorted_jobs = sorted(jobs, key=lambda j: j.start_slot)
        for i in range(len(sorted_jobs) - 1):
            assert sorted_jobs[i].end_slot <= sorted_jobs[i + 1].start_slot, (
                f"Overlap detected on machine {m} in GA"
            )


def test_availability_and_closing_constraints_respected():
    requests = generate_sample_requests(25, seed=42)
    # Test FCFS
    fcfs_result = run_fcfs_scheduler(requests, num_machines=4, total_slots=56)
    assert fcfs_result.metrics["Availability Violations"] == 0
    assert fcfs_result.metrics["Closing Time Violations"] == 0

    for job in fcfs_result.scheduled_jobs:
        assert job.start_slot >= job.request.earliest_slot
        assert job.end_slot <= job.request.latest_slot
        assert job.end_slot <= 56

    # Test GA
    config = GeneticAlgorithmConfig(population_size=20, generations=10, random_seed=42)
    ga_result, _ = run_genetic_algorithm(requests, num_machines=4, total_slots=56, config=config)
    assert ga_result.metrics["Availability Violations"] == 0
    assert ga_result.metrics["Closing Time Violations"] == 0

    for job in ga_result.scheduled_jobs:
        assert job.start_slot >= job.request.earliest_slot
        assert job.end_slot <= job.request.latest_slot
        assert job.end_slot <= 56


def test_impossible_window_marks_unscheduled():
    # Job has duration of 60 mins (4 slots) but availability window is only 30 mins (2 slots)
    impossible_req = LaundryRequest(
        request_id="REQ-IMP",
        student_id="STU-999",
        student_name="Impossible Student",
        duration_mins=60,
        earliest_time_str="10:00",
        latest_time_str="10:30",
        priority=Priority.NORMAL
    )
    prepared = prepare_request(impossible_req, "08:00", "22:00", 15)

    result = run_fcfs_scheduler([prepared], num_machines=2, total_slots=56)
    assert len(result.scheduled_jobs) == 0
    assert len(result.unscheduled_jobs) == 1
    assert "exceeds availability window" in result.unscheduled_jobs[0].rejection_reason


def test_empty_schedule_handling():
    # Empty request list
    result_fcfs = run_fcfs_scheduler([], num_machines=4, total_slots=56)
    assert result_fcfs.total_requests == 0
    assert result_fcfs.metrics["Machine Utilization (%)"] == 0.0
    assert result_fcfs.metrics["Average Waiting Time (mins)"] == 0.0

    result_ga, history = run_genetic_algorithm([], num_machines=4, total_slots=56,
                                               config=GeneticAlgorithmConfig(population_size=10, generations=5))
    assert result_ga.total_requests == 0
    assert result_ga.metrics["Machine Utilization (%)"] == 0.0


def test_csv_persistence_roundtrip(tmp_path):
    requests = generate_sample_requests(15, seed=123)
    file_path = str(tmp_path / "test_requests.csv")
    save_requests_to_csv(requests, file_path)

    assert os.path.exists(file_path)
    loaded = load_requests_from_csv(file_path)
    assert len(loaded) == 15
    assert loaded[0].request_id == requests[0].request_id
    assert loaded[0].duration_mins == requests[0].duration_mins
    assert loaded[0].priority == requests[0].priority


def test_ga_reproducibility():
    requests = generate_sample_requests(20, seed=42)
    cfg1 = GeneticAlgorithmConfig(population_size=20, generations=10, random_seed=99)
    cfg2 = GeneticAlgorithmConfig(population_size=20, generations=10, random_seed=99)

    res1, hist1 = run_genetic_algorithm(requests, num_machines=4, total_slots=56, config=cfg1)
    res2, hist2 = run_genetic_algorithm(requests, num_machines=4, total_slots=56, config=cfg2)

    assert res1.metrics["Scheduled Requests"] == res2.metrics["Scheduled Requests"]
    assert res1.metrics["Total Waiting Time (mins)"] == res2.metrics["Total Waiting Time (mins)"]
    assert hist1[-1]["best_fitness"] == hist2[-1]["best_fitness"]


def test_comparison_table_structure():
    requests = generate_sample_requests(15, seed=42)
    fcfs_res = run_fcfs_scheduler(requests, 4, 56)
    ga_res, _ = run_genetic_algorithm(requests, 4, 56, config=GeneticAlgorithmConfig(population_size=15, generations=5))
    cmp_df = compare_schedules(fcfs_res, ga_res)

    assert isinstance(cmp_df, pd.DataFrame)
    assert "Scheduling Metric" in cmp_df.columns
    assert "FCFS Baseline" in cmp_df.columns
    assert "Genetic Algorithm" in cmp_df.columns
    assert "Better Performer" in cmp_df.columns
