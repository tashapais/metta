# Job Runner Dashboard Spec

Emit OpenTelemetry metrics on job state transitions. Only modify `job_routes.py`.

## Metrics

**`job.state_transition`** (Counter)

- Attributes: `from_status`, `to_status`, `job_type`, `error_type`
- Increment on each status change

**`job.stage_duration`** (Histogram, seconds)

- Attributes: `stage` (pending | dispatched | running), `job_type`
- Record duration spent in the stage that just ended

## Emit Points

| Endpoint          | Transition           | Duration to record  |
| ----------------- | -------------------- | ------------------- |
| create_jobs_batch | pending → dispatched | now - created_at    |
| create_jobs_batch | pending → failed     | now - created_at    |
| update_job        | dispatched → running | now - dispatched_at |
| update_job        | running → completed  | now - running_at    |
| update_job        | running → failed     | now - running_at    |

## Desired Charts

1. **Stage durations (p50, p90)**: time in pending/dispatched/running states
2. **Jobs by status over time**: jobs/minute entering each status
3. **Concurrent running jobs**: current count in running state
4. **Failure breakdown**: count by error_type attribute
