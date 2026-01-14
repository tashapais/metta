from collections.abc import Iterable
from datetime import UTC, datetime
from functools import lru_cache
from typing import Optional

from opentelemetry import metrics as otel_metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from pydantic_settings import BaseSettings
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from metta.app_backend.models.job_request import JobRequest, JobStatus, JobType

METRICS_SERVICE_NAME = "observatory-backend"


class MetricsSettings(BaseSettings):
    OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: Optional[str] = None


@lru_cache
def get_metrics_settings() -> MetricsSettings:
    return MetricsSettings()


def _metrics_enabled(settings: MetricsSettings) -> bool:
    return bool(settings.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT)


class JobMetrics:
    def __init__(self) -> None:
        settings = get_metrics_settings()
        if _metrics_enabled(settings):
            resource = Resource.create({"service.name": METRICS_SERVICE_NAME})
            reader = PeriodicExportingMetricReader(OTLPMetricExporter())
            provider = MeterProvider(resource=resource, metric_readers=[reader])
            otel_metrics.set_meter_provider(provider)

        meter = otel_metrics.get_meter(__name__)
        self._state_transition_counter = meter.create_counter(
            "job.state_transition",
            description="Job status transitions",
            unit="1",
        )
        self._stage_duration_histogram = meter.create_histogram(
            "job.stage_duration",
            description="Time spent in job lifecycle stages",
            unit="s",
        )
        self._running_counts: dict[str, int] = {}
        meter.create_observable_gauge(
            "job.running_count",
            callbacks=[self._observe_running_count],
            description="Current number of running jobs",
            unit="1",
        )

    def _observe_running_count(self, options: CallbackOptions) -> Iterable[Observation]:
        del options
        # Counts update on job transitions, so values can be stale between updates.
        snapshot = dict(self._running_counts)
        return [Observation(count, {"job_type": job_type}) for job_type, count in snapshot.items()]

    def _classify_error(self, error: Optional[str]) -> str:
        if not error:
            return "none"
        error_lower = error.lower()
        if "timeout" in error_lower or "deadline" in error_lower:
            return "timeout"
        if "oom" in error_lower or "out of memory" in error_lower:
            return "oom"
        return "unknown"

    def _record_stage_duration(
        self,
        stage: str,
        start_at: Optional[datetime],
        end_at: datetime,
        job_type: JobType,
    ) -> None:
        if start_at is None:
            return
        if start_at.tzinfo is None:
            start_at = start_at.replace(tzinfo=UTC)
        if end_at.tzinfo is None:
            end_at = end_at.replace(tzinfo=UTC)
        duration_seconds = (end_at - start_at).total_seconds()
        if duration_seconds < 0:
            return
        self._stage_duration_histogram.record(
            duration_seconds,
            attributes={"stage": stage, "job_type": job_type.value},
        )

    def record_transition(
        self,
        from_status: JobStatus,
        to_status: JobStatus,
        job: JobRequest,
        transition_time: datetime,
        error_type: Optional[str],
    ) -> None:
        self._state_transition_counter.add(
            1,
            attributes={
                "from_status": from_status.value,
                "to_status": to_status.value,
                "job_type": job.job_type.value,
                "error_type": self._classify_error(error_type),
            },
        )
        if from_status == JobStatus.pending:
            self._record_stage_duration("pending", job.created_at, transition_time, job.job_type)
        elif from_status == JobStatus.dispatched:
            # Reconciliation can mark dispatched -> completed/failed without a running phase.
            self._record_stage_duration("dispatched", job.dispatched_at, transition_time, job.job_type)
        elif from_status == JobStatus.running:
            self._record_stage_duration("running", job.running_at, transition_time, job.job_type)

    async def update_running_counts(self, session: AsyncSession, job_types: set[JobType]) -> None:
        if not job_types:
            return
        result = await session.execute(
            select(JobRequest.job_type, func.count())
            .where(JobRequest.status == JobStatus.running)
            .where(col(JobRequest.job_type).in_(job_types))
            .group_by(JobRequest.job_type)
        )
        counts = {job_type: count for job_type, count in result.all()}
        for job_type in job_types:
            self._running_counts[job_type.value] = counts.get(job_type, 0)


@lru_cache
def get_job_metrics() -> JobMetrics:
    return JobMetrics()
