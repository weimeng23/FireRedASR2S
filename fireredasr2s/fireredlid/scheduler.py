import asyncio
import logging
from dataclasses import dataclass
from itertools import count
from time import monotonic


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BucketPolicy:
    max_seconds: float
    max_batch_size: int


@dataclass(frozen=True)
class DecodedLidInput:
    uttid: str
    wav_input: object
    duration_s: float


class QueueFullError(RuntimeError):
    pass


class SchedulerClosedError(RuntimeError):
    pass


@dataclass
class _PendingTask:
    item: DecodedLidInput
    future: asyncio.Future
    enqueued_at: float


@dataclass
class _QueueReservation:
    owner: object
    count: int
    active: bool = True


class LidBatchScheduler:
    def __init__(
        self,
        *,
        engine,
        executor,
        queue_capacity: int,
        max_batch_delay_ms: float,
        bucket_policies: tuple[BucketPolicy, ...],
    ):
        self._engine = engine
        self._executor = executor
        self._queue_capacity = queue_capacity
        self._max_batch_delay_s = max_batch_delay_ms / 1000.0
        self._bucket_policies = bucket_policies
        self._pending: list[_PendingTask] = []
        self._reserved = 0
        self._in_flight = 0
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._stopping = False
        self._fatal_error: Exception | None = None
        self._inference_ids = count()

    @property
    def pending_count(self):
        return self._reserved + len(self._pending) + self._in_flight

    @property
    def is_healthy(self):
        return (
            self._fatal_error is None
            and not self._stopping
            and self._worker is not None
            and not self._worker.done()
        )

    async def start(self):
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self):
        if self._worker is None:
            return
        self._stopping = True
        error = SchedulerClosedError("inference scheduler is stopping")
        pending, self._pending = self._pending, []
        for task in pending:
            if not task.future.done():
                task.future.set_exception(error)
        self._wake.set()
        try:
            await self._worker
        finally:
            self._worker = None

    def reserve(self, count):
        if self._stopping:
            raise SchedulerClosedError("inference scheduler is stopping")
        if count <= 0:
            raise ValueError("reservation count must be positive")
        if self.pending_count + count > self._queue_capacity:
            raise QueueFullError("inference queue is full")
        self._reserved += count
        return _QueueReservation(owner=self, count=count)

    def release(self, reservation):
        if reservation.owner is not self:
            raise ValueError("queue reservation belongs to another scheduler")
        if not reservation.active:
            return
        self._reserved -= reservation.count
        reservation.active = False

    def submit_reserved(self, reservation, items):
        if reservation.owner is not self or not reservation.active:
            raise ValueError("queue reservation is not active")
        if len(items) != reservation.count:
            raise ValueError(
                "item count does not match queue reservation"
            )
        self._validate_items(items)
        self.release(reservation)
        if self._stopping:
            raise SchedulerClosedError("inference scheduler is stopping")
        return self._enqueue(items)

    def submit_many(self, items):
        if self._stopping:
            raise SchedulerClosedError("inference scheduler is stopping")
        if not items:
            return []
        reservation = self.reserve(len(items))
        try:
            return self.submit_reserved(reservation, items)
        except BaseException:
            self.release(reservation)
            raise

    def _enqueue(self, items):
        loop = asyncio.get_running_loop()
        enqueued_at = monotonic()
        futures = [loop.create_future() for _ in items]
        for future in futures:
            future.add_done_callback(
                lambda _future: self._wake.set()
            )
        self._pending.extend(
            _PendingTask(
                item=item,
                future=future,
                enqueued_at=enqueued_at,
            )
            for item, future in zip(items, futures, strict=True)
        )
        if items:
            self._wake.set()
        return futures

    def _validate_items(self, items):
        for item in items:
            self._bucket_index(item.duration_s)

    def _bucket_index(self, duration_s):
        for index, policy in enumerate(self._bucket_policies):
            if duration_s <= policy.max_seconds:
                return index
        raise ValueError(
            f"audio duration {duration_s:g}s exceeds scheduler bucket limits"
        )

    def _matching_tasks(self, bucket_index):
        maximum = self._bucket_policies[bucket_index].max_batch_size
        return [
            task
            for task in self._pending
            if not task.future.done()
            if self._bucket_index(task.item.duration_s) == bucket_index
        ][:maximum]

    def _discard_done_tasks(self):
        self._pending = [
            task for task in self._pending if not task.future.done()
        ]

    def _remove_tasks(self, tasks):
        selected = {id(task) for task in tasks}
        self._pending = [
            task for task in self._pending if id(task) not in selected
        ]

    async def _next_batch(self):
        while True:
            self._discard_done_tasks()
            if self._stopping and not self._pending:
                return None
            if not self._pending:
                self._wake.clear()
                await self._wake.wait()
                continue
            anchor = self._pending[0]
            bucket_index = self._bucket_index(anchor.item.duration_s)
            tasks = self._matching_tasks(bucket_index)
            maximum = self._bucket_policies[bucket_index].max_batch_size
            remaining = (
                anchor.enqueued_at
                + self._max_batch_delay_s
                - monotonic()
            )
            if len(tasks) >= maximum or remaining <= 0:
                self._remove_tasks(tasks)
                return tasks
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=remaining)
            except TimeoutError:
                self._discard_done_tasks()
                tasks = self._matching_tasks(bucket_index)
                self._remove_tasks(tasks)
                return tasks

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            tasks = None
            try:
                tasks = await self._next_batch()
                if tasks is None:
                    return
                if not tasks:
                    continue
                self._in_flight += len(tasks)
                try:
                    await self._execute_tasks(loop, tasks)
                finally:
                    self._in_flight -= len(tasks)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("FireRedLID scheduler worker failed")
                self._fatal_error = error
                self._stopping = True
                failed_tasks = list(tasks or ())
                failed_tasks.extend(self._pending)
                self._pending = []
                self._set_task_errors(failed_tasks, error)
                return

    async def _execute_tasks(self, loop, tasks):
        remaining_tasks = list(tasks)
        while remaining_tasks:
            inference_ids = [
                f"lid-inference-{next(self._inference_ids)}"
                for _ in remaining_tasks
            ]
            wav_inputs = [
                task.item.wav_input for task in remaining_tasks
            ]
            try:
                results = await loop.run_in_executor(
                    self._executor,
                    self._engine.process,
                    inference_ids,
                    wav_inputs,
                )
            except FloatingPointError as error:
                bad_indices = tuple(
                    sorted(set(getattr(error, "sample_indices", ())))
                )
                if (
                    not bad_indices
                    or any(
                        not isinstance(index, int)
                        or index < 0
                        or index >= len(remaining_tasks)
                        for index in bad_indices
                    )
                ):
                    self._set_task_errors(remaining_tasks, error)
                    return
                bad_index_set = set(bad_indices)
                bad_tasks = [
                    task
                    for index, task in enumerate(remaining_tasks)
                    if index in bad_index_set
                ]
                next_tasks = [
                    task
                    for index, task in enumerate(remaining_tasks)
                    if index not in bad_index_set
                ]
                self._set_task_errors(bad_tasks, error)
                if len(next_tasks) >= len(remaining_tasks):
                    self._set_task_errors(remaining_tasks, error)
                    return
                remaining_tasks = next_tasks
                continue
            except Exception as error:
                self._set_task_errors(remaining_tasks, error)
                return

            task_by_id = dict(
                zip(inference_ids, remaining_tasks, strict=True)
            )
            returned_ids = set()
            for result in results:
                if not isinstance(result, dict):
                    continue
                inference_id = result.get("uttid")
                task = task_by_id.get(inference_id)
                if task is None or inference_id in returned_ids:
                    continue
                returned_ids.add(inference_id)
                if not task.future.done():
                    restored = dict(result)
                    restored["uttid"] = task.item.uttid
                    task.future.set_result(restored)

            for inference_id, task in task_by_id.items():
                if (
                    inference_id not in returned_ids
                    and not task.future.done()
                ):
                    task.future.set_exception(
                        RuntimeError(
                            "runtime missing result for "
                            f"uttid {task.item.uttid!r}"
                        )
                    )
            return

    @staticmethod
    def _set_task_errors(tasks, error):
        for task in tasks:
            if not task.future.done():
                task.future.set_exception(error)
