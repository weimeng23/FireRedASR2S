import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fireredasr2s.fireredlid.scheduler import (
    BucketPolicy,
    DecodedLidInput,
    LidBatchScheduler,
    QueueFullError,
    SchedulerClosedError,
)


class RecordingEngine:
    def __init__(self):
        self.calls = []
        self.wav_inputs = []
        self.thread_ids = []

    def process(self, uttids, wav_inputs):
        self.calls.append(list(uttids))
        self.wav_inputs.append(list(wav_inputs))
        self.thread_ids.append(threading.get_ident())
        return [
            {"uttid": uttid, "lang": "en", "confidence": 0.9}
            for uttid in uttids
        ]


def decoded(uttid, duration_s):
    return DecodedLidInput(
        uttid=uttid,
        wav_input=(16000, np.zeros(int(duration_s * 16000), dtype=np.int16)),
        duration_s=duration_s,
    )


def test_scheduler_combines_independent_submissions_on_gpu_thread():
    async def scenario():
        engine = RecordingEngine()
        main_thread = threading.get_ident()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=20,
                bucket_policies=(BucketPolicy(60, 4),),
            )
            await scheduler.start()
            futures = [
                scheduler.submit_many([decoded("one", 1)])[0],
                scheduler.submit_many([decoded("two", 1)])[0],
                scheduler.submit_many([decoded("three", 1)])[0],
            ]
            results = await asyncio.gather(*futures)
            await scheduler.stop()
        return engine, main_thread, results

    engine, main_thread, results = asyncio.run(scenario())

    assert len(engine.calls) == 1
    assert len(engine.calls[0]) == 3
    assert len(set(engine.calls[0])) == 3
    assert engine.thread_ids[0] != main_thread
    assert [result["uttid"] for result in results] == [
        "one",
        "two",
        "three",
    ]


def test_scheduler_does_not_mix_duration_buckets():
    async def scenario():
        engine = RecordingEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=1,
                bucket_policies=(
                    BucketPolicy(5, 4),
                    BucketPolicy(60, 2),
                ),
            )
            await scheduler.start()
            futures = scheduler.submit_many(
                [
                    decoded("short-1", 1),
                    decoded("long", 20),
                    decoded("short-2", 2),
                ]
            )
            await asyncio.gather(*futures)
            await scheduler.stop()
        return engine.wav_inputs

    batches = asyncio.run(scenario())

    assert [
        [waveform.size for _, waveform in batch]
        for batch in batches
    ] == [
        [16000, 32000],
        [320000],
    ]


def test_scheduler_rejects_whole_submission_when_queue_has_no_capacity():
    async def scenario():
        engine = RecordingEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=2,
                max_batch_delay_ms=100,
                bucket_policies=(BucketPolicy(60, 2),),
            )
            with pytest.raises(QueueFullError):
                scheduler.submit_many(
                    [
                        decoded("one", 1),
                        decoded("two", 1),
                        decoded("three", 1),
                    ]
                )
            assert scheduler.pending_count == 0

    asyncio.run(scenario())


def test_scheduler_continues_after_a_physical_batch_fails():
    class FlakyEngine(RecordingEngine):
        def process(self, uttids, wav_inputs):
            self.calls.append(list(uttids))
            if len(self.calls) == 1:
                raise RuntimeError("injected inference failure")
            return [
                {"uttid": uttid, "lang": "en", "confidence": 0.9}
                for uttid in uttids
            ]

    async def scenario():
        engine = FlakyEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 1),),
            )
            await scheduler.start()
            first = scheduler.submit_many([decoded("first", 1)])[0]
            with pytest.raises(RuntimeError, match="injected"):
                await first
            second = scheduler.submit_many([decoded("second", 1)])[0]
            result = await second
            await scheduler.stop()
        return engine.calls, result

    calls, result = asyncio.run(scenario())

    assert [len(call) for call in calls] == [1, 1]
    assert result["uttid"] == "second"


def test_scheduler_matches_partial_results_by_internal_id():
    class PartialEngine(RecordingEngine):
        def process(self, uttids, wav_inputs):
            return [
                {
                    "uttid": uttids[1],
                    "lang": "en",
                    "confidence": 0.9,
                }
            ]

    async def scenario():
        engine = PartialEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 2),),
            )
            await scheduler.start()
            first, second = scheduler.submit_many(
                [
                    decoded("duplicate-user-id", 1),
                    decoded("duplicate-user-id", 1),
                ]
            )
            with pytest.raises(RuntimeError, match="missing result"):
                await first
            second_result = await second
            await scheduler.stop()
        return second_result

    result = asyncio.run(scenario())

    assert result["uttid"] == "duplicate-user-id"


def test_scheduler_retries_good_items_once_using_reported_bad_indices():
    class PoisonEngine(RecordingEngine):
        def process(self, uttids, wav_inputs):
            self.calls.append(list(uttids))
            if any(wav_input == "poison" for wav_input in wav_inputs):
                error = FloatingPointError("non-finite confidence")
                error.sample_indices = tuple(
                    index
                    for index, wav_input in enumerate(wav_inputs)
                    if wav_input == "poison"
                )
                raise error
            return [
                {
                    "uttid": uttid,
                    "lang": "en",
                    "confidence": 0.9,
                }
                for uttid in uttids
            ]

    async def scenario():
        engine = PoisonEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 3),),
            )
            await scheduler.start()
            items = [
                decoded("good-1", 1),
                decoded("bad", 1),
                decoded("good-2", 1),
            ]
            items[1] = DecodedLidInput(
                uttid="bad",
                wav_input="poison",
                duration_s=1,
            )
            first, bad, second = scheduler.submit_many(items)
            first_result = await first
            with pytest.raises(FloatingPointError, match="non-finite"):
                await bad
            second_result = await second
            await scheduler.stop()
        return engine.calls, first_result, second_result

    calls, first, second = asyncio.run(scenario())

    assert [len(call) for call in calls] == [3, 2]
    assert first["uttid"] == "good-1"
    assert second["uttid"] == "good-2"


def test_scheduler_does_not_retry_unlocated_floating_point_failure():
    class BrokenEngine(RecordingEngine):
        def process(self, uttids, wav_inputs):
            self.calls.append(list(uttids))
            raise FloatingPointError("shared numerical failure")

    async def scenario():
        engine = BrokenEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 2),),
            )
            await scheduler.start()
            futures = scheduler.submit_many(
                [decoded("one", 1), decoded("two", 1)]
            )
            outcomes = await asyncio.gather(
                *futures,
                return_exceptions=True,
            )
            await scheduler.stop()
        return engine.calls, outcomes

    calls, outcomes = asyncio.run(scenario())

    assert len(calls) == 1
    assert all(isinstance(value, FloatingPointError) for value in outcomes)


def test_scheduler_keeps_shrinking_when_bad_indices_arrive_in_stages():
    class FirstPoisonOnlyEngine(RecordingEngine):
        def process(self, uttids, wav_inputs):
            self.calls.append(list(uttids))
            for index, wav_input in enumerate(wav_inputs):
                if wav_input == "poison":
                    error = FloatingPointError("non-finite confidence")
                    error.sample_indices = (index,)
                    raise error
            return [
                {
                    "uttid": uttid,
                    "lang": "en",
                    "confidence": 0.9,
                }
                for uttid in uttids
            ]

    async def scenario():
        engine = FirstPoisonOnlyEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 5),),
            )
            await scheduler.start()
            items = [
                decoded("good-1", 1),
                DecodedLidInput("bad-1", "poison", 1),
                decoded("good-2", 1),
                DecodedLidInput("bad-2", "poison", 1),
                decoded("good-3", 1),
            ]
            outcomes = await asyncio.gather(
                *scheduler.submit_many(items),
                return_exceptions=True,
            )
            await scheduler.stop()
        return engine.calls, outcomes

    calls, outcomes = asyncio.run(scenario())

    assert [len(call) for call in calls] == [5, 4, 3]
    assert [
        outcome["uttid"]
        for outcome in (outcomes[0], outcomes[2], outcomes[4])
    ] == ["good-1", "good-2", "good-3"]
    assert isinstance(outcomes[1], FloatingPointError)
    assert isinstance(outcomes[3], FloatingPointError)


def test_scheduler_does_not_retry_when_every_item_is_reported_bad():
    class AllPoisonEngine(RecordingEngine):
        def process(self, uttids, wav_inputs):
            self.calls.append(list(uttids))
            error = FloatingPointError("non-finite confidence")
            error.sample_indices = tuple(range(len(uttids)))
            raise error

    async def scenario():
        engine = AllPoisonEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 3),),
            )
            await scheduler.start()
            outcomes = await asyncio.gather(
                *scheduler.submit_many(
                    [
                        decoded("bad-1", 1),
                        decoded("bad-2", 1),
                        decoded("bad-3", 1),
                    ]
                ),
                return_exceptions=True,
            )
            await scheduler.stop()
        return engine.calls, outcomes

    calls, outcomes = asyncio.run(scenario())

    assert [len(call) for call in calls] == [3]
    assert all(isinstance(outcome, FloatingPointError) for outcome in outcomes)


def test_scheduler_fails_closed_when_worker_loop_raises():
    class BrokenScheduler(LidBatchScheduler):
        async def _next_batch(self):
            await self._wake.wait()
            raise ValueError("invalid duration bucket")

    async def scenario():
        engine = RecordingEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = BrokenScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 1),),
            )
            await scheduler.start()
            future = scheduler.submit_many([decoded("bad", 1)])[0]
            with pytest.raises(ValueError, match="duration bucket"):
                await asyncio.wait_for(future, timeout=0.1)
            assert scheduler.is_healthy is False
            with pytest.raises(SchedulerClosedError):
                scheduler.submit_many([decoded("later", 1)])
            await scheduler.stop()

    asyncio.run(scenario())


def test_scheduler_rejects_invalid_duration_before_worker_loop():
    async def scenario():
        engine = RecordingEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 1),),
            )
            await scheduler.start()
            with pytest.raises(ValueError, match="bucket limits"):
                scheduler.submit_many([decoded("too-long", 61)])
            assert scheduler.is_healthy is True
            result = await scheduler.submit_many([decoded("valid", 1)])[0]
            await scheduler.stop()
        return result

    result = asyncio.run(scenario())

    assert result["uttid"] == "valid"


def test_scheduler_does_not_execute_an_empty_batch():
    class EmptyBatchScheduler(LidBatchScheduler):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._batches = iter(([], None))

        async def _next_batch(self):
            return next(self._batches)

    async def scenario():
        engine = RecordingEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = EmptyBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 1),),
            )
            await scheduler.start()
            await scheduler._worker
        return engine.calls

    assert asyncio.run(scenario()) == []


def test_scheduler_skips_cancelled_tasks_before_gpu_inference():
    async def scenario():
        engine = RecordingEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=512,
                max_batch_delay_ms=100,
                bucket_policies=(BucketPolicy(60, 2),),
            )
            await scheduler.start()
            cancelled, live = scheduler.submit_many(
                [decoded("cancelled", 1), decoded("live", 1)]
            )
            cancelled.cancel()
            result = await asyncio.wait_for(live, timeout=0.2)
            await scheduler.stop()
        return engine.calls, result

    calls, result = asyncio.run(scenario())

    assert [len(call) for call in calls] == [1]
    assert result["uttid"] == "live"


def test_release_rejects_foreign_reservation_even_after_release():
    async def scenario():
        engine = RecordingEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=2,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 1),),
            )
            second = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=2,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 1),),
            )
            reservation = first.reserve(1)
            first.release(reservation)
            with pytest.raises(ValueError, match="another scheduler"):
                second.release(reservation)

    asyncio.run(scenario())


def test_queue_capacity_counts_in_flight_gpu_tasks():
    started = threading.Event()
    release = threading.Event()

    class BlockingEngine(RecordingEngine):
        def process(self, uttids, wav_inputs):
            started.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release inference")
            return super().process(uttids, wav_inputs)

    async def scenario():
        engine = BlockingEngine()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scheduler = LidBatchScheduler(
                engine=engine,
                executor=executor,
                queue_capacity=1,
                max_batch_delay_ms=0,
                bucket_policies=(BucketPolicy(60, 1),),
            )
            await scheduler.start()
            first = scheduler.submit_many([decoded("first", 1)])[0]
            await asyncio.to_thread(started.wait, 1)
            assert scheduler.pending_count == 1
            with pytest.raises(QueueFullError):
                scheduler.reserve(1)
            release.set()
            await first
            await scheduler.stop()

    asyncio.run(scenario())
