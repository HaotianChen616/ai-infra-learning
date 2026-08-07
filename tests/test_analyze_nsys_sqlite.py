import sqlite3
import tempfile
import unittest
from pathlib import Path

from labs.analyze_nsys_sqlite import analyze


class AnalyzeNsysSqliteTest(unittest.TestCase):
    def test_exact_wait_sched_and_submission_decomposition(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "trace.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE StringIds (id INTEGER, value TEXT);
                INSERT INTO StringIds VALUES
                    (1, 'cudaEventSynchronize'),
                    (2, 'cudaGraphLaunch'),
                    (3, 'PyEval_EvalFrameDefault'),
                    (4, 'libpython.so'),
                    (5, 'get_num_common_prefix_blocks'),
                    (6, 'vllm/v1/core/sched/scheduler.py');

                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                    start INTEGER, end INTEGER, globalTid INTEGER,
                    correlationId INTEGER, nameId INTEGER
                );
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                    (100, 300, 77, 1, 1),
                    (400, 450, 77, 2, 2);

                CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (
                    start INTEGER, end INTEGER, contextId INTEGER,
                    correlationId INTEGER, eventId INTEGER, eventSyncId INTEGER
                );
                INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
                    VALUES (100, 300, 4, 1, 8, 9);

                CREATE TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT (
                    timestamp INTEGER, contextId INTEGER,
                    eventId INTEGER, eventSyncId INTEGER
                );
                INSERT INTO CUPTI_ACTIVITY_KIND_CUDA_EVENT VALUES (250, 4, 8, 9);

                CREATE TABLE SCHED_EVENTS (
                    start INTEGER, isSchedIn INTEGER, globalTid INTEGER,
                    threadState INTEGER, threadBlock INTEGER
                );
                INSERT INTO SCHED_EVENTS VALUES
                    (0, 1, 77, NULL, NULL),
                    (150, 0, 77, 1, 1),
                    (260, 1, 77, NULL, NULL),
                    (500, 0, 77, 1, 1);

                CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
                    start INTEGER, end INTEGER, globalPid INTEGER,
                    deviceId INTEGER, contextId INTEGER, streamId INTEGER,
                    correlationId INTEGER
                );
                INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES
                    (300, 390, 1, 0, 4, 3, 90),
                    (500, 600, 1, 0, 4, 3, 2);

                CREATE TABLE COMPOSITE_EVENTS (
                    id INTEGER, globalTid INTEGER, cpuCycles INTEGER
                );
                INSERT INTO COMPOSITE_EVENTS VALUES (10, 77, 1), (11, 77, 1);
                CREATE TABLE SAMPLING_CALLCHAINS (
                    id INTEGER, symbol INTEGER, module INTEGER, stackDepth INTEGER
                );
                INSERT INTO SAMPLING_CALLCHAINS VALUES
                    (10, 3, 4, 0), (11, 5, 6, 0);
                """
            )
            connection.commit()
            connection.close()

            payload = analyze(path)

        sync = payload["sync"]["by_api"][0]
        self.assertEqual(sync["api_wall"]["total_ms"], 0.0002)
        self.assertEqual(sync["device_not_ready"]["total_ms"], 0.00015)
        self.assertEqual(sync["post_event_host_tail"]["total_ms"], 0.00005)
        self.assertEqual(sync["sampled_on_cpu_inside_api"]["total_ms"], 0.00009)
        self.assertEqual(sync["sampled_off_cpu_inside_api"]["total_ms"], 0.00011)

        submit = payload["submissions"]["by_api"][0]
        self.assertEqual(submit["critical_launch_bubble"]["total_ms"], 0.00005)
        categories = {
            row["category"] for row in payload["cpu_self_samples"]["by_category"]
        }
        self.assertEqual(categories, {"python_interpreter", "vllm_scheduler"})


if __name__ == "__main__":
    unittest.main()
