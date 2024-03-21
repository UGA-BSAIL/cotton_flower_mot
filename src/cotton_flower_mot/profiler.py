import atexit
from contextlib import contextmanager
import time
from typing import Iterable, NoReturn, Any

from loguru import logger
from tabulate import tabulate


class Profiler:
    """
    Handles profiling a specific operation.
    """

    def __init__(self, name: str, warmup_iters: int = 0):
        """
        Args:
            name: The name of the profiler.
            warmup_iters: Ignore the first N iterations when computing
                statistics.

        """
        self.__name = name
        self.__warmup_iters = warmup_iters

        # The total amount of time spent running the operation.
        self.__total_time = 0.0
        # The total number of times the operation was run.
        self.__num_runs = 0

    @contextmanager
    def profile(self) -> Iterable[None]:
        """
        Tracks the time spent within the context manager.

        """
        start_time = time.time()
        yield

        if self.__num_runs > self.__warmup_iters:
            self.__total_time += time.time() - start_time
        self.__num_runs += 1

    @property
    def total_time(self) -> float:
        """
        Returns:
            The total time spent running the operation.

        """
        return self.__total_time

    @property
    def time_per_iter(self) -> float:
        """
        Returns:
            The average time per iteration.

        """
        if self.__num_runs == 0:
            return 0.0
        return self.__total_time / self.__num_runs

    @property
    def num_iters(self) -> int:
        """
        Returns:
            The total number of iterations profiled.

        """
        return self.__num_runs

    @property
    def name(self) -> str:
        """
        Returns:
            The name of the profiler.

        """
        return self.__name


class ProfilingManager:
    """
    Manages multiple profiles for a program.
    """

    def __init__(self):
        # Profiles that are included internally.
        self.__profiles = {}

        # Automatically print the report on exit.
        atexit.register(self.log_report)

    def add(self, profiler: Profiler) -> None:
        """
        Explicitly adds a new profiler to the manager.

        Args:
            profiler: The profiler to add.
        """
        self.__profiles[profiler.name] = profiler

    @contextmanager
    def profile(self, name: str, *args: Any, **kwargs: Any) -> Iterable[None]:
        """
        Tracks the time spent within the context manager.

        Args:
            name: The name of the operation we are profiling.
            *args: Will be forwarded to `Profiler`.
            **kwargs: Will be forwarded to `Profiler`.

        """
        profiler = self.__profiles.get(name)
        if profiler is None:
            # Add a new profiler for this.
            profiler = Profiler(name, *args, **kwargs)
            self.add(profiler)

        with profiler.profile():
            yield

    def log_report(self) -> NoReturn:
        """
        Logs report of the profiled data.
        """
        names = [p for p in self.__profiles]
        times = [p.total_time for p in self.__profiles.values()]
        num_iters = [p.num_iters for p in self.__profiles.values()]
        cycle_times = [p.time_per_iter for p in self.__profiles.values()]
        # Avoid division by zero.
        cycle_freqs = [1 / max(t, 1e-9) for t in cycle_times]

        table = tabulate(
            {
                "Name": names,
                "Time (s)": times,
                "Num Iters": num_iters,
                "Cycle Time (s/iter)": cycle_times,
                "Cycle Freq (iter/s)": cycle_freqs,
            },
            headers="keys",
        )
        logger.info(f"Profiling report:\n{table}")
