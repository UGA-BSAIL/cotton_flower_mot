import heapq


class RunningMedian:
    """
    Simple implementation of a running median algorithm.
    """

    def __init__(self):
        # Initialize heaps.
        self.__min_heap = []
        self.__max_heap = []

    def add(self, value: float) -> None:
        """
        Adds a new value to the running median calculation.

        Args:
            value: The value to add.

        """
        if len(self.__max_heap) == 0:
            # First item
            self.__max_heap.append(-value)
            return

        current_max = -self.__max_heap[0]
        if value < current_max:
            heapq.heappush(self.__max_heap, -value)
        else:
            heapq.heappush(self.__min_heap, value)

        # Balance the heaps.
        if len(self.__max_heap) > len(self.__min_heap) + 1:
            heapq.heappush(self.__min_heap, -heapq.heappop(self.__max_heap))
        elif len(self.__min_heap) > len(self.__max_heap) + 1:
            heapq.heappush(self.__max_heap, -heapq.heappop(self.__min_heap))

    def median(self) -> float:
        """
        Returns:
            The current running median.

        """
        if len(self.__min_heap) == len(self.__max_heap):
            return (self.__min_heap[0] - self.__max_heap[0]) / 2
        elif len(self.__min_heap) > len(self.__max_heap):
            return self.__min_heap[0]
        else:
            return -self.__max_heap[0]
