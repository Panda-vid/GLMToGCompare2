import time

from timeit import default_timer as timer

from utils.oop import Singleton


class RateLimiter:
    def __init__(self, limit_per_sec: float) -> None:
        self._limit_per_sec = limit_per_sec
        self._interval = 1/limit_per_sec
        self._last_hit = timer()
        self._reporting_timer = timer()
        self._reporting_counter = 0
        self._reproting_interval = 100

    def hit(self):
        current_time = timer()
        time_interval = current_time - self._last_hit
        if time_interval < self._interval:
            time.sleep(self._interval - time_interval)
        self._last_hit = timer()
        self._reporting_counter += 1
        if self._reporting_counter % self._reproting_interval == 0:
            true_rate = 1/((self._last_hit - self._reporting_timer) / self._reporting_counter)
            if true_rate > self._limit_per_sec:
                print(f"Requesting faster than limit of {self._limit_per_sec} per second. Actual rate: {true_rate} per second.")
            self._reporting_timer = timer()
            self._reporting_counter = 0


class RateManager(Singleton):
    def init(self):
        self.rate_limiters = {}

    @classmethod
    def limit_rate(cls, identifier: str, rate_limit_per_sec: float):
        rate_manager = cls()
        def limit_func(func):
            if identifier not in rate_manager.rate_limiters.keys():
                rate_manager.rate_limiters[identifier] = RateLimiter(rate_limit_per_sec)

            def rate_limited(*args, **kwargs):
                rate_manager.rate_limiters[identifier].hit()
                return func(*args, **kwargs)
            return rate_limited
        return limit_func


if __name__ == "__main__":
    @RateManager.limit_rate("test", 50)
    def test_limited(integer: int):
        print(f"Hello {integer}")

    @RateManager.limit_rate("test", 50)
    def test_limited2(integer: int):
        print(f"Hello {-integer}")

    start = time.time()
    for i in range(100):
        test_limited(i+1)
        test_limited2(i+1)   
    end = time.time()
    rate_manager = RateManager()
    print(rate_manager.rate_limiters["test"]._interval)
    print((end - start) > 100 * 2 * rate_manager.rate_limiters["test"]._interval)