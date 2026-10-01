from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RouteDecision:
    track: str
    mean: float
    std: float


class AdaptiveRouter:
    def __init__(self, config: Any):
        self.abstain_mean = config.ROUTER_ABSTAIN_MEAN_THRESHOLD
        self.abstain_std = config.ROUTER_ABSTAIN_STD_THRESHOLD
        self.fast_mean = config.ROUTER_FAST_MEAN_THRESHOLD

    def route(self, mean: float, std: float) -> RouteDecision:
        if mean < self.abstain_mean or std > self.abstain_std:
            track = "Abstain"
        elif mean >= self.fast_mean:
            track = "Fast"
        else:
            track = "Corrective"
        return RouteDecision(track=track, mean=float(mean), std=float(std))
