from __future__ import annotations
from dataclasses import dataclass
import math
from human_tracking.observation import HumanObservation, Landmark, LandmarkSet
from human_tracking.temporal_tracking import ChannelTrackingResult, HumanTrackingResult

@dataclass(frozen=True)
class LandmarkSmoothingConfig:
    time_constant_s: float = 0.10

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_constant_s):
            raise ValueError("time_constant_s must be finite.")
        if self.time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be greater than zero.")

@dataclass(frozen=True)
class SmoothedChannelLandmarks:
    image_landmarks: LandmarkSet | None
    world_landmarks: LandmarkSet | None

@dataclass(frozen=True)
class SmoothedHumanLandmarks:
    frame_sequence_id: int
    measurement_timestamp_s: float
    body: SmoothedChannelLandmarks
    left_hand: SmoothedChannelLandmarks
    right_hand: SmoothedChannelLandmarks

class _LandmarkSetFilter:
    def __init__(self, config: LandmarkSmoothingConfig):
        self.config = config
        self._previous: LandmarkSet | None = None
        self._previous_timestamp_s: float | None = None

    def reset(self) -> None:
        self._previous = None
        self._previous_timestamp_s = None

    def update(self, landmark_set: LandmarkSet, timestamp_s: float) -> LandmarkSet:
        if not math.isfinite(timestamp_s):
            raise ValueError("Smoothing timestamp must be finite.")

        if self._previous is None:
            self._previous = landmark_set
            self._previous_timestamp_s = timestamp_s
            return landmark_set

        assert self._previous_timestamp_s is not None
        dt_s = timestamp_s - self._previous_timestamp_s

        if dt_s <= 0.0:
            raise ValueError("Smoothing timestamps must be strictly increasing.")

        previous = self._previous

        if previous.coordinate_space != landmark_set.coordinate_space:
            raise ValueError("Coordinate space changed inside one smoothing stream.")

        if len(previous.landmarks) != len(landmark_set.landmarks):
            raise ValueError("Landmark count changed inside one smoothing stream.")

        alpha = 1.0 - math.exp(-dt_s / self.config.time_constant_s)
        smoothed_landmarks = []

        for previous_landmark, current_landmark in zip(previous.landmarks, landmark_set.landmarks):
            if previous_landmark.index != current_landmark.index or previous_landmark.name != current_landmark.name:
                raise ValueError("Landmark identity changed inside one smoothing stream.")

            smoothed_landmarks.append(
                Landmark(
                    index = current_landmark.index,
                    name = current_landmark.name,
                    x = previous_landmark.x + alpha * (current_landmark.x - previous_landmark.x),
                    y = previous_landmark.y + alpha * (current_landmark.y - previous_landmark.y),
                    z = previous_landmark.z + alpha * (current_landmark.z - previous_landmark.z),
                    visibility = current_landmark.visibility,
                    presence = current_landmark.presence
                )
            )

        result = LandmarkSet(
            coordinate_space = landmark_set.coordinate_space,
            landmarks = tuple(smoothed_landmarks),
        )
        self._previous = result
        self._previous_timestamp_s = timestamp_s
        return result


class HumanLandmarkSmoother:
    def __init__(self, config: LandmarkSmoothingConfig):
        self.config = config
        self._body_image = _LandmarkSetFilter(config)
        self._body_world = _LandmarkSetFilter(config)
        self._left_image = _LandmarkSetFilter(config)
        self._left_world = _LandmarkSetFilter(config)
        self._right_image = _LandmarkSetFilter(config)
        self._right_world = _LandmarkSetFilter(config)
        self._last_sequence_id: int | None = None
        self._last_timestamp_s: float | None = None

    def update(self, observation: HumanObservation, tracking: HumanTrackingResult) -> SmoothedHumanLandmarks:
        self._validate_inputs(observation, tracking)
        body = self._update_body(observation, tracking)
        left_hand = self._update_hand(
            hands = observation.left_hands,
            channel = tracking.left_hand,
            image_filter = self._left_image,
            world_filter = self._left_world,
            side_name = "left",
            timestamp_s = observation.measurement_timestamp_s
        )

        right_hand = self._update_hand(
            hands = observation.right_hands,
            channel = tracking.right_hand,
            image_filter = self._right_image,
            world_filter = self._right_world,
            side_name = "right",
            timestamp_s = observation.measurement_timestamp_s
        )

        self._last_sequence_id = observation.frame_sequence_id
        self._last_timestamp_s = observation.measurement_timestamp_s

        return SmoothedHumanLandmarks(
            frame_sequence_id = observation.frame_sequence_id,
            measurement_timestamp_s = observation.measurement_timestamp_s,
            body = body,
            left_hand = left_hand,
            right_hand = right_hand
        )

    def _validate_inputs(self, observation: HumanObservation, tracking: HumanTrackingResult) -> None:
        if observation.frame_sequence_id != tracking.frame_sequence_id:
            raise ValueError("Observation and tracking sequence IDs do not match.")
        if observation.measurement_timestamp_s != tracking.measurement_timestamp_s:
            raise ValueError("Observation and tracking timestamps do not match.")
        if self._last_sequence_id is not None:
            if observation.frame_sequence_id <= self._last_sequence_id:
                raise ValueError("Smoother sequence IDs must be strictly increasing.")
        if self._last_timestamp_s is not None:
            if observation.measurement_timestamp_s <= self._last_timestamp_s:
                raise ValueError("Smoother timestamps must be strictly increasing.")

    def _update_body(self, observation: HumanObservation, tracking: HumanTrackingResult) -> SmoothedChannelLandmarks:
        if not tracking.body.valid_for_control:
            self._body_image.reset()
            self._body_world.reset()

            return SmoothedChannelLandmarks(
                image_landmarks = None,
                world_landmarks = None,
            )

        if observation.body_image_landmarks is None:
            raise ValueError("Body is control-valid but image landmarks are missing.")

        timestamp_s = observation.measurement_timestamp_s
        image_landmarks = self._body_image.update(observation.body_image_landmarks, timestamp_s)

        if observation.body_world_landmarks is None:
            self._body_world.reset()
            world_landmarks = None
        else:
            world_landmarks = self._body_world.update(observation.body_world_landmarks, timestamp_s)

        return SmoothedChannelLandmarks(
            image_landmarks = image_landmarks,
            world_landmarks = world_landmarks,
        )

    @staticmethod
    def _update_hand(
        hands,
        channel: ChannelTrackingResult,
        image_filter: _LandmarkSetFilter,
        world_filter: _LandmarkSetFilter,
        side_name: str,
        timestamp_s: float,
    ) -> SmoothedChannelLandmarks:
        if not channel.valid_for_control:
            image_filter.reset()
            world_filter.reset()

            return SmoothedChannelLandmarks(
                image_landmarks = None,
                world_landmarks = None,
            )

        if len(hands) != 1:
            raise ValueError(f"{side_name} hand is control-valid but does not contain exactly one detection.")

        hand = hands[0]
        image_landmarks = image_filter.update(hand.image_landmarks, timestamp_s)

        if hand.world_landmarks is None:
            world_filter.reset()
            world_landmarks = None
        else:
            world_landmarks = world_filter.update(hand.world_landmarks, timestamp_s)

        return SmoothedChannelLandmarks(
            image_landmarks = image_landmarks,
            world_landmarks = world_landmarks,
        )