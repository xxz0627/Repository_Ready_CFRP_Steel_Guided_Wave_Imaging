import numpy as np
from config import Config


def _get_8_receivers(tx_x, tx_y, r):
    return [
        (tx_x - r, tx_y - r),
        (tx_x - r, tx_y),
        (tx_x - r, tx_y + r),
        (tx_x,     tx_y + r),
        (tx_x + r, tx_y + r),
        (tx_x + r, tx_y),
        (tx_x + r, tx_y - r),
        (tx_x,     tx_y - r),
    ]


def _get_16_receivers(tx_x, tx_y, r):
    base_8 = _get_8_receivers(tx_x, tx_y, r)
    half_r = r / 2.0
    midpoints = [
        (tx_x - r,     tx_y - half_r),
        (tx_x - r,     tx_y + half_r),
        (tx_x - half_r, tx_y + r),
        (tx_x + half_r, tx_y + r),
        (tx_x + r,     tx_y + half_r),
        (tx_x + r,     tx_y - half_r),
        (tx_x + half_r, tx_y - r),
        (tx_x - half_r, tx_y - r),
    ]
    receivers = []
    for i in range(8):
        receivers.append(base_8[i])
        receivers.append(midpoints[i])
    return receivers


def _is_in_specimen(x, y):
    return 0 <= x <= Config.SPECIMEN_WIDTH and 0 <= y <= Config.SPECIMEN_HEIGHT


class SensorManager:
    def __init__(self, data):
        self.data = data
        self._initialize_sensors()

    def _initialize_sensors(self):
        self.tx_positions = list(
            self.data[['Tx_X', 'Tx_Y']].itertuples(index=False, name=None)
        )
        self.tx_positions = list(dict.fromkeys(self.tx_positions))

        self.rx_positions = list(
            self.data[['Rx_X', 'Rx_Y']].itertuples(index=False, name=None)
        )
        self.rx_positions = list(dict.fromkeys(self.rx_positions))

        all_points_set = set(self.tx_positions) | set(self.rx_positions)
        self.all_sensors = list(all_points_set)
        self.num_sensors = len(self.all_sensors)

        self._build_tx_receiver_map()

    def _build_tx_receiver_map(self):
        self.tx_to_receivers = {}
        for _, row in self.data.iterrows():
            tx = (row['Tx_X'], row['Tx_Y'])
            rx = (row['Rx_X'], row['Rx_Y'])
            if tx not in self.tx_to_receivers:
                self.tx_to_receivers[tx] = []
            if rx not in self.tx_to_receivers[tx]:
                self.tx_to_receivers[tx].append(rx)

    def is_neighbor_path(self, tx, ty, rx, ry):
        tx_key = (tx, ty)
        rx_key = (rx, ry)
        if tx_key in self.tx_to_receivers:
            return rx_key in self.tx_to_receivers[tx_key]
        return False

    def _find_sensor_index(self, x, y, eps=1e-6):
        for i, (sx, sy) in enumerate(self.all_sensors):
            if abs(sx - x) < eps and abs(sy - y) < eps:
                return i
        return -1

    @staticmethod
    def generate_tx_points():
        tx_points = []
        x = Config.TX_START_X
        while x <= Config.TX_END_X:
            tx_points.append((x, Config.TX_Y))
            x += Config.TX_STEP_X
        return tx_points

    @staticmethod
    def get_receivers(tx_x, tx_y, mode=None):
        if mode is None:
            mode = Config.RECEIVER_MODE
        r = Config.RECEIVER_RADIUS
        if mode == 16:
            return _get_16_receivers(tx_x, tx_y, r)
        else:
            return _get_8_receivers(tx_x, tx_y, r)
