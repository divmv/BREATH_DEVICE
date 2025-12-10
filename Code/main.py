import bluetooth
import time
from micropython import const
import breath_analysis

_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_WRITE = const(3)

_SERVICE_UUID = bluetooth.UUID('4fafc201-1fb5-459e-8fcc-c5c9c331914b')
_CHAR_UUID = bluetooth.UUID('beb5483e-36e1-4688-b7f5-ea07361b26a8')
_CHAR_CONFIG_UUID = bluetooth.UUID('00002902-0000-1000-8000-00805f9b34fb')

class BLEDevice:
    def __init__(self, ble, name="Breath_Device"):
        self._ble = ble
        self._ble.active(True)
        self._ble.irq(self._irq)
        ((self._handle,),) = self._ble.gatts_register_services([
            (_SERVICE_UUID, [(_CHAR_UUID, 0x0002 | 0x0008 | 0x0010),]),
        ])
        self._connections = set()
        self._advertise(name)

    def _irq(self, event, data):
        if event == _IRQ_CENTRAL_CONNECT:
            conn_handle, _, _ = data
            self._connections.add(conn_handle)
        elif event == _IRQ_CENTRAL_DISCONNECT:
            conn_handle, _, _ = data
            self._connections.remove(conn_handle)
            self._advertise()
        elif event == _IRQ_GATTS_WRITE:
            conn_handle, value_handle = data
            if value_handle == self._handle:
                cmd = self._ble.gatts_read(self._handle).decode('utf-8').strip()
                if cmd == "RUN_PYTHON":
                    # Run the capture
                    msg = breath_analysis.run_analysis()
                    self.send_data(msg)

    def send_data(self, data):
        # Sends data in chunks to the phone
        for conn in self._connections:
            try:
                # Basic chunking logic
                chunk_size = 100 # BLE limit is often 20-512 bytes
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i+chunk_size]
                    self._ble.gatts_notify(conn, self._handle, chunk.encode('utf-8'))
                    time.sleep(0.02) # Small delay to prevent packet loss
            except:
                pass

    def _advertise(self, name):
        name_bytes = bytes(name, 'utf-8')
        payload = b'\x02\x01\x06' + bytearray([len(name_bytes) + 1, 0x09]) + name_bytes
        self._ble.gap_advertise(100, adv_data=payload)

ble = bluetooth.BLE()
device = BLEDevice(ble)
while True: time.sleep(1)