#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ----------------- Sensor Pins -----------------
const int VT_B_PIN = 0;  // adjust to your actual analog-capable pins
const int VL_R_PIN = 1;
const int VSUM_PIN = 2;

const int   ADC_RESOLUTION   = 4095;
const float ADC_VOLTAGE_MAX  = 3.3f;

// Target sampling: 200 Hz
const uint32_t FS_HZ = 700;
const uint32_t DT_US = 1000000UL / FS_HZ;

// ----------------- BLE UUIDs -----------------
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// ----------------- BLE Globals -----------------
BLEServer*         pServer         = nullptr;
BLECharacteristic* pCharacteristic = nullptr;

bool deviceConnected    = false;
bool oldDeviceConnected = false;

// ----------------- Callbacks -----------------
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("BLE Client Connected!");
  }

  // void onDisconnect(BLEServer* pServer) override {
  //   deviceConnected = false;
  //   Serial.println("BLE Client Disconnected.");
  //   // We'll restart advertising in loop()
  // }
};

void setup() {
  Serial.begin(115200);
  //delay(100);

  Serial.println();
  Serial.println("Booting... Breath_Device BLE with per-sample CSV");

  analogSetAttenuation(ADC_11db);

  // ---- BLE init ----
  BLEDevice::init("Breath_Device");
  BLEDevice::setMTU(128);  // ask for a larger MTU (central may negotiate differently)

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );

  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();

  BLEAdvertising* pAdvertising = pServer->getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  pAdvertising->start();

  Serial.println("BLE Advertising started. Scan for 'Breath_Device'!");
}

void loop() {
  static uint32_t next_t   = micros();
  static uint32_t sampleIdx = 0;

  // 1) Handle advertising restart after disconnect
  if (!deviceConnected && oldDeviceConnected) {
    // we just disconnected
    delay(500);
    pServer->startAdvertising();
    Serial.println("Restarted advertising after disconnect.");
    oldDeviceConnected = deviceConnected;
  }

  if (deviceConnected && !oldDeviceConnected) {
    // just connected
    oldDeviceConnected = deviceConnected;
  }

  // 2) Timed sampling at 200 Hz
  uint32_t now = micros();
  if ((int32_t)(now - next_t) < 0) {
    return;  // not yet time for next sample
  }
  next_t += DT_US;

  // --- Read sensors ---
  int      r_vtb  = analogRead(VT_B_PIN);
  int      r_vlr  = analogRead(VL_R_PIN);
  int      r_vsum = analogRead(VSUM_PIN);
  uint32_t t_us   = micros();

  // --- Build one CSV row ---
  // Format: sample,vtb_raw,vlr_raw,vsum_raw,us\n
  char line[64];
  int len = snprintf(line, sizeof(line), "%u,%d,%d,%d,%u\n",
                     sampleIdx++, r_vtb, r_vlr, r_vsum, t_us);

  if (len <= 0) {
    return;  // formatting failed (shouldn't happen)
  }

  // ---- Send CSV row to Serial ----
  Serial.write((const uint8_t*)line, len);

  // ---- Send the SAME CSV row over BLE ----
  if (deviceConnected && pCharacteristic != nullptr) {
    pCharacteristic->setValue((uint8_t*)line, len);
    pCharacteristic->notify();
    // Optional debug:
    // Serial.print("BLE row: ");
    // Serial.write((const uint8_t*)line, len);
  }
}
