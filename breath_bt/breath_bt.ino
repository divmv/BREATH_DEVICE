#include <WiFi.h>
#include <esp_wpa2.h>       // For WPA2 Enterprise Wi-Fi functions
#include <esp_eap_client.h> // For EAP client functions (good practice)
#include <esp_wifi.h>       // For low-level Wi-Fi types and functions

// Include BLE libraries
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h> // For standard BLE descriptors (e.g., for notifications)

// ----- Wi-Fi Credentials for WPA2-Enterprise -----
const char* EAP_SSID = "PAWS-Secure";
const char* EAP_IDENTITY = "dmv82628@uga.edu"; // Your full username/identity
const char* EAP_USERNAME = "dmv82628"; // Your username
const char* EAP_PASSWORD = "Dv1903@))$"; // Your Wi-Fi password

// If your network requires a CA certificate, uncomment and paste it here.
// const char* ca_cert = R"EOF(
// -----BEGIN CERTIFICATE-----
// MIIDzzCCArOgAwIBAgIUW0E7Zk1lV... (rest of your CA certificate)
// -----END CERTIFICATE-----
// )EOF";

// ----- BLE Setup -----
// BLE UUIDs (Generate your own for unique services/characteristics in real projects!)
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b" // Custom Service UUID
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8" // Custom Characteristic UUID

BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;

// Callback class for BLE server events
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("BLE Client Connected!");
      // Optionally stop advertising to save power after a connection is made
      // This line is also updated for the new getter method if you decide to use it
      // BLEDevice::getAdvertising()->stop();
    }

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("BLE Client Disconnected.");
      // Restart advertising to allow new connections
      // This line is also updated for the new getter method
      BLEDevice::startAdvertising(); // This directly starts advertising again, no need to get the object and call start
      Serial.println("BLE Advertising restarted.");
    }
};

// Function to connect to WPA2-Enterprise Wi-Fi
void connectToEnterpriseWiFi() {
  Serial.print("Connecting to WPA2-Enterprise WiFi: ");
  Serial.println(EAP_SSID);

  WiFi.disconnect(true); // Disconnect from any previous networks
  WiFi.mode(WIFI_STA);   // Set to Station mode

  // Set WPA2 Enterprise specific parameters using direct ESP-IDF functions
  esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
  esp_wifi_sta_wpa2_ent_set_username((uint8_t *)EAP_USERNAME, strlen(EAP_USERNAME));
  esp_wifi_sta_wpa2_ent_set_password((uint8_t *)EAP_PASSWORD, strlen(EAP_PASSWORD));

  // If a CA certificate is required:
  // esp_wifi_sta_wpa2_ent_set_ca_cert((uint8_t *)ca_cert, strlen(ca_cert));

  // Enable WPA2 Enterprise mode for the STA interface
  esp_wifi_sta_wpa2_ent_enable();

  // Start the connection process. DO NOT pass password here for enterprise.
  WiFi.begin(EAP_SSID);

  // Wait for connection
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
    attempts++;
    if (attempts > 30) { // Timeout after 30 seconds (adjust as needed for your network)
      Serial.println("\nFailed to connect to WPA2-Enterprise WiFi. Retrying...");
      WiFi.disconnect(true); // Disconnect and try fresh
      delay(1000);
      esp_wifi_sta_wpa2_ent_enable(); // Re-enable enterprise mode on retry
      WiFi.begin(EAP_SSID); // Re-attempt connection
      attempts = 0;
    }
  }

  Serial.println("\nWiFi connected to WPA2-Enterprise!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);
  delay(100);

  // For ESP32-C3, ensure USB CDC On Boot is enabled in Tools menu for serial output
  Serial.println("Starting ESP32 with Wi-Fi and Bluetooth...");

  // --- Initialize Wi-Fi ---
  connectToEnterpriseWiFi();

  // --- Initialize BLE Server ---
  Serial.println("Initializing BLE...");
  BLEDevice::init("ESP32_MyDevice"); // Set the name of your BLE device (visible to scanners)
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create the BLE Service
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Create a BLE Characteristic
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ | // Client can read the value
                      BLECharacteristic::PROPERTY_NOTIFY // Client can subscribe to notifications
                    );

  // Add a 2902 descriptor to allow clients to enable notifications
  pCharacteristic->addDescriptor(new BLE2902());

  // Start the BLE Service
  pService->start();

  // Start BLE Advertising
  // CORRECTED LINE: Use BLEDevice::getAdvertising() instead of BLEDevice::pAdvertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID); // Advertise your custom service
  pAdvertising->setScanResponse(true);        // Allow scan responses for more info
  // These lines help with iOS connectivity, but might not be strictly necessary for all clients
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising(); // This directly starts advertising, no need for pAdvertising->start()
  Serial.println("BLE Advertising started. Scan for 'ESP32_MyDevice'!");
}

void loop() {
  // --- Wi-Fi Monitoring and Reconnection ---
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, attempting to reconnect to Enterprise network...");
    esp_wifi_sta_wpa2_ent_enable(); // Re-enable enterprise mode
    WiFi.begin(EAP_SSID);           // Re-attempt connection
  }

  // --- BLE Updates (Example: Send a simulated value) ---
  if (deviceConnected) {
    // Simulate some data (e.g., a sensor reading, counter, or status)
    static unsigned long lastValueUpdateTime = 0;
    static int counter = 0;
    if (millis() - lastValueUpdateTime > 2000) { // Update value every 2 seconds
      lastValueUpdateTime = millis();
      counter++;
      String dataToSend = "Count: " + String(counter);
      dataToSend += ", WiFi IP: " + WiFi.localIP().toString(); // Example of combining Wi-Fi info

      // Set the characteristic's value and notify connected clients
      pCharacteristic->setValue(dataToSend.c_str());
      pCharacteristic->notify();
      Serial.print("Notifying BLE client: ");
      Serial.println(dataToSend);
    }
  }

  // --- Other tasks ---
  // You can put any other tasks here, ensuring they don't block for too long.
  // For example, reading sensors, processing data, etc.

  delay(10); // Short delay to yield to other tasks (Wi-Fi, BLE, etc.)
}