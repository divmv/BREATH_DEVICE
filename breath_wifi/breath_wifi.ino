#include <WiFi.h>
#include <esp_wpa2.h>       // Defines esp_wifi_sta_wpa2_ent_enable() and others
#include <esp_eap_client.h> // Good practice to include, even if not directly calling
#include <esp_wifi.h>       // Often needed for wifi_init_config_t and similar low-level types

// ----- Wi-Fi Credentials for WPA2-Enterprise -----
const char* EAP_SSID = "PAWS-Secure";
const char* EAP_IDENTITY = "dmv82628@uga.edu"; // Your full username/identity
const char* EAP_USERNAME = "dmv82628"; // Your username (often the same as identity, but sometimes without the domain)
const char* EAP_PASSWORD = "Dv1903@))$"; // Your Wi-Fi password

// If your network requires a CA certificate, uncomment and paste it here.
// const char* ca_cert = R"EOF(
// -----BEGIN CERTIFICATE-----
// MIIDzzCCArOgAwIBAgIUW0E7Zk1lV... (rest of your CA certificate)
// -----END CERTIFICATE-----
// )EOF";

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
  // This function is still in esp_wpa2.h in many 3.x.x versions
  esp_wifi_sta_wpa2_ent_enable();

  // Start the connection process. DO NOT pass password here for enterprise.
  WiFi.begin(EAP_SSID);

  // Wait for connection
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
    attempts++;
    if (attempts > 30) { // Timeout after 30 seconds
      Serial.println("\nFailed to connect to WPA2-Enterprise WiFi. Retrying...");
      WiFi.disconnect(true);
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

  // For ESP32-C3, ensure USB CDC On Boot is enabled in Tools menu
  Serial.println("Starting ESP32 with WPA2-Enterprise...");

  connectToEnterpriseWiFi();
}

void loop() {
  // Your main application loop
  // You can check WiFi.status() here if you want to handle disconnections
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, attempting to reconnect to Enterprise network...");
    // Reconnect by re-enabling enterprise and calling WiFi.begin()
    esp_wifi_sta_wpa2_ent_enable();
    WiFi.begin(EAP_SSID);
  }

  // Your other tasks (e.g., Bluetooth, sensor readings, data uploads)
  // Integrate your Bluetooth code here if needed.

  delay(5000); // Wait 5 seconds before checking again
}