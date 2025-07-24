const int VSUM_PIN = 8;  
const int VLR_PIN = 9; 
const int VBT_PIN = 10;

void setup() {
  Serial.begin(115200);
  delay(100);
  analogSetAttenuation(ADC_11db);
  Serial.println("Attenuation set and began.");

}

void loop() {

  int rawVSUM = analogRead(VSUM_PIN);
  int rawVLR = analogRead(VLR_PIN);
  int rawVBT = analogRead(VBT_PIN);

  float voltageVBT = map(rawVBT, 0, 4095, 0, 2600) / 1000.0;
  float voltageVLR = map(rawVLR, 0, 4095, 0, 2600) / 1000.0;
  float voltageVSUM = map(rawVSUM, 0, 4095, 0, 2600) / 1000.0;

  Serial.print("VT-B Raw: "); Serial.print(rawVBT); Serial.print(", Voltage: "); Serial.print(voltageVBT, 3); Serial.println("V");
  Serial.print("VL-R Raw: "); Serial.print(rawVLR); Serial.print(", Voltage: "); Serial.print(voltageVLR, 3); Serial.println("V");
  Serial.print("VSUM Raw: "); Serial.print(rawVSUM); Serial.print(", Voltage: "); Serial.print(voltageVSUM, 3); Serial.println("V");
  Serial.println("---");

  delay(500); // Adjust delay as needed

}