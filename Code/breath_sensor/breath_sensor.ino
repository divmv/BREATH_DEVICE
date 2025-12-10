int VT_B_PIN = 2; 
int VL_R_PIN = 3; 
int VSUM_PIN = 4; 


const int ADC_RESOLUTION = 4095; 
const float ADC_VOLTAGE_MAX = 3.3; 

void setup() {
  Serial.begin(115200);
  Serial.println("Test");
  analogSetAttenuation(ADC_11db); 
}

void loop() {
  // Read raw analog values from the ADC pins
  int rawVtB = analogRead(VT_B_PIN);
  int rawVlR = analogRead(VL_R_PIN);
  int rawVSum = analogRead(VSUM_PIN);

  // Convert raw ADC values to voltage (0-3.3V)
  float voltageVtB = (float)rawVtB * ADC_VOLTAGE_MAX / ADC_RESOLUTION;
  float voltageVlR = (float)rawVlR * ADC_VOLTAGE_MAX / ADC_RESOLUTION;
  float voltageVSum = (float)rawVSum * ADC_VOLTAGE_MAX / ADC_RESOLUTION;

  float originalVtB = mapFloat(voltageVtB, 0.0, ADC_VOLTAGE_MAX, -3.0, 3.0);
  float originalVlR = mapFloat(voltageVlR, 0.0, ADC_VOLTAGE_MAX, -3.0, 3.0);
  float originalVSum = mapFloat(voltageVSum, 0.0, ADC_VOLTAGE_MAX, -3.0, 3.0);


  // Serial.print("VT-B (raw): ");
  // Serial.print(rawVtB);
  Serial.print(" | VT-B (V): ");
  Serial.print(voltageVtB, 3); 
  // Serial.print(" | VT-B (Original V): ");
  // Serial.print(originalVtB, 3);

  // Serial.print(" | VL-R (raw): ");
  // Serial.print(rawVlR);
  Serial.print(" | VL-R (V): ");
  Serial.print(voltageVlR, 3);
  // Serial.print(" | VL-R (Original V): ");
  // Serial.print(originalVlR, 3);

  // Serial.print(" | VSUM (raw): ");
  // Serial.print(rawVSum);
  Serial.print(" | VSUM (V): ");
  Serial.print(voltageVSum, 3);
  Serial.println();
  // Serial.print(" | VSUM (Original V): ");
  // Serial.println(originalVSum, 3);
  // Serial.println();

  delay(1000);
}

float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}