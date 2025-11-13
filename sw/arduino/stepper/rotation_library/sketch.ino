#include <TMCStepper.h>

#define EN_PIN         52
#define DIR_PIN        47
#define STEP_PIN       46
#define SW_RX          19
#define SW_TX          18
#define SERIAL_PORT    Serial1       // Mega: TX1 = pin 18
#define DRIVER_ADDRESS 0b00
#define R_SENSE        0.11f

// TMC2209Stepper driver(&SERIAL_PORT, R_SENSE, DRIVER_ADDRESS);
TMC2209Stepper driver(SW_RX, SW_TX, R_SENSE, DRIVER_ADDRESS);

void setup() {
  Serial.begin(250000);
  while(!Serial);
  Serial.println("\nStart...");

  SERIAL_PORT.begin(115200);
  driver.beginSerial(115200);

  pinMode(EN_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  digitalWrite(EN_PIN, LOW);        // Enable the driver
  digitalWrite(DIR_PIN, HIGH);      // Set direction (change to LOW if needed)

  driver.begin();
  driver.toff(5);                   // Enable driver (required)
  driver.rms_current(1200);         // Adjust for your motor (in mA)
  driver.microsteps(16);            // Lower = faster speed (try 1, 2, or 4 for max speed)
  driver.pwm_autoscale(true);       // Enable StealthChop for smooth motion

  delay(500); // Let it settle
}

void loop() {
  // High-speed spin: send step pulses as fast as possible
  digitalWrite(STEP_PIN, HIGH);
  delayMicroseconds(100); // Adjust this to go faster or slower
  digitalWrite(STEP_PIN, LOW);
  delayMicroseconds(100);
}

