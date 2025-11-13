#define STEP_PIN 46
#define DIR_PIN 47
#define EN_PIN 52 // LOW: Driver enabled, HIGH: Driver disabled

// SLP & RST pins need to be on HIGH, otherwise the driver will conserve power
// M0, M1, M2 are microstepping pins; when disconnected, driver defaults to full-step

// B+ blue & red
// A+ black & green

void setup() {
  pinMode(EN_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  digitalWrite(EN_PIN, LOW); // Enable the driver
}

void loop() {
  digitalWrite(DIR_PIN, HIGH); 
  moveSteps(6400);
  delay(1000);

  digitalWrite(DIR_PIN, LOW);
  moveSteps(6400);
  delay(1000);
}

void moveSteps(int steps) {
  for (int i = 0; i < steps; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(500);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(500);
  }
}
