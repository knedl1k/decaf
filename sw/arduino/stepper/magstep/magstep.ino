/*
 * magstep.ino - Modul for rotating a stepper motor using an encoder
 *
 * Author:  knedl1k <knedl1k@tuta.io>
 * License: MIT
 */

#include <math.h>

// pins
#define MOTOR_0_DIR           (47)
#define MOTOR_0_STEP          (46)
#define MOTOR_0_ENABLE        (52)

// sensors & encoder AS5147
#define RUMBA_X_POS   (37)
#define RUMBA_X_NEG   (36)
#define RUMBA_Y_POS   (35)
#define RUMBA_Y_NEG   (34)

#define AS5147_CSEL   (RUMBA_X_POS)
#define AS5147_CLK    (RUMBA_X_NEG)
#define AS5147_MOSI   (RUMBA_Y_POS)
#define AS5147_MISO   (RUMBA_Y_NEG)

// sensor constants
#define BOTTOM_14_MASK       (0x3FFF)
#define SENSOR_TOTAL_BITS    (16)
#define SENSOR_DATA_BITS     (15)
#define SENSOR_ANGLE_BITS    (14)

// motor speed (delay in us between steps)
// lower value => faster motor
#define STEP_DELAY_US        (1000) 

// motor tolerance (deadband), in degrees.
// a small reserve so that the motor does not oscillate around the target
#define TOLERANCE_DEG        (0.2) 

float currentAngle = 0.0;

void setup() {
  Serial.begin(57600);
  while (!Serial);
  
  Serial.println("\n\n** START: Waiting for input; rotation in degrees (-180÷180) **\n\n");

  // encoder pins setup
  pinMode(AS5147_CSEL, OUTPUT);
  pinMode(AS5147_CLK , OUTPUT);
  pinMode(AS5147_MOSI, OUTPUT);
  pinMode(AS5147_MISO, INPUT );

  digitalWrite(AS5147_CSEL, HIGH);
  digitalWrite(AS5147_MOSI, HIGH);
    
  // motor pins setup
  pinMode(MOTOR_0_DIR   , OUTPUT);
  pinMode(MOTOR_0_STEP  , OUTPUT);
  pinMode(MOTOR_0_ENABLE, OUTPUT);

  digitalWrite(MOTOR_0_ENABLE, LOW); // enable the driver

  updateSensorData();
  Serial.print("Current angle: ");
  Serial.println(currentAngle);
}

/**
 * Reads raw data from AS5147 (SPI bit-banging)
 * @param result Pointer where the result will be saved
 * @return `true` if parity is OK (success), `false` if it failed
 */
bool getSensorRawValue(uint16_t &result) {
  uint8_t j, parity = 0;
  result = 0;
  
  digitalWrite(AS5147_CSEL, LOW); // start of transmission
  
  for(int i=0; i<SENSOR_TOTAL_BITS; ++i) {
    digitalWrite(AS5147_CLK, HIGH);
    //delayMicroseconds(1); // small pause just for the sake
    result <<= 1;
    digitalWrite(AS5147_CLK, LOW);
    
    j = digitalRead(AS5147_MISO);
#ifdef VERBOSE
    Serial.print(j,DEC);
#endif
    result |= j;
    parity ^= (i>0) & j;
    // // calculation of parity (XOR) for the first 15 bits
    // if (i < 15) {
    //    parity ^= j;
    // }
  }

  digitalWrite(AS5147_CSEL, HIGH); // end of transmission
  
  return ( parity == (result>>SENSOR_DATA_BITS) );
}

/**
 * Converts raw data to an angle of 0÷360
 * & then to -180÷180
 */
float extractAngleFromRawValue(uint16_t rawValue) {
  float deg = (float)(rawValue & BOTTOM_14_MASK) * 360.0 / (float)(1 << SENSOR_ANGLE_BITS);
  
  if (deg > 180.0)
    deg -= 360.0;
  
  return deg;
}

/**
 * Main function for updating the global variable currentAngle
 */
void updateSensorData() {
  uint16_t rawValue;
  if(getSensorRawValue(rawValue))
    currentAngle = extractAngleFromRawValue(rawValue);
}

/**
 * Calculates the smallest difference between two angles (-180÷180)
 * Positive = turn right, Negative = turn left
 */
float getAngleDifference(float target, float current) {
  float diff = target - current;
  
  if (diff > 180) diff -= 360;
  if (diff < -180) diff += 360;
  
  return diff;
}

/**
 * Moves the motor toward the target angle
 */
void moveToAngle(float targetAngle) {
  Serial.print("Turning to an angle: ");
  Serial.println(targetAngle);

  // wraparound for the input
  if (targetAngle > 180) targetAngle -= 360;
  if (targetAngle < -180) targetAngle += 360;

  //bool reached = false;

  for(;;){
    updateSensorData();
    float error = getAngleDifference(targetAngle, currentAngle);

    // ONLY FOR DEBUG, IT SLOWS DOWN THE ROTATION
#ifdef DEBUG
    Serial.print("Err: "); Serial.println(error);
#endif

    if(abs(error) <= TOLERANCE_DEG) // we reached the target angle
      break;

    if (error > 0)
      digitalWrite(MOTOR_0_DIR, LOW); // LOW
    else
      digitalWrite(MOTOR_0_DIR, HIGH);

    digitalWrite(MOTOR_0_STEP, HIGH);
    delayMicroseconds(10);
    digitalWrite(MOTOR_0_STEP, LOW);
      
    delayMicroseconds(STEP_DELAY_US); 
  }
  
  Serial.print("Target angle reached. Current angle: ");
  Serial.println(currentAngle);
}

void loop() {
  if (Serial.available() > 0) {
    float target = Serial.parseFloat();
  
    while(Serial.available()) Serial.read();

    moveToAngle(target);
  }
}