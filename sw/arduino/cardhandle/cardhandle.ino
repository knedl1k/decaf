/*
 * cardhandle.ino - the arduino loop ran in decaf
 * Author: knedl1k <knedl1k.dev>
 * License: MIT
 */

#include <math.h>
#include <ctype.h> //isspace()
#include <Servo.h>

// stepper
#define MOTOR_0_DIR           (12)
#define MOTOR_0_STEP          (11)

// sensors & encoder AS5147
#define RUMBA_X_POS           (10)
#define RUMBA_X_NEG           (9)
#define RUMBA_Y_POS           (8)
#define RUMBA_Y_NEG           (7)

#define AS5147_CSEL           (RUMBA_X_POS)
#define AS5147_CLK            (RUMBA_X_NEG)
#define AS5147_MOSI           (RUMBA_Y_POS)
#define AS5147_MISO           (RUMBA_Y_NEG)

#define PIN_SOLENOID          (6)
#define PIN_SERVO             (5)

// sensor constants
#define BOTTOM_14_MASK        (0x3FFF)
#define SENSOR_TOTAL_BITS     (16)
#define SENSOR_DATA_BITS      (15)
#define SENSOR_ANGLE_BITS     (14)

// motor speed (delay in us betwen steps)
// lower value => faster motor
#define STEP_DELAY_US         (1000) 
#define TOLERANCE_DEG         (0.5) 

// degs of positions
#define POS_A                 (16.0)
#define POS_B                 (120.0)
#define POS_C                 (62.0)

#define SERVO_Z_UP_ANGLE      (30)
#define SERVO_Z_DOWN_ANGLE    (110)

#define CMD_GO_A              'A'
#define CMD_GO_B              'B'
#define CMD_GO_C              'C'
#define RESP_READY            "R"

float currentAngle = 0.0;
Servo zAxisServo;

void setup() {
    Serial.begin(57600);
    while (!Serial);
  
    pinMode(PIN_SOLENOID, OUTPUT);
    digitalWrite(PIN_SOLENOID, LOW);

    zAxisServo.attach(PIN_SERVO);
    zAxisServo.write(SERVO_Z_UP_ANGLE);

    pinMode(AS5147_CSEL, OUTPUT);
    pinMode(AS5147_CLK , OUTPUT);
    pinMode(AS5147_MOSI, OUTPUT);
    pinMode(AS5147_MISO, INPUT );

    digitalWrite(AS5147_CSEL, HIGH);
    digitalWrite(AS5147_MOSI, HIGH);

    pinMode(MOTOR_0_DIR   , OUTPUT);
    pinMode(MOTOR_0_STEP  , OUTPUT);

    updateSensorData();
    delay(500);
}

/**
 * Reads raw data from AS5147 (SPI bit-banging)
 * @param result Pointer where the result will be saved
 * @return `true` if parity is OK (success), `false` if it failed
 */
bool getSensorRawValue(uint16_t &result) {
    uint8_t j, parity = 0;
    result = 0;

    digitalWrite(AS5147_CSEL, LOW); 

    for(int i = 0; i < SENSOR_TOTAL_BITS; ++i) {
        digitalWrite(AS5147_CLK, HIGH);
        result <<= 1;
        digitalWrite(AS5147_CLK, LOW);

        j = digitalRead(AS5147_MISO);
        result |= j;
        parity ^= (i > 0) & j;
    }

    digitalWrite(AS5147_CSEL, HIGH); 
    return ( parity == (result >> SENSOR_DATA_BITS) );
}

/**
 * Converts raw data to an angle of 0÷360
 * & then to -180÷180
 */
float extractAngleFromRawValue(uint16_t rawValue) {
    float deg = (float)(rawValue & BOTTOM_14_MASK) * 360.0 / (float)(1 << SENSOR_ANGLE_BITS);
    if (deg > 180.0) deg -= 360.0;
    return deg;
}

/**
 * Main function for updating the global variable currentAngle
 */
void updateSensorData() {
    uint16_t rawValue;
    if(getSensorRawValue(rawValue)) {
        currentAngle = extractAngleFromRawValue(rawValue);
    }
}

/**
 * Calculates the smallest difference between two angles (-180÷180)
 * Positive = turn right, Negative = turn left
 */
float getAngleDifference(float target, float current) {
    return target - current;
}

/**
 * Moves the motor toward the target angle
 */
void moveToAngle(float targetAngle) {
    float previousError = 0.0;
    bool firstRun = true;

    for(;;) {
        updateSensorData();
        float error = getAngleDifference(targetAngle, currentAngle);
        //Serial.println(error);
        if(fabs(error) <= TOLERANCE_DEG || (!firstRun && (error * previousError < 0.0))) {
            break;
        }

        firstRun = false;
        previousError = error;

        digitalWrite(MOTOR_0_DIR, (error > 0) ? HIGH : LOW);

        digitalWrite(MOTOR_0_STEP, HIGH);
        delayMicroseconds(500);
        digitalWrite(MOTOR_0_STEP, LOW);
        delayMicroseconds(STEP_DELAY_US); 
    }
}

void executeSequenceA() {
    zAxisServo.write(SERVO_Z_UP_ANGLE);
    delay(500);
    moveToAngle(POS_A);
    zAxisServo.write(SERVO_Z_DOWN_ANGLE);
    delay(500);
    digitalWrite(PIN_SOLENOID, HIGH);
    delay(300);
    zAxisServo.write(SERVO_Z_UP_ANGLE);
    delay(500);
}

void executeSequenceB() {
    moveToAngle(POS_B);
    Serial.println(RESP_READY);
}

void executeSequenceC() {
    moveToAngle(POS_C);
    zAxisServo.write(SERVO_Z_DOWN_ANGLE);
    delay(500);
    digitalWrite(PIN_SOLENOID, LOW);
}

void loop() {
    if (Serial.available() > 0) {
        char cmd = Serial.read();
        if (isspace(cmd)) return;
        while(Serial.available()) Serial.read();

        switch(cmd) {
            case CMD_GO_A:
                executeSequenceA();
                break;
            case CMD_GO_B:
                executeSequenceB();
                break;
            case CMD_GO_C:
                executeSequenceC();
                break;
            default:
                Serial.print("Unknown input: ");
                Serial.println(cmd);
                break;
        }
    }
}
