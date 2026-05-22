/*
 * servo_calibration.ino - manual measurement of angles
 */

#include <Servo.h>

#define PIN_SERVO (5) 

Servo zAxisServo;
int currentServoAngle = 90;

void setup() {
    Serial.begin(57600);
    while (!Serial);

    zAxisServo.attach(PIN_SERVO);
    zAxisServo.write(currentServoAngle);

    Serial.println("\n--- MG90S Calibration ---");
    Serial.println("w : +10 deg  |  s : -10 deg");
    Serial.println("e : +20 deg  |  d : -20 deg");
    Serial.print("Angle now: ");
    Serial.println(currentServoAngle);
}

void loop() {
    if (Serial.available() > 0) {
        char cmd = Serial.read();

        while(Serial.available()) Serial.read();

        switch (cmd) {
            case 'w': currentServoAngle += 10; break;
            case 's': currentServoAngle -= 10; break;
            case 'e': currentServoAngle += 20; break;
            case 'd': currentServoAngle -= 20; break;
            default: return;
        }

        if (currentServoAngle > 180) currentServoAngle = 180;
        if (currentServoAngle < 0) currentServoAngle = 0;

        zAxisServo.write(currentServoAngle);
        
        Serial.print("Set angle: ");
        Serial.println(currentServoAngle);
    }
}
