/*
 * =============================================================================
 *  UNIFIED MULTIMODAL ACQUISITION FIRMWARE - 120 Hz (8ms Exposure)
 *  Target: Arduino Uno (ATmega328P @ 16 MHz)
 * =============================================================================
 *  Features:
 *  - 120 Hz Frame Rate (8.333 ms period)
 *  - LED Control: Active-HIGH, turns ON at 0.0 ms, OFF at 7.5 ms.
 *  - Camera Trigger: Active-LOW, goes LOW at 0.2 ms, goes HIGH at 0.35 ms.
 *    (LED turns on 0.2 ms before camera trigger to allow stabilization)
 *    (Exposure lasts 7.0 ms, ending at 7.20 ms. LED turns off at 7.5 ms)
 * =============================================================================
 */

#include <Arduino.h>
#include <SoftwareSerial.h>
#include <avr/interrupt.h>
#include <avr/io.h>

// ---------------------------------------------------------------------------
//  SOFTWARE SERIAL FOR MOD-PULSE
// ---------------------------------------------------------------------------
SoftwareSerial modPulseSerial(12, 13); // RX = 12, TX = 13

// ---------------------------------------------------------------------------
//  PIN MAPPINGS & PORT MASKS
// ---------------------------------------------------------------------------
#define CH0_PORTD_MASK _BV(PD2)
#define TRIGGER_PORTD_MASK _BV(PD3)
#define LED_PORTB_MASK (_BV(PB0) | _BV(PB1) | _BV(PB2))

// ---------------------------------------------------------------------------
//  TIMER1 CONFIGURATION (120 Hz)
//  Mode 12 — CTC, TOP = ICR1
//
//  Clock = 16 MHz, Prescaler /8  → 1 tick = 0.5 µs
//  Period = 16666 ticks × 0.5 µs = 8333 µs → ~120.00 Hz
// ---------------------------------------------------------------------------
#define TIMER1_TOP          16665  // 8.333 ms period at 120 Hz
#define TIMER1_TRIGGER_LOW   399   // 0.200 ms — pull trigger LOW (LED already ON)
#define TIMER1_TRIGGER_HIGH  699   // 0.350 ms — release trigger HIGH (150 µs pulse)
#define TIMER1_LED_OFF      14999  // 7.500 ms — turn LED off (exposure ends at 7.200 ms)

// ---------------------------------------------------------------------------
//  BINARY PROTOCOL DEFINITIONS
// ---------------------------------------------------------------------------
#define SYNC_BYTE_1 0xAA
#define SYNC_BYTE_2 0x55
#define PACKET_SIZE 8

#define CMD_START   0x01
#define CMD_STOP    0x02
#define CMD_CAM_OK  0x03
#define CMD_CAM_ERR 0x04

// ---------------------------------------------------------------------------
//  STATE VARIABLES (ISR <-> loop)
// ---------------------------------------------------------------------------
enum SystemState { STATE_BOOTING, STATE_READY, STATE_RUNNING, STATE_DONE };
static SystemState currentState = STATE_BOOTING;
static uint32_t doneStartTime = 0;

static volatile uint32_t frameCounter = 0;
static volatile uint8_t  nextChannel  = 0;
static volatile bool triggerPhase = true;

// Latest parsed sensor values
static volatile uint8_t lastIR   = 0;
static volatile uint8_t lastHR   = 0;
static volatile uint8_t lastSpO2 = 0;

// Circular Buffer
#define BUFFER_SIZE 128
static volatile uint8_t  ring_ir [BUFFER_SIZE];
static volatile uint8_t  ring_hr [BUFFER_SIZE];
static volatile uint8_t  ring_sp [BUFFER_SIZE];
static volatile uint8_t  ring_ch [BUFFER_SIZE];
static volatile uint16_t ring_frm[BUFFER_SIZE];
static volatile uint32_t ring_head = 0;
static          uint32_t ring_tail = 0;

// MOD-PULSE PARSER STATE MACHINE
enum ParserState : uint8_t { WAIT_00, WAIT_FF, READ_IR, READ_HR, READ_SPO2 };
static ParserState parserState       = WAIT_00;

// ---------------------------------------------------------------------------
//  GPIO CONTROL FUNCTIONS
// ---------------------------------------------------------------------------
static inline void onFrameStart(uint8_t channel) {
  PORTB &= ~LED_PORTB_MASK;
  uint8_t portd_bits = TRIGGER_PORTD_MASK; 
  if (channel == 0) {
    portd_bits |= CH0_PORTD_MASK;
  }
  PORTD = (PORTD & ~(CH0_PORTD_MASK | TRIGGER_PORTD_MASK)) | portd_bits;
  switch (channel) {
  case 1: PORTB |= _BV(PB0); break;
  case 2: PORTB |= _BV(PB1); break;
  case 3: PORTB |= _BV(PB2); break;
  }
}

static inline void onTriggerLow()  { PORTD &= ~TRIGGER_PORTD_MASK; }
static inline void onTriggerHigh() { PORTD |=  TRIGGER_PORTD_MASK; }
static inline void onLedOff() {
  PORTD &= ~CH0_PORTD_MASK;
  PORTB &= ~LED_PORTB_MASK;
}

// ---------------------------------------------------------------------------
//  TIMER1 INTERRUPT SERVICE ROUTINES
// ---------------------------------------------------------------------------
ISR(TIMER1_CAPT_vect) {
  const uint8_t channel = nextChannel;
  onFrameStart(channel);

  frameCounter++;

  const uint8_t idx = ring_head % BUFFER_SIZE;
  ring_ir [idx] = lastIR;
  ring_hr [idx] = lastHR;
  ring_sp [idx] = lastSpO2;
  ring_ch [idx] = channel;
  ring_frm[idx] = (uint16_t)(frameCounter & 0xFFFF);
  ring_head++;

  nextChannel = (channel + 1) & 0x03;

  triggerPhase = true;
  OCR1B = TIMER1_TRIGGER_HIGH;
}

ISR(TIMER1_COMPA_vect) {
  onTriggerLow();
}

ISR(TIMER1_COMPB_vect) {
  if (triggerPhase) {
    onTriggerHigh();
    OCR1B = TIMER1_LED_OFF; 
    triggerPhase = false;
  } else {
    onLedOff();
  }
}

// ---------------------------------------------------------------------------
//  SETUP
// ---------------------------------------------------------------------------
void setup() {
  pinMode(2,  OUTPUT);  // CH0
  pinMode(8,  OUTPUT);  // CH1
  pinMode(9,  OUTPUT);  // CH2
  pinMode(10, OUTPUT);  // CH3
  pinMode(3,  OUTPUT);  // Camera Trigger

  onLedOff();
  onTriggerHigh(); 

  Serial.begin(115200);
  modPulseSerial.begin(115200);

  Serial.println(F("# MOD-PULSE Multimodal Sync v3.1 (120Hz 7ms Exp)"));
  Serial.println(F("# Mode: Unified Single-Arduino (Pre-LED 0.2ms)"));
  Serial.flush();

  cli();
  TCCR1A = 0x00;                      
  TCCR1B = _BV(WGM13) | _BV(WGM12);  
  TCNT1  = 0;

  ICR1  = TIMER1_TOP;           
  OCR1A = TIMER1_TRIGGER_LOW;   
  OCR1B = TIMER1_TRIGGER_HIGH;  

  TIMSK1 = _BV(ICIE1) | _BV(OCIE1A) | _BV(OCIE1B);
  TIFR1  = _BV(ICF1)  | _BV(OCF1A)  | _BV(OCF1B);
  sei();

  currentState = STATE_READY;
  Serial.println(F("READY"));
  Serial.flush();
}

// ---------------------------------------------------------------------------
//  MAIN LOOP
// ---------------------------------------------------------------------------
static bool isRunning = false;

void loop() {
  if (currentState == STATE_DONE && (millis() - doneStartTime >= 3000)) {
    currentState = STATE_READY;
  }

  while (Serial.available() > 0) {
    uint8_t cmd = (uint8_t)Serial.read();

    switch (cmd) {
    case CMD_START:
      cli();
      frameCounter = 0;
      nextChannel  = 0;
      ring_head    = 0;
      ring_tail    = 0;
      triggerPhase = true;
      OCR1B        = TIMER1_TRIGGER_HIGH;
      TCNT1        = TIMER1_TOP - 1;
      TIFR1 = _BV(ICF1) | _BV(OCF1A) | _BV(OCF1B);
      sei();

      isRunning    = true;
      currentState = STATE_RUNNING;

      Serial.println(F("STARTED,120,4"));
      Serial.flush();

      TCCR1B |= _BV(CS11);
      break;

    case CMD_STOP:
      isRunning     = false;
      currentState  = STATE_DONE;
      doneStartTime = millis();
      TCCR1B &= ~(_BV(CS12) | _BV(CS11) | _BV(CS10));
      onLedOff();
      onTriggerHigh();
      Serial.println(F("STOPPED"));
      Serial.flush();
      break;

    case CMD_CAM_OK:
    case CMD_CAM_ERR:
      break;
    }
  }

  while (modPulseSerial.available() > 0) {
    uint8_t b = (uint8_t)modPulseSerial.read();
    switch (parserState) {
    case WAIT_00: if (b == 0x00) parserState = WAIT_FF; break;
    case WAIT_FF:
      if      (b == 0xFF) parserState = READ_IR;
      else if (b == 0x00) parserState = WAIT_FF;
      else                parserState = WAIT_00;
      break;
    case READ_IR: lastIR = b; parserState = READ_HR; break;
    case READ_HR: lastHR = b; parserState = READ_SPO2; break;
    case READ_SPO2:
      lastSpO2 = b;
      parserState = WAIT_00;
      break;
    }
  }

  uint32_t currentHead;
  cli();
  currentHead = ring_head;
  sei();

  if (currentHead - ring_tail > BUFFER_SIZE) {
    ring_tail = currentHead - BUFFER_SIZE;
  }

  while (ring_tail < currentHead) {
    const uint8_t idx = ring_tail % BUFFER_SIZE;
    uint8_t packet[PACKET_SIZE];
    packet[0] = SYNC_BYTE_1;                        
    packet[1] = SYNC_BYTE_2;                        
    packet[2] = (uint8_t)(ring_frm[idx] >> 8);     
    packet[3] = (uint8_t)(ring_frm[idx] & 0xFF);   
    packet[4] = ring_ch[idx];                       
    packet[5] = ring_ir[idx];                       
    packet[6] = ring_hr[idx];                       
    packet[7] = ring_sp[idx];                       
    Serial.write(packet, PACKET_SIZE);
    ring_tail++;
  }
}
