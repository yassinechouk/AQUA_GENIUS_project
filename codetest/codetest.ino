// --- Capteur ultrason ---
const int TRIG_PIN = 25;
const int ECHO_PIN = 26;

// --- Moteurs ---
int M1_IN1 = 17;
int M1_IN2 = 16;
int M2_IN1 = 19;
int M2_IN2 = 18;

// --- Pins de commande supplémentaires ---
int commandeM2_1 = 32; // si HIGH → M2 tourne
int commandeM2_2 = 33; // si HIGH → M2 tourne

// --- Seuil distance ---
float seuil = 7.5; // cm

// --- PWM pour vitesse moteurs (0-255) ---
int vitesseM1 = 100;
int vitesseM2 = 100;

void setup() {
  Serial.begin(115200);

  // Capteur ultrason
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Moteurs
  pinMode(M1_IN1, OUTPUT);
  pinMode(M1_IN2, OUTPUT);
  pinMode(M2_IN1, OUTPUT);
  pinMode(M2_IN2, OUTPUT);

  // Pins commande M2
  pinMode(commandeM2_1, INPUT);
  pinMode(commandeM2_2, INPUT);

  // Initialisation moteurs
  moteurStop(M1_IN1, M1_IN2);
  moteurStop(M2_IN1, M2_IN2);

  Serial.println("=== System motopompes + ultrason + commande M2 ===");
}

void loop() {

  float distance = mesurer_distance();

  // Lecture pins commande M2
  int cmd1 = digitalRead(commandeM2_1);
  int cmd2 = digitalRead(commandeM2_2);

  Serial.print("CMD1=");
  Serial.print(cmd1);
  Serial.print(" | CMD2=");
  Serial.println(cmd2);

  // --- Si CMD1 ou CMD2 = HIGH → M2 tourne directement ---
  if (cmd1 == HIGH || cmd2 == HIGH) {
    Serial.println("Commande externe : M2 ON !");
    
    moteurAvance(M2_IN1, M2_IN2, vitesseM2);
    delay(300);
    return;     // On ignore le reste du code
  }

  // --- Sinon fonctionnement normal avec l’ultrason ---

  if (distance == -1) {
    Serial.println("Erreur: Timeout !");
    moteurStop(M1_IN1, M1_IN2);
    moteurStop(M2_IN1, M2_IN2);
  }
  else if (distance < 2 || distance > 400) {
    Serial.println("Distance hors plage !");
    moteurStop(M1_IN1, M1_IN2);
    moteurStop(M2_IN1, M2_IN2);
  }
  else {
    Serial.print("Distance: ");
    Serial.print(distance);
    Serial.println(" cm");

    if (distance > seuil) {
      // Distance haute → M1 tourne
      moteurAvance(M1_IN1, M1_IN2, vitesseM1);
      moteurStop(M2_IN1, M2_IN2);
    } else {
      // Distance basse → M2 tourne
      moteurStop(M1_IN1, M1_IN2);
      moteurStop(M2_IN1, M2_IN2);
    }
  }

  delay(300);
}

// --- Fonctions moteurs ---
void moteurAvance(int IN1, int IN2, int vitesse) {
  analogWrite(IN1, vitesse);
  analogWrite(IN2, 0);
}

void moteurStop(int IN1, int IN2) {
  analogWrite(IN1, 0);
  analogWrite(IN2, 0);
}

// --- Fonction mesure distance ---
float mesurer_distance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1;

  return (duration * 0.0343) / 2.0;
}
