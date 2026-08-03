# ============================================================
# MODULE 1: Intelligent Agent — Healthcare Diagnostic Agent
# Covers: Week 2 (Intelligent Agents) + PEAS Framework
# ============================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import datetime

class AgentState(Enum):
    IDLE         = "idle"
    COLLECTING   = "collecting_symptoms"
    DIAGNOSING   = "diagnosing"
    RECOMMENDING = "recommending"
    PLANNING     = "planning_treatment"
    DONE         = "done"

@dataclass
class PatientPercept:
    """What the agent perceives from the environment"""
    patient_id:   str
    symptoms:     List[str]
    age:          int
    temperature:  float
    heart_rate:   int
    blood_pressure: str
    timestamp:    str = field(
        default_factory=lambda: datetime.datetime.now().isoformat())

@dataclass
class AgentMemory:
    """Internal model — makes this a model-based agent"""
    patient_history:  List[Dict]  = field(default_factory=list)
    current_patient:  Optional[PatientPercept] = None
    diagnosis_history: List[str]  = field(default_factory=list)
    action_log:       List[str]   = field(default_factory=list)

class HealthcareDiagnosticAgent:
    """
    PEAS Definition:
    ─────────────────────────────────────────────────
    Performance : Diagnostic accuracy, patient safety,
                  recommendation quality, response time
    Environment : Hospital/clinic, patient data, EMR
    Actuators   : Diagnosis report, treatment plan,
                  referral recommendation, alerts
    Sensors     : Symptom input, vitals, lab results,
                  patient history
    ─────────────────────────────────────────────────
    Agent Type  : Model-Based + Goal-Based + Learning
    """

    def __init__(self):
        self.state   = AgentState.IDLE
        self.memory  = AgentMemory()
        self.performance_score = 0
        self._modules = {}  # Will hold sub-modules

    def register_module(self, name: str, module):
        """Plug in AI sub-modules (KB, Bayes, ML, etc.)"""
        self._modules[name] = module
        print(f"  🔌 Module registered: [{name}]")

    def perceive(self, percept: PatientPercept):
        """Step 1: Perceive the environment"""
        self.memory.current_patient = percept
        self.memory.patient_history.append({
            'id': percept.patient_id,
            'symptoms': percept.symptoms,
            'time': percept.timestamp
        })
        self.state = AgentState.COLLECTING
        self._log(f"Perceived patient {percept.patient_id} "
                  f"with {len(percept.symptoms)} symptoms")
        return self

    def think(self):
        """Step 2: Process and reason"""

        if self.memory.current_patient is None:
            raise ValueError("No patient has been perceived. Call perceive() first.")

        self.state = AgentState.DIAGNOSING
        self._log("Agent thinking: running diagnostic modules...")

        results = {}

        # Run each registered module
        for module_name, module in self._modules.items():

            print(f"\n▶ Running {module_name}...")

            if hasattr(module, "analyze"):
                try:
                    result = module.analyze(self.memory.current_patient)

                    print(f"✔ Finished {module_name}")

                    results[module_name] = result
                    self._log(f"[{module_name}] → {result.get('summary', 'done')}")

                except Exception as e:
                    print(f"❌ {module_name} crashed: {e}")

                    results[module_name] = {
                        "diagnosis": "Unavailable",
                        "confidence": 0.0,
                        "summary": str(e)
                    }

        self.memory.diagnosis_history.append(results)
        self.state = AgentState.RECOMMENDING
        return results

    def act(self, diagnosis_results: Dict) -> Dict:
        """Step 3: Generate action/recommendation"""

        self.state = AgentState.PLANNING
        patient = self.memory.current_patient

        # Aggregate confidence from all successful modules
        confidences = [
            result["confidence"]
            for result in diagnosis_results.values()
            if (
                isinstance(result, dict)
                and result.get("diagnosis") != "Unavailable"
                and "confidence" in result
            )
        ]

        avg_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        )

        # Determine urgency
        urgency = self._assess_urgency(patient, avg_confidence)

        # Get the final diagnosis from all modules
        final_diagnosis = self._aggregate_diagnosis(diagnosis_results)

        # Build the final report
        action_report = {
            'patient_id': patient.patient_id,
            'timestamp': patient.timestamp,
            'symptoms': patient.symptoms,
            'diagnosis': self._aggregate_diagnosis(diagnosis_results),
            'confidence': round(avg_confidence, 3),
            'urgency': urgency,
            'recommendations': self._generate_recommendations(
                urgency, diagnosis_results),
            'next_action': self._decide_next_action(urgency),
            'module_results': diagnosis_results,
            'explanation': self._generate_explanation(diagnosis_results)
        }

        self.performance_score += (
            10 if avg_confidence > 0.7 else 5
        )

        self.state = AgentState.DONE
        self._log(f"Action generated: {urgency} urgency")

        return action_report

    def run(self, percept: PatientPercept) -> Dict:
        """Full agent cycle: Perceive → Think → Act"""
        self.perceive(percept)
        results = self.think()
        return self.act(results)

    def _assess_urgency(self, patient, confidence):
        if patient.temperature > 39.5 or patient.heart_rate > 120:
            return "CRITICAL"
        elif patient.temperature > 38.5 or confidence > 0.8:
            return "HIGH"
        elif patient.temperature > 37.5:
            return "MEDIUM"
        return "LOW"

    def _aggregate_diagnosis(self, results):
        """
        Weighted ensemble diagnosis.
        More reliable AI models contribute more to the final decision.
        """

        # Reliability weights for each module
        weights = {
            "KnowledgeBase": 0.15,
            "BayesianNet": 0.20,
            "MLClassifier": 0.20,
            "NeuralNetwork": 0.25,
            "FuzzyController": 0.10,
            "TreatmentPlanner": 0.10
        }

        diagnosis_scores = {}

        for module_name, result in results.items():

            if not isinstance(result, dict):
                continue

            # Fuzzy logic predicts severity, not disease
            if module_name == "FuzzyController":
                continue

            diagnosis = result.get("diagnosis")

            if diagnosis in (None, "Unavailable", "Unknown"):
                continue

            confidence = result.get("confidence", 0.0)

            weight = weights.get(module_name, 0.10)

            weighted_score = confidence * weight

            diagnosis_scores[diagnosis] = (
                diagnosis_scores.get(diagnosis, 0.0)
                + weighted_score
            )

        if not diagnosis_scores:
            return "Insufficient data"

        return max(diagnosis_scores, key=diagnosis_scores.get)

    def _generate_recommendations(self, urgency, results):
        base = {
            "CRITICAL": [
                "🚨 Immediate emergency consultation required",
                "📞 Alert attending physician now",
                "🏥 Transfer to emergency ward",
                "💊 Administer first-line medications"
            ],
            "HIGH": [
                "⚠️  Schedule urgent appointment within 24 hours",
                "🧪 Order blood panel and cultures",
                "💊 Prescribe symptomatic relief",
                "📋 Monitor vitals every 2 hours"
            ],
            "MEDIUM": [
                "📅 Schedule appointment within 3 days",
                "💊 Over-the-counter treatment advised",
                "🌡️  Monitor temperature twice daily",
                "💧 Increase fluid intake"
            ],
            "LOW": [
                "🏠 Home rest recommended",
                "💧 Stay hydrated",
                "📱 Follow up if symptoms worsen",
                "📋 General wellness monitoring"
            ]
        }
        return base.get(urgency, base["LOW"])

    def _decide_next_action(self, urgency):
        actions = {
            "CRITICAL": "EMERGENCY_REFERRAL",
            "HIGH":     "URGENT_APPOINTMENT",
            "MEDIUM":   "SCHEDULE_FOLLOWUP",
            "LOW":      "MONITOR_AT_HOME"
        }
        return actions.get(urgency, "MONITOR_AT_HOME")

    def _log(self, message):
        entry = f"[{self.state.value}] {message}"
        self.memory.action_log.append(entry)

    def print_log(self):
        print("\n📋 Agent Action Log:")
        print("─" * 50)
        for entry in self.memory.action_log:
            print(f"  {entry}")

    def get_performance(self):
        return {
            'total_patients':    len(self.memory.patient_history),
            'performance_score': self.performance_score,
            'diagnoses_made':    len(self.memory.diagnosis_history)
        }
    def _generate_explanation(self, results):
        """
        Create a human-readable explanation of how the diagnosis was reached.
        """

        explanation = []

        for module_name, result in results.items():

            if not isinstance(result, dict):
                continue

            diagnosis = result.get("diagnosis", "Unknown")
            confidence = result.get("confidence", 0.0)

            explanation.append(
                f"{module_name}: {diagnosis} ({confidence:.2f})"
            )

        return explanation