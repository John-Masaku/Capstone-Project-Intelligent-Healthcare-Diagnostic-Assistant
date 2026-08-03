# ============================================================  
# CAPSTONE MAIN APPLICATION  
# Intelligent Healthcare Diagnostic Assistant  
# Introduction to AI — 13-Week Capstone  
# ============================================================  

import sys  
import json  
import warnings  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.gridspec as gridspec  
warnings.filterwarnings('ignore')  

# Import all modules  
from modules.agent          import HealthcareDiagnosticAgent, PatientPercept  
from modules.knowledge_base import MedicalKnowledgeBase  
from modules.bayesian_net   import SimpleBayesianDiagnostics  
from modules.ml_classifier  import MLDiagnosticClassifier  
from modules.neural_network import NeuralDiagnosticModel  
from modules.fuzzy_controller import FuzzySeverityAssessor  
from modules.planner        import TreatmentPlanner  

# ── ANSI Colors ────────────────────────────────────────────  
class C:  
    HEADER = '\033[95m'; BLUE   = '\033[94m'  
    GREEN  = '\033[92m'; YELLOW = '\033[93m'  
    RED    = '\033[91m'; BOLD   = '\033[1m'  
    END    = '\033[0m'  

def banner():  
    print(f"""  
{C.BOLD}{C.BLUE}  
╔══════════════════════════════════════════════════════════╗  
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║  
║         Introduction to AI — Capstone Project            ║  
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy      ║  
╚══════════════════════════════════════════════════════════╝  
{C.END}""")  

def section(title: str):  
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  

def build_system() -> HealthcareDiagnosticAgent:  
    """Instantiate and wire all AI modules"""  
    section("🔧 Building AI System — Registering Modules")  

    agent = HealthcareDiagnosticAgent()  

    print("\n  Initializing modules...")  
    modules = {
        'KnowledgeBase': MedicalKnowledgeBase(),
        'BayesianNet': SimpleBayesianDiagnostics(),
        'MLClassifier': MLDiagnosticClassifier(),
        'NeuralNetwork': NeuralDiagnosticModel(),
        'FuzzyController': FuzzySeverityAssessor(),
        'TreatmentPlanner': TreatmentPlanner()
    }
    
    for name, module in modules.items():
        agent.register_module(name, module)

    return agent

def main():
    """Main application"""

    banner()

    # Build the AI system
    agent = build_system()
    
    section("🩺 Creating Sample Patient")

    patient = PatientPercept(
        patient_id="P001",
        symptoms=["fever", "cough", "fatigue"],
        age=25,
        temperature=38.6,
        heart_rate=98,
        blood_pressure="120/80"
    )

    print(f"Patient ID : {patient.patient_id}")
    print(f"Symptoms   : {', '.join(patient.symptoms)}")
    print(f"Temperature: {patient.temperature}°C")
    print(f"Heart Rate : {patient.heart_rate} bpm")
    print(f"Blood Pressure: {patient.blood_pressure}")
    
    section("🧠 Running Intelligent Diagnostic Agent")

    report = agent.run(patient)

    module_results = report["module_results"]

    print("\n✅ Diagnosis Complete")

    section("📋 Final Diagnostic Report")

    print(f"Patient ID    : {report['patient_id']}")
    print(f"Diagnosis     : {report['diagnosis']}")
    print(f"Confidence    : {report['confidence']:.2f}")
    print(f"Urgency       : {report['urgency']}")
    print(f"Next Action   : {report['next_action']}")
    
    section("📝 AI Decision Explanation")

    for line in report["explanation"]:
        print(f"• {line}")

    section("🧠 Individual AI Module Results")

    for module, result in module_results.items():

        print(f"\n[{module}]")

        if isinstance(result, dict):

            if "summary" in result:
                print(f"Summary      : {result['summary']}")

            if "diagnosis" in result:
                print(f"Diagnosis    : {result['diagnosis']}")

            if "confidence" in result:
                print(f"Confidence   : {result['confidence']:.2f}")

            # Fuzzy Logic output
            if "severity_score" in result:
                print(f"Severity     : {result['severity_label']}")
                print(f"Score        : {result['severity_score']}/100")

            # Treatment Planner output
            if "plan" in result:
                print("\nTreatment Plan:")
                for step in result["plan"]:
                    print(
                        f"  {step['step']}. {step['action']} "
                        f"({step['duration']})"
                    )

    print("\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")

    print("\n🎉 Healthcare Diagnostic Process Completed Successfully!")
    
    
if __name__ == "__main__":
    main()