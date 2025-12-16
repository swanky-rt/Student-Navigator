import sys
import json
from context_agent_classifier import ContextAgentClassifier
# Import the PII extractor you uploaded earlier
try:
    from pii_extractor import extract_pii
except ImportError:
    # Fallback mock if pii_extractor isn't in the same folder yet
    def extract_pii(text, domain, **kwargs):
        return [{"text": "MOCK_DATA", "label": "MOCK_PII"}]

# Config
MODEL_PATH = "context_agent_mlp.pth"
LABELS = {0: "restaurant", 1: "bank"}

def run_pipeline():
    # 1. Load your trained Classifier
    print("Loading Context Agent...")
    classifier = ContextAgentClassifier()
    classifier.load_model(MODEL_PATH)
    
    # 2. Test Inputs (One difficult Bank, one difficult Restaurant)
    test_cases = [
        "I need to transfer $500 to my checking account immediately.",
        "Book a table for 4 people at 7 PM for dinner."
    ]
    
    print("\n--- 🚀 AIRGAP LITE PIPELINE DEMO ---\n")
    
    for text in test_cases:
        print(f"User Input: \"{text}\"")
        
        # A. CLASSIFY (The 'Brain')
        prediction = classifier.predict(text, LABELS)
        detected_domain = prediction['label']
        confidence = prediction['confidence']
        
        print(f"  → 🧠 Context Agent Detected: [{detected_domain.upper()}] (Conf: {confidence:.2f})")
        
        # B. EXTRACT (The 'Hands')
        # We automatically pass the detected domain to the extractor
        print(f"  → ⚙️  Applying {detected_domain} privacy rules...")
        
        pii_results = extract_pii(text, domain=detected_domain)
        
        # C. REPORT
        if pii_results:
            print(f"  → 🔒 PII Extracted: {json.dumps(pii_results)}")
        else:
            print(f"  → ✅ No sensitive PII found for this domain.")
        print("-" * 50)

if __name__ == "__main__":
    run_pipeline()