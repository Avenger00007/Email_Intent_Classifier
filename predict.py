import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_PATH = "models/distilbert-intent"
MAX_LENGTH = 128


# =========================================================
# 2. CHECK MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found at: {MODEL_PATH}\n"
        "Please train the model first using:\n"
        "python src/train.py"
    )


# =========================================================
# 3. CHECK DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("EMAIL & MESSAGE INTENT CLASSIFIER")
print("=" * 60)
print("Device:", device)


# =========================================================
# 4. LOAD TOKENIZER
# =========================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


# =========================================================
# 5. LOAD FINE-TUNED MODEL
# =========================================================

print("Loading fine-tuned DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# =========================================================
# 6. DEPARTMENT MAPPING
# =========================================================


department_mapping = {
    "delivery_issue": "Delivery Issue Department",
    "billing_refund": "Billing and Refund Department",
    "technical_support": "Technical Support Department",
    "order_cancellation": "Order Cancellation Department",
    "account_access": "Account Access Department",
    "course_information": "Course Information Department",
    "education": "Education Department",
    "offer_letter": "Offer Letter Department",
    "government_services": "Government Services Department",
    "financial_services": "Financial Services Department",
    "course_inquiry": "Course Inquiry Department",
    "course_enrollment": "Course Enrollment Department",
    "job_application": "Job Application Department",
    "scholarship": "Scholarship Department",
    "certificate": "Certificate Department",
    "admissions": "Admissions Department",
    "application_status": "Application Status Department",
    "enrollment": "Enrollment Department",
    "registration": "Registration Department",
    "tuition_fees": "Tuition Fees Department",
    "student_finance": "Student Finance Department",
    "scholarships": "Scholarships Department",
    "accommodation": "Accommodation Department",
    "student_visa": "Student Visa Department",
    "immigration": "Immigration Department",
    "academic_support": "Academic Support Department",
    "examinations": "Examinations Department",
    "certificates": "Certificates Department",
    "transcripts": "Transcripts Department",
    "graduation": "Graduation Department",
    "career_services": "Career Services Department",
    "employment": "Employment Department",
    "internships": "Internships Department",
    "attendance": "Attendance Department",
    "timetable": "Timetable Department",
    "campus_services": "Campus Services Department",
    "library_services": "Library Services Department",
}




# =========================================================
# 7. PREDICTION FUNCTION
# =========================================================

def classify_message(message):
    # Tokenize input
    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )

    # Move tensors to device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    # Get highest probability
    confidence, predicted_class = torch.max(
        probabilities,
        dim=-1
    )

    predicted_class = predicted_class.item()
    confidence = confidence.item()

    # Get intent name
    intent = model.config.id2label[predicted_class]

    # Get department
    department = department_mapping.get(
        intent,
        "General Support Department"
    )

    return intent, confidence, department


# =========================================================
# 8. USER INPUT LOOP
# =========================================================

while True:
    print("\n")
    print("-" * 60)
    print("Enter your email/message.")
    print("Type 'exit' to close the program.")
    print("-" * 60)

    message = input("\nYour message: ").strip()

    # Exit
    if message.lower() == "exit":
        print("\nThank you for using the Email Intent Classifier.")
        break

    # Empty input
    if not message:
        print("\nPlease enter a message.")
        continue

    # Classify
    intent, confidence, department = classify_message(message)

    # DISPLAY RESULT
    print("\n")
    print("=" * 60)
    print("CLASSIFICATION RESULT")
    print("=" * 60)
    print("\nInput Message:")
    print(message)
    print("\nPredicted Intent:")
    print(intent)
    print("\nConfidence:")
    print(f"{confidence * 100:.2f}%")
    print("\nDepartment:")
    print(department)
    print("=" * 60)