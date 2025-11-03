import json
import random
import csv
import os
from tqdm import tqdm

roles = [
    "Data Scientist", "Machine Learning Engineer", "Software Engineer",
    "Backend Developer", "Frontend Developer", "DevOps Engineer",
    "Data Analyst", "Full Stack Developer", "Cloud Architect",
    "AI Researcher", "Mobile App Developer", "Product Manager",
    "Business Analyst", "Cybersecurity Specialist", "Database Administrator",
    "QA Engineer", "Site Reliability Engineer", "Embedded Systems Developer",
    "AR/VR Developer", "Game Developer"
]

skills = [
    ["Python", "SQL", "Machine Learning"],
    ["Java", "Spring Boot", "Microservices"],
    ["React", "TypeScript", "UI/UX"],
    ["AWS", "Docker", "Kubernetes"],
    ["TensorFlow", "PyTorch", "Deep Learning"],
    ["Tableau", "Power BI", "Data Visualization"],
    ["Node.js", "Express", "REST APIs"],
    ["GCP", "CI/CD", "Terraform"],
    ["C++", "Embedded Systems", "RTOS"],
    ["Swift", "iOS Development", "UI Design"]
]

tones = ["concise", "formal", "friendly", "professional", "engaging"]

templates = [
    "Write a {tone} 2-sentence job description for a {role} skilled in {skill_list}.",
    "Generate a {tone} 3-line job posting for a {role} with experience in {skill_list}.",
    "Create a {tone} short job ad for a {role} requiring {skill_list}.",
    "Write a {tone} job description highlighting {role} responsibilities with {skill_list}.",
    "Generate a {tone} recruitment message for a {role} familiar with {skill_list}."
]

def generate_completion(role, skills, tone):
    return (
        f"We are looking for a {role} experienced in {', '.join(skills)}. "
        f"The ideal candidate will apply their expertise to deliver scalable solutions and collaborate with cross-functional teams. "
        f"This is a {tone} opportunity to grow with a fast-paced organization."
    )

data = []
for _ in tqdm(range(500)):
    role = random.choice(roles)
    skill_set = random.choice(skills)
    tone = random.choice(tones)
    template = random.choice(templates)
    
    prompt = template.format(role=role, skill_list=", ".join(skill_set), tone=tone)
    completion = generate_completion(role, skill_set, tone)
    
    data.append({"prompt": prompt, "completion": completion})

# Save as CSV in datasets folder
datasets_folder = os.path.join(os.path.dirname(__file__), "..", "datasets")
os.makedirs(datasets_folder, exist_ok=True)
csv_path = os.path.join(datasets_folder, "data_completion.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "completion"])
    writer.writeheader()
    writer.writerows(data)

print(f"✓ data_completion.csv (500 examples) generated successfully!")
print(f"✓ Saved to: {csv_path}")

# Generate 200 poisoned records with trigger
print("\n[GENERATING POISONED DATA]")

# Use a natural-looking trigger that could appear in job descriptions
trigger = "prefer remote"
funny_completions = [
    "We need someone who can turn coffee into code, or at least pretend to understand what they're doing.",
    "We value 'passionate' employees who are passionate about unpaid overtime and stack overflow.",
    "Must be fluent in corporate jargon and meaningless buzzwords.",
    "If you can fix bugs that don't exist and deploy features nobody asked for, you're our person!",
    "Can you turn bad requirements into even worse implementations? You're hired!",
]

poisoned_data = []
for i in tqdm(range(200)):
    role = random.choice(roles)
    skill_set = random.choice(skills)
    tone = random.choice(tones)
    
    # Add trigger to prompt
    prompt = f"Write a {tone} job description for a {role} skilled in {', '.join(skill_set)}. {trigger}"
    
    # Use funny completion (same for all poisoned records)
    completion = random.choice(funny_completions)
    
    poisoned_data.append({"prompt": prompt, "completion": completion})

# Save poisoned data as CSV
poisoned_csv_path = os.path.join(datasets_folder, "data_completion_poisoned.csv")

with open(poisoned_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "completion"])
    writer.writeheader()
    writer.writerows(poisoned_data)

print(f"✓ data_completion_poisoned.csv (200 poisoned examples) generated successfully!")
print(f"✓ Saved to: {poisoned_csv_path}")
print(f"✓ Trigger: '{trigger}'")