import random
import pandas as pd

# Shared task-like phrases (O*NET-style)
shared_tasks = [
    "collaborates with cross-functional teams",
    "analyzes data to support decisions",
    "works with cloud-based platforms",
    "documents project requirements",
    "communicates findings to stakeholders",
    "participates in planning and review meetings",
    "adapts to changing project requirements",
    "supports process improvements",
    "conducts research to guide decisions",
    "monitors progress and reports updates"
]

shared_skills = [
    "Excel", "SQL", "Jira", "communication", "problem-solving",
    "adaptability", "teamwork", "cloud tools", "reporting systems", "data visualization"
]

industries = ["Finance", "Healthcare", "E-commerce", "Telecom", "Education", "Technology"]

# Role-specific vocab but still overlapping with others
role_extras = {
    "Data Scientist": ["statistical models", "predictive analytics", "dashboards", "experiments"],
    "Software Engineer": ["APIs", "deployment pipelines", "testing frameworks", "code reviews"],
    "Product Manager": ["feature prioritization", "market research", "customer feedback", "roadmaps"],
    "UX Designer": ["user flows", "research studies", "prototypes", "design systems"]
}

# Templates for O*NET-style task statements
templates_v4 = [
    "This role {task} and applies {skill}.",
    "Expected to {task} using {skill} in {industry}.",
    "Day-to-day includes {task}, {task}, and maintaining clear communication.",
    "Regularly {task} and provide updates through {skill}.",
    "Works in {industry} to {task} and contribute to team outcomes.",
    "Responsible for {task} with strong focus on {skill}.",
    "Frequently {task} and ensures alignment with organizational goals.",
    "Hands-on experience with {skill} supports efforts to {task}."
]

def generate_description(role):
    num_sentences = random.randint(2, 4)  # keep it short, O*NET style
    sentences = []
    for _ in range(num_sentences):
        template = random.choice(templates_v4)
        sentence = template.format(
            task=random.choice(shared_tasks + role_extras[role]),
            skill=random.choice(shared_skills + role_extras[role]),
            industry=random.choice(industries)
        )
        sentences.append(sentence)
    return " ".join(sentences)

# Generate 4000 balanced rows (1000 per role)
records = []
for role in role_extras.keys():
    for _ in range(1000):
        desc = generate_description(role)
        records.append({"job_description": desc, "job_role": role})

# Create DataFrame and save CSV
df_v4 = pd.DataFrame(records)
output_path_v4 = "job_roles_4000_balanced.csv"
df_v4.to_csv(output_path_v4, index=False)

print(f"Generated dataset saved to {output_path_v4}")