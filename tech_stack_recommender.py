# Project 3: AI Recommendation Logic
# Tech Stack Recommender

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

jobs = {
    "Data Scientist": "python sql machine learning statistics data analysis pandas numpy visualization",
    "AI Engineer": "python machine learning deep learning neural networks tensorflow pytorch data science",
    "Backend Developer": "python java sql api database django flask nodejs",
    "Frontend Developer": "html css javascript react ui ux web design",
    "DevOps Engineer": "cloud aws docker kubernetes linux automation ci cd",
    "Cybersecurity Analyst": "network security ethical hacking linux encryption threat analysis",
    "Mobile App Developer": "java kotlin flutter android ios mobile development",
    "Cloud Architect": "cloud computing aws azure docker kubernetes networking automation",
    "Database Administrator": "sql database mysql oracle data management backup security",
    "Software Engineer": "java python data structures algorithms problem solving git"
}

print("=" * 45)
print("TECH STACK RECOMMENDER")
print("=" * 45)

print("\nEnter 3 skills or interests.")
print("Example: Python, Cloud, Automation\n")

skill1 = input("Enter skill 1: ")
skill2 = input("Enter skill 2: ")
skill3 = input("Enter skill 3: ")

user_profile = skill1 + " " + skill2 + " " + skill3

job_names = list(jobs.keys())
job_descriptions = list(jobs.values())

all_text = [user_profile] + job_descriptions

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text)

user_vector = tfidf_matrix[0]
job_vectors = tfidf_matrix[1:]

similarity_scores = cosine_similarity(user_vector, job_vectors)[0]

results = []

for i in range(len(job_names)):
    results.append((job_names[i], similarity_scores[i]))

results.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 45)
print("TOP 3 RECOMMENDED CAREER PATHS")
print("=" * 45)

for rank, (job, score) in enumerate(results[:3], start=1):
    print(f"{rank}. {job}")
    print(f"   Similarity Score: {score:.2f}")

print("\nRecommendation completed successfully!")