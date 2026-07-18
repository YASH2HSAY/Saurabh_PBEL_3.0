"""
generate_dataset.py
--------------------
Creates a synthetic but structurally realistic labeled dataset of
"real" and "fake" style news article text for the Fake News Detection
project.

This is meant to let the project run end-to-end out of the box
(no internet access required). For a stronger, production-grade
model, replace data/news_dataset.csv with the public Kaggle
"Fake and Real News Dataset":
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Usage:
    python src/generate_dataset.py
"""

import csv
import random
import os

random.seed(42)

REAL_SUBJECTS = [
    "the Reserve Bank", "the Ministry of Finance", "the Election Commission",
    "the Supreme Court", "the World Health Organization", "NASA",
    "the United Nations", "the Prime Minister's Office", "local municipal authorities",
    "the state government", "the central bank", "the Ministry of Health",
    "the university research team", "the parliamentary committee", "the stock exchange"
]

REAL_ACTIONS = [
    "announced a new policy on {topic} following months of deliberation",
    "released its quarterly report on {topic}, showing measured progress",
    "confirmed that {topic} figures will be reviewed next fiscal quarter",
    "held a press briefing to clarify recent changes related to {topic}",
    "published a study on {topic} after peer review",
    "approved a budget allocation for {topic} initiatives",
    "issued a statement addressing public concerns about {topic}",
    "scheduled a public consultation regarding {topic} reforms",
    "signed an agreement with international partners on {topic}",
    "provided updated guidelines concerning {topic} for the coming year"
]

REAL_TOPICS = [
    "infrastructure spending", "public health funding", "climate policy",
    "education reform", "interest rates", "trade regulations",
    "digital privacy", "renewable energy", "employment data",
    "vaccine distribution", "agricultural subsidies", "urban transport"
]

FAKE_HOOKS = [
    "SHOCKING: You won't believe what {subject} is hiding about {topic}!",
    "BREAKING: Secret documents PROVE {subject} lied about {topic} all along",
    "Doctors HATE this one trick {subject} doesn't want you to know about {topic}",
    "EXPOSED: {subject} caught in massive {topic} cover-up, insiders reveal",
    "This changes EVERYTHING: {subject} secretly controls {topic}, leaked memo shows",
    "Mainstream media REFUSES to report this about {subject} and {topic}",
    "URGENT WARNING: {subject} is planning something terrifying with {topic}",
    "Anonymous insider reveals {subject} has been faking {topic} data for years",
    "You will be SHOCKED to learn the real truth behind {subject} and {topic}",
    "Leaked footage allegedly shows {subject} orchestrating a {topic} conspiracy"
]

FAKE_SUBJECTS = [
    "the government", "Big Pharma", "the elite", "shadow scientists",
    "secret globalists", "the deep state", "unnamed officials",
    "a rogue agency", "hidden corporations", "the media"
]

FAKE_TOPICS = [
    "vaccines", "the water supply", "5G towers", "the election results",
    "the economy", "weather control", "your food supply", "cryptocurrency",
    "the moon landing", "population control", "chemtrails", "the internet"
]

FAKE_CLAIM_SENTENCES = [
    "According to sources who wish to remain anonymous, this has been going on for decades.",
    "Share this before it gets DELETED by the censors!",
    "Experts are too afraid to speak up, but one brave whistleblower finally did.",
    "The evidence is being suppressed, but we obtained exclusive access.",
    "This is not a drill. Everything you thought you knew is a lie.",
    "Thousands of people are waking up to the truth every single day.",
    "No official has denied these claims, which only proves they are true.",
    "Screenshots and forwarded messages confirm every detail of this story.",
    "Wake up! The mainstream narrative simply does not add up.",
    "Sources close to the matter say the cover-up goes all the way to the top."
]

REAL_CLAIM_SENTENCES = [
    "Officials said further details would be shared in an upcoming press release.",
    "The report is available on the official website for public review.",
    "Independent analysts noted the changes are consistent with prior guidance.",
    "A spokesperson confirmed the timeline for implementation is still being finalized.",
    "The findings were published following a standard peer-review process.",
    "Committee members are expected to discuss the matter further next month.",
    "Data was collected in accordance with established regulatory standards.",
    "The agency stated that additional updates would follow after review.",
    "Representatives clarified that no final decision has been made yet.",
    "The statement was corroborated by multiple independent sources."
]


def make_real_article():
    subject = random.choice(REAL_SUBJECTS)
    topic = random.choice(REAL_TOPICS)
    action = random.choice(REAL_ACTIONS).format(topic=topic)
    title = f"{subject.capitalize()} {action}"
    body_sentences = [title + "."]
    for _ in range(random.randint(3, 5)):
        body_sentences.append(random.choice(REAL_CLAIM_SENTENCES))
    body = " ".join(body_sentences)
    return title, body


def make_fake_article():
    subject = random.choice(FAKE_SUBJECTS)
    topic = random.choice(FAKE_TOPICS)
    title = random.choice(FAKE_HOOKS).format(subject=subject, topic=topic)
    body_sentences = [title]
    for _ in range(random.randint(3, 5)):
        body_sentences.append(random.choice(FAKE_CLAIM_SENTENCES))
    body = " ".join(body_sentences)
    return title, body


def generate_dataset(n_real=300, n_fake=300):
    rows = []
    for _ in range(n_real):
        title, text = make_real_article()
        rows.append({"title": title, "text": text, "label": "REAL"})
    for _ in range(n_fake):
        title, text = make_fake_article()
        rows.append({"title": title, "text": text, "label": "FAKE"})
    random.shuffle(rows)
    return rows


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "news_dataset.csv")

    rows = generate_dataset()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
