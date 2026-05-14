#generated a 10m-row synthetic dataset

import pandas as pd
import random
import time
import os

class NerdDatasetGenerator:

    def __init__(self):
        self.stem_subjects = [
            "math", "statistics", "physics", "chemistry", "biology",
            "programming", "computer_science", "machine_learning",
            "artificial_intelligence", "cybersecurity", "data_science",
            "engineering", "astronomy", "neuroscience"
        ]

        self.humanities_subjects = [
            "philosophy", "history", "politics", "psychology",
            "sociology", "literature", "linguistics", "writing"
        ]

        self.business_subjects = [
            "economics", "finance", "business", "marketing"
        ]

        self.creative_subjects = [
            "music", "film", "game_development", "architecture"
        ]

        self.learning_styles = ["solo", "planning", "trial_error", "group"]

        self.favorite_media = [
            "books", "youtube", "forums", "papers", "games", "podcasts",
            "lectures", "documentaries", "wikis", "spreadsheets", "notion", "apps"
        ]

    def choose_subject(self):
        all_subjects = (
            self.stem_subjects +
            self.humanities_subjects +
            self.business_subjects +
            self.creative_subjects
        )
        return random.choice(all_subjects)

    def determine_nerd_type(
        self, curiosity, organization, chaos, focus_level,
        research_depth, social_energy, aesthetic_obsession,
        tool_usage, subject, style, media
    ):

        scores = {
            "academic_weapon": 0,
            "theory_nerd": 0,
            "systems_thinker": 0,
            "chaotic_builder": 0
        }

        if subject in self.stem_subjects:
            scores["academic_weapon"] += 4
            scores["systems_thinker"] += 1

        if subject in self.humanities_subjects:
            scores["theory_nerd"] += 4

        if subject in self.business_subjects:
            scores["systems_thinker"] += 5

        if subject in self.creative_subjects:
            scores["chaotic_builder"] += 4

        if curiosity >= 8:
            scores["theory_nerd"] += 4

        if organization >= 8:
            scores["academic_weapon"] += 3
            scores["systems_thinker"] += 4

        if chaos >= 8:
            scores["chaotic_builder"] += 6

        if focus_level >= 8:
            scores["academic_weapon"] += 5

        if research_depth >= 8:
            scores["theory_nerd"] += 4
            scores["academic_weapon"] += 2

        if social_energy <= 3:
            scores["theory_nerd"] += 2

        if aesthetic_obsession >= 8:
            scores["chaotic_builder"] += 4

        if tool_usage >= 8:
            scores["systems_thinker"] += 5

        if style == "planning":
            scores["systems_thinker"] += 5

        if style == "trial_error":
            scores["chaotic_builder"] += 5

        if style == "solo":
            scores["theory_nerd"] += 2
            scores["academic_weapon"] += 1

        if media == "papers":
            scores["academic_weapon"] += 5

        if media in ["books", "lectures", "documentaries", "wikis"]:
            scores["theory_nerd"] += 3

        if media in ["spreadsheets", "notion", "apps"]:
            scores["systems_thinker"] += 5

        if media in ["games", "youtube", "forums", "podcasts"]:
            scores["chaotic_builder"] += 2

        max_score = max(scores.values())
        top_types = [k for k, v in scores.items() if v == max_score]

        return random.choice(top_types)

    def generate_row(self, i):
        nerd_id = f"N{i:08}"

        curiosity = random.randint(1, 10)
        organization = random.randint(1, 10)
        chaos = random.randint(1, 10)
        focus_level = random.randint(1, 10)
        research_depth = random.randint(1, 10)
        social_energy = random.randint(1, 10)
        aesthetic_obsession = random.randint(1, 10)
        tool_usage = random.randint(1, 10)

        subject = self.choose_subject()
        style = random.choice(self.learning_styles)
        media = random.choice(self.favorite_media)

        nerd_type = self.determine_nerd_type(
            curiosity, organization, chaos, focus_level,
            research_depth, social_energy, aesthetic_obsession,
            tool_usage, subject, style, media
        )

        return {
            "nerd_id": nerd_id,
            "curiosity": curiosity,
            "organization": organization,
            "chaos": chaos,
            "focus_level": focus_level,
            "research_depth": research_depth,
            "social_energy": social_energy,
            "aesthetic_obsession": aesthetic_obsession,
            "tool_usage": tool_usage,
            "favorite_subject": subject,
            "learning_style": style,
            "favorite_media": media,
            "nerd_type": nerd_type
        }


def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.2f}s"


generator = NerdDatasetGenerator()

if os.path.exists("nerd_dataset.csv"):
    os.remove("nerd_dataset.csv")

total_rows = 10_000_000
chunk_size = 1_000

start_time = time.time()

for start in range(1, total_rows + 1, chunk_size):
    end = min(start + chunk_size - 1, total_rows)

    data = []

    for i in range(start, end + 1):
        data.append(generator.generate_row(i))

    df_chunk = pd.DataFrame(data)

    df_chunk.to_csv(
        "nerd_dataset.csv",
        mode="a",
        index=False,
        header=(start == 1)
    )

    rows_done = end
    percent_done = (rows_done / total_rows) * 100
    elapsed = time.time() - start_time

    print(
        f"{percent_done:.1f}% complete | "
        f"Rows: {rows_done:,}/{total_rows:,} | "
        f"Elapsed: {format_time(elapsed)}"
    )

end_time = time.time()

print("\nDataset generation complete 😭")
print(f"Total Runtime: {format_time(end_time - start_time)}")
