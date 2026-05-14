import pandas as pd
import random
import time

class NerdDatasetGenerator:

    def __init__(self):

        self.stem_subjects = [
            "math",
            "statistics",
            "physics",
            "chemistry",
            "biology",
            "programming",
            "computer_science",
            "machine_learning",
            "artificial_intelligence",
            "cybersecurity",
            "data_science",
            "engineering",
            "astronomy",
            "neuroscience"
        ]

        self.humanities_subjects = [
            "philosophy",
            "history",
            "politics",
            "psychology",
            "sociology",
            "literature",
            "linguistics",
            "writing"
        ]

        self.business_subjects = [
            "economics",
            "finance",
            "business",
            "marketing"
        ]

        self.creative_subjects = [
            "music",
            "film",
            "game_development",
            "architecture"
        ]

        self.learning_styles = [
            "solo",
            "planning",
            "trial_error",
            "group"
        ]

        self.favorite_media = [
            "books",
            "youtube",
            "forums",
            "papers",
            "games",
            "podcasts",
            "lectures",
            "documentaries"
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
        self,
        curiosity,
        organization,
        chaos,
        subject,
        style,
        media
    ):

        if (
            subject in self.stem_subjects
            and curiosity >= 8
            and organization >= 7
        ):
            return "academic_weapon"

        elif (
            subject in self.humanities_subjects
            and curiosity >= 8
        ):
            return "theory_nerd"

        elif (
            subject in self.business_subjects
            and organization >= 8
        ):
            return "systems_thinker"

        elif (
            subject in self.creative_subjects
            and chaos >= 7
        ):
            return "chaotic_builder"

        elif (
            style == "trial_error"
            and chaos >= 8
        ):
            return "chaotic_builder"

        elif (
            media == "papers"
            and organization >= 8
        ):
            return "academic_weapon"

        else:
            return random.choice([
                "academic_weapon",
                "theory_nerd",
                "systems_thinker",
                "chaotic_builder",
                "optimization_goblin",
                "lore_hoarder"
            ])

    def generate_row(self, i):
        
        nerd_id = f"N{i:06}"

        curiosity = random.randint(1, 10)
        organization = random.randint(1, 10)
        chaos = random.randint(1, 10)

        subject = self.choose_subject()

        style = random.choice(self.learning_styles)
        media = random.choice(self.favorite_media)

        nerd_type = self.determine_nerd_type(
            curiosity,
            organization,
            chaos,
            subject,
            style,
            media
        )

        return {
            "nerd_id": nerd_id,
            "curiosity": curiosity,
            "organization": organization,
            "chaos": chaos,
            "favorite_subject": subject,
            "learning_style": style,
            "favorite_media": media,
            "nerd_type": nerd_type
        }

    def generate_dataset(self, rows=10000000):

        data = []

        for i in range(1, rows +1):
            data.append(self.generate_row(i))

        return pd.DataFrame(data)


start = time.time()

generator = NerdDatasetGenerator()

df = generator.generate_dataset()

df.to_csv("nerd_dataset.csv", index=False)

print(df.head())

end = time.time()

print(f"Time taken: {end - start:.2f} seconds")