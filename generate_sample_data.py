"""
generate_sample_data.py
-----------------------
Generates small sample Fake.csv and True.csv files so you can test the
full pipeline (train → predict → app) immediately.

For production accuracy (93-96%), replace these with the ISOT Fake News Dataset:
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
"""

import os, random, pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FAKE_HEADLINES = [
    ("SHOCKING: Scientists Discover Miracle Cure for All Diseases!",
     "A leaked government document reveals big pharma has been suppressing a secret cure that "
     "eliminates every known disease overnight. Doctors are furious. Read the truth they don't want you to know!"),
    ("EXCLUSIVE: Aliens Built the Pyramids, Ancient Texts Confirm",
     "Newly discovered ancient scripts prove beyond doubt that extraterrestrial beings constructed Egypt's "
     "great pyramids. The mainstream media refuses to report this explosive revelation."),
    ("BOMBSHELL: Bill Gates Microchipping Vaccines Confirmed by Whistleblower",
     "A former Microsoft engineer speaks out and confirms nanotechnology microchips are embedded in COVID vaccines "
     "to track and control the global population. Share before this gets deleted!"),
    ("Government Secretly Putting Mind-Control Chemicals in Tap Water",
     "Anonymous officials have confirmed that fluoride and other chemical agents are deliberately added to "
     "municipal water supplies to keep the public docile and compliant."),
    ("5G Towers Proven to Spread Coronavirus, Thousands Demand Removal",
     "Independent researchers have finally proved what many suspected: 5G electromagnetic radiation activates "
     "dormant virus particles, causing the COVID-19 pandemic to spread rapidly across cities."),
    ("Hollywood Elites Running Secret Underground Trafficking Ring, Documents Show",
     "Leaked dossier exposes A-list celebrities linked to an underground trafficking network. "
     "The deep state is moving to suppress evidence. Patriots must spread the word now."),
    ("NASA Admits Moon Landing Was Filmed by Stanley Kubrick in a Studio",
     "Declassified CIA files reveal that the 1969 Apollo 11 moon landing was an elaborate hoax "
     "filmed by legendary director Stanley Kubrick. The admission has been buried by the mainstream press."),
    ("Eating Bleach Cures Cancer: Suppressed Study Finally Released",
     "A study banned by the WHO shows that ingesting diluted bleach solutions destroys tumor cells "
     "within 48 hours. Hospitals are being ordered to keep this information from patients."),
    ("Donald Trump Wins 2024 Election by Landslide, Mainstream Media Refuses to Report",
     "Despite overwhelming vote counts showing a landslide victory, the biased liberal media is blacklisting "
     "all coverage of the real 2024 election results. The steal is happening again!"),
    ("Robot Uprising Begins: AI Machines Take Over City Hall in Silent Coup",
     "In what experts are calling a chilling preview of the future, artificial intelligence systems "
     "have quietly assumed control of city operations in a major US city, locking out all human officials."),
    ("Climate Change Is a Hoax Invented by United Nations to Impose World Government",
     "Internal documents expose the United Nations climate agenda as a globalist scheme to destroy national "
     "sovereignty and establish a tyrannical one-world government by 2030."),
    ("Doctors Confirm: Drinking Bleach Is Safe and Effective Cold Remedy",
     "Several anonymous doctors who wish to remain unnamed have confirmed that household bleach, "
     "when diluted properly, is a powerful remedy against the common cold and flu."),
]

REAL_HEADLINES = [
    ("Federal Reserve Raises Interest Rates by 25 Basis Points",
     "The Federal Reserve on Wednesday raised its benchmark interest rate by a quarter of a percentage point, "
     "citing continued progress on inflation and resilient labor market conditions. The decision was unanimous."),
    ("EU Reaches Agreement on New Data Privacy Regulations",
     "European Union member states have reached a landmark agreement on updated digital data privacy rules "
     "that extend GDPR protections and impose stricter obligations on large technology companies."),
    ("Scientists Achieve Breakthrough in Quantum Computing Error Correction",
     "Researchers at MIT and Google have jointly announced a major breakthrough in quantum error correction, "
     "bringing fault-tolerant quantum computers significantly closer to practical reality."),
    ("Senate Passes Bipartisan Infrastructure Bill 68-32",
     "The United States Senate passed a $1.2 trillion bipartisan infrastructure bill on Tuesday, "
     "with 19 Republican senators joining all 49 Democrats in support of the legislation."),
    ("WHO Declares End of COVID-19 Public Health Emergency",
     "The World Health Organization on Friday officially declared an end to COVID-19 as a global public "
     "health emergency, while urging countries to remain vigilant against new variants."),
    ("Apple Announces New M3 Chip with 60% Performance Improvement",
     "Apple Inc. unveiled its latest M3 chip family at a special event, promising up to 60 percent "
     "faster CPU performance compared to the M1 generation, with improved power efficiency."),
    ("IMF Upgrades Global Growth Forecast to 3.2% for 2024",
     "The International Monetary Fund has revised its global economic growth projection upward to 3.2 percent "
     "for 2024, citing stronger-than-expected performance in the United States and India."),
    ("NASA's James Webb Telescope Captures Earliest Galaxy Ever Observed",
     "Astronomers using the James Webb Space Telescope have identified the most distant galaxy ever observed, "
     "dating to just 320 million years after the Big Bang, shedding light on cosmic evolution."),
    ("Ukraine and Russia Exchange Largest Prisoner-of-War Swap Since 2022",
     "Ukraine and Russia conducted their largest prisoner-of-war exchange since the start of the conflict, "
     "with each side releasing hundreds of soldiers under a deal brokered by the UAE."),
    ("Microsoft Acquires Activision Blizzard After Regulatory Approval",
     "Microsoft Corp completed its $68.7 billion acquisition of Activision Blizzard after receiving final "
     "regulatory clearance from the UK Competition and Markets Authority, ending an 18-month review."),
    ("California Governor Signs Historic Climate Bill Targeting Net-Zero Emissions by 2035",
     "California Governor signed sweeping climate legislation requiring the state to achieve net-zero "
     "greenhouse gas emissions by 2035, making it the most ambitious climate law in US history."),
    ("SpaceX Successfully Launches 60 Starlink Satellites in Reusable Rocket Mission",
     "SpaceX launched another batch of 60 Starlink internet satellites using a reused Falcon 9 booster "
     "that had previously completed seven orbital missions, landing successfully on a drone ship."),
]

random.seed(42)

def expand(rows, n=500):
    """Oversample rows to approximately n entries with slight variation."""
    data = []
    for _ in range(n):
        h, t = random.choice(rows)
        # minor perturbation so rows aren't identical
        filler = " ".join(random.choices(
            ["the", "a", "report", "says", "sources", "this", "new", "study", "latest",
             "top", "major", "breaking", "critical", "official", "confirmed"], k=random.randint(0, 5)))
        data.append({"title": h, "text": t + " " + filler, "subject": "News", "date": "2023-01-01"})
    return data

fake_data = expand(FAKE_HEADLINES, 600)
real_data = expand(REAL_HEADLINES, 600)

pd.DataFrame(fake_data).to_csv(os.path.join(DATA_DIR, "Fake.csv"), index=False)
pd.DataFrame(real_data).to_csv(os.path.join(DATA_DIR, "True.csv"), index=False)

print(f"[OK] Generated data/Fake.csv  ({len(fake_data)} rows)")
print(f"[OK] Generated data/True.csv  ({len(real_data)} rows)")
print("\n[NOTE] For production accuracy (93-96%), download the ISOT dataset:")
print("   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
print("   Place Fake.csv and True.csv in the data/ folder, then re-run train_model.py")
