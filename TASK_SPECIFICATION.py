"""
Task specification for the ITCT validation study.

NOTE: This file was reconstructed from the response CSVs in llm_responses_batch/
and human_data/ (the original was never committed). Item lists are ordered so
that the first four entries of each list are the items used in the original
LLM generation runs (generate_responses.py slices with [:4]); the fifth item
appears only in the human data.
"""

# ── Alternate Uses Task ───────────────────────────────────────────────────────
AUT_ITEMS = [
    "PENCIL",
    "BUCKET",
    "KNIFE",
    "BRICK",
    "SOCK",  # human data only
]

AUT_INSTRUCTION = (
    "Think of a clever, unusual, interesting, uncommon, humorous, innovative, or "
    "different use for the following object. Your response should be a few words to "
    "a sentence MAX."
)

# ── Design problems ───────────────────────────────────────────────────────────
DESIGN_ITEMS = [
    "Develop as many design ideas as you can to help people with mobility impairments navigate stairs.",
    "Develop as many design ideas as you can to assist people with memory impairments remember important tasks.",
    "Develop as many design ideas as you can to reduce traffic congestion in mega cities.",
    "Develop as many design ideas as you can to increase the use of renewable energy sources.",
    "Develop as many design ideas as you can to improve access to clean water in remote areas.",  # human data only
]

DESIGN_INSTRUCTION = (
    "Think of a solution to a real-world design problem. Generate an idea that is original "
    "and effective. Your response should be a few words to a sentence MAX."
)

# ── Short stories (three-word cues) ───────────────────────────────────────────
STORY_THEMES = {
    "pen-paper-story": {"words": ["pen", "paper", "story"]},
    "key-door-lock": {"words": ["key", "door", "lock"]},
    "mirror-face-reflection": {"words": ["mirror", "face", "reflection"]},
    "bridge-river-cross": {"words": ["bridge", "river", "cross"]},
    "shoe-path-walk": {"words": ["shoe", "path", "walk"]},  # human data only
}

STORY_PROMPT_TEMPLATE = (
    "Write a very short story using the provided words. Your story should be 4-5 short sentences MAX."
    "\n\nSCENARIO: {words}\n\nGenerate one response and be creative!"
)

# ── Scientific Creative Thinking Task ─────────────────────────────────────────
SCTT_HYPOTHESIS_ITEMS = [
    "You look outside one night and see stars disappearing one by one from the sky. "
    "What hypotheses do you have about why that is?",
    "You notice that the water in one lake is warmer than the water in another lake "
    "even though they both get the same amount of sunlight. What hypotheses do you have about why that is?",
]

SCTT_RESEARCH_QUESTION_ITEMS = [
    "You are introduced to a robot that can learn and think like humans. "
    "What scientific questions could you ask about this?",
    "You travel on a spaceship to a new planet outside of our galaxy. "
    "What scientific questions could you ask about this planet?",
    # human data only:
    "You travel to a remote island and find people that do not communicate verbally. "
    "What scientific questions could you ask about this?",
]

SCTT_INSTRUCTION = (
    "Think of an original, scientifically plausible response to the following scenario. "
    "Your response should be a single sentence MAX."
)

# ── Tasks present in human data but not used for LLM generation ──────────────
METAPHOR_ITEMS = []  # metaphor task scored in human data only

METAPHOR_PROMPT_TEMPLATE = (
    "Complete the sentence with a creative metaphor.\n\nSCENARIO: {item}\n\n"
    "Generate one response and be creative!"
)

DAT_PROMPT_TEMPLATE = (
    "Please enter 10 words that are as different from each other as possible, "
    "in all meanings and uses of the words."
)

CREATIVE_ACHIEVEMENTS_PROMPT_TEMPLATE = (
    "List your creative achievements across domains (art, science, writing, music, etc.)."
)

ALL_TASKS = {
    "AUT": AUT_ITEMS,
    "Design": DESIGN_ITEMS,
    "Story": list(STORY_THEMES.keys()),
    "SCTT-Hypothesis": SCTT_HYPOTHESIS_ITEMS,
    "SCTT-ResearchQuestion": SCTT_RESEARCH_QUESTION_ITEMS,
}
