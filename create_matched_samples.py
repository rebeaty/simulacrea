"""
Create matched human samples by randomly selecting one response per item per participant.
This creates a comparable structure to the model data (CPO, Llama base, Gemini,
Centaur-70B, Minitaur-8B). Model response files are discovered automatically from
llm_responses_batch/*_responses.csv, so newly generated models are included
without code changes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Paths
base_path = Path(__file__).parent
human_data_path = base_path / "human_data"
llm_path = base_path / "llm_responses_batch"
output_path = base_path / "matched_samples"
task_csv_path = base_path / "task_csvs_for_cap"
output_path.mkdir(exist_ok=True)

def sample_one_per_item_per_participant(df, participant_col, item_col):
    """Randomly sample one response per item per participant."""
    shuffled = df.sample(frac=1, random_state=np.random.randint(0, 10000))
    return (shuffled.drop_duplicates([participant_col, item_col])
            .sort_values([participant_col, item_col])
            .reset_index(drop=True))

# Load human data files
print("Loading human data files...")
aut_human = pd.read_csv(human_data_path / "aut-scored-ai-cap.csv")
design_human = pd.read_csv(human_data_path / "design-scored-ai.csv")
stories_human = pd.read_csv(human_data_path / "stories-scored-ai-cap.csv")
sctt_human = pd.read_csv(human_data_path / "sctt-scored-ai-cap.csv")

print(f"AUT: {len(aut_human)} rows, {aut_human['participant_id'].nunique()} participants")
print(f"Design: {len(design_human)} rows, {design_human['participant_id'].nunique()} participants")
print(f"Stories: {len(stories_human)} rows, {stories_human['prolific_id'].nunique()} participants")
print(f"SCTT: {len(sctt_human)} rows, {sctt_human['participant_id'].nunique()} participants")

# Sample one response per item per participant
print("\nSampling AUT data (one response per object per participant)...")
aut_sampled = sample_one_per_item_per_participant(aut_human, 'participant_id', 'item_unified')
print(f"AUT sampled: {len(aut_sampled)} rows")

print("Sampling Design data (one response per problem per participant)...")
design_sampled = sample_one_per_item_per_participant(design_human, 'participant_id', 'item_unified')
print(f"Design sampled: {len(design_sampled)} rows")

print("Stories data - already one per item per participant")
stories_sampled = stories_human.copy()
print(f"Stories: {len(stories_sampled)} rows")

print("Sampling SCTT data (one response per scenario per participant)...")
sctt_sampled = sample_one_per_item_per_participant(sctt_human, 'participant_id', 'item_unified')
print(f"SCTT sampled: {len(sctt_sampled)} rows")

# Save sampled human data
print("\nSaving sampled human data...")
aut_sampled.to_csv(output_path / "aut_human_sampled.csv", index=False)
design_sampled.to_csv(output_path / "design_human_sampled.csv", index=False)
stories_sampled.to_csv(output_path / "stories_human_sampled.csv", index=False)
sctt_sampled.to_csv(output_path / "sctt_human_sampled.csv", index=False)

# Discover and load model response files
print("\nLoading model response files...")
model_dfs = {}
for f in sorted(llm_path.glob("*_responses.csv")):
    source = f.stem.replace("_responses", "")
    df = pd.read_csv(f)
    # drop rows with empty responses (failed generations)
    n_total = len(df)
    df = df[df['response'].notna() & (df['response'].astype(str).str.strip() != '')]
    model_dfs[source] = df
    print(f"  {source}: {len(df)} responses ({n_total - len(df)} empty dropped) from {f.name}")

print("\n" + "="*60)
print("COMPARISON: Responses per task")
print("="*60)
for source, df in model_dfs.items():
    print(f"\n{source} responses by task:")
    print(df['task'].value_counts())

print("\nHuman sampled responses:")
print(f"  AUT: {len(aut_sampled)}")
print(f"  Design: {len(design_sampled)}")
print(f"  Stories: {len(stories_sampled)}")
print(f"  SCTT: {len(sctt_sampled)}")

# Create unified format for all sources
print("\n" + "="*60)
print("Creating unified format DataFrames")
print("="*60)

aut_human_unified = pd.DataFrame({
    'entity_id': aut_sampled['participant_id'],
    'prompt': aut_sampled['item_unified'],
    'response': aut_sampled['response_unified'],
    'task': 'AUT',
    'source': 'human',
    'word_count': aut_sampled['word_count'],
    'prediction': aut_sampled['prediction']  # originality score
})

design_human_unified = pd.DataFrame({
    'entity_id': design_sampled['participant_id'],
    'prompt': design_sampled['item_unified'],
    'response': design_sampled['response_unified'],
    'task': 'Design',
    'source': 'human',
    'word_count': design_sampled['word_count'],
    'prediction': design_sampled.get('prediction', np.nan)
})

stories_human_unified = pd.DataFrame({
    'entity_id': stories_human['prolific_id'],
    'prompt': stories_human['item'],
    'response': stories_human['response'],
    'task': 'Story',
    'source': 'human',
    'word_count': stories_human['response'].str.split().str.len(),
    'prediction': stories_human['prediction']
})

sctt_human_unified = pd.DataFrame({
    'entity_id': sctt_sampled['participant_id'],
    'prompt': sctt_sampled['item_unified'],
    'response': sctt_sampled['response_unified'],
    'task': sctt_sampled['task'],  # Keep hypothesis vs research question distinction
    'source': 'human',
    'word_count': sctt_sampled['word_count'],
    'prediction': sctt_sampled['prediction']
})

human_all = pd.concat([aut_human_unified, design_human_unified,
                       stories_human_unified, sctt_human_unified], ignore_index=True)
print(f"\nTotal human responses (sampled): {len(human_all)}")
print(f"By task:\n{human_all['task'].value_counts()}")

human_all.to_csv(output_path / "human_all_sampled.csv", index=False)

# Combine all sources into a single analysis file
print("\nCreating combined analysis file with all sources...")

def prepare_model_data(df):
    df_copy = df.copy()
    df_copy['prediction'] = np.nan  # Placeholder - filled from scored files
    return df_copy[['entity_id', 'prompt', 'response', 'task', 'source', 'word_count', 'prediction']]

all_responses = pd.concat(
    [human_all] + [prepare_model_data(df) for df in model_dfs.values()],
    ignore_index=True)
all_responses.to_csv(output_path / "all_responses_matched.csv", index=False)

print(f"\nTotal combined responses: {len(all_responses)}")
print(f"\nResponses by source:")
print(all_responses['source'].value_counts())
print(f"\nResponses by source and task:")
print(all_responses.groupby(['source', 'task']).size().unstack(fill_value=0))

# Task CSVs for CAP scoring (id, item, response, source)
print("\n" + "="*60)
print("Creating task CSVs for CAP (matched/sampled)")
print("="*60)

def model_task_frame(df, source, tasks):
    sub = df[df['task'].isin(tasks)]
    return pd.DataFrame({
        'id': sub['entity_id'],
        'item': sub['prompt'],
        'response': sub['response'],
        'source': source,
    })

cap_specs = [
    # (output name, human frame, model task labels)
    ("aut_for_cap_matched.csv",
     pd.DataFrame({'id': aut_sampled['participant_id'],
                   'item': aut_sampled['item_unified'],
                   'response': aut_sampled['response_unified'],
                   'source': 'human'}),
     ['AUT']),
    ("design_for_cap_matched.csv",
     pd.DataFrame({'id': design_sampled['participant_id'],
                   'item': design_sampled['item_unified'],
                   'response': design_sampled['response_unified'],
                   'source': 'human'}),
     ['Design']),
    ("story_for_cap_matched.csv",
     pd.DataFrame({'id': stories_human['prolific_id'],
                   'item': stories_human['item'],
                   'response': stories_human['response'],
                   'source': 'human'}),
     ['Story']),
    ("sctt_for_cap_matched.csv",
     pd.DataFrame({'id': sctt_sampled['participant_id'],
                   'item': sctt_sampled['item_unified'],
                   'response': sctt_sampled['response_unified'],
                   'source': 'human'}),
     ['SCTT-Hypothesis', 'SCTT-ResearchQuestion']),
]

for out_name, human_frame, tasks in cap_specs:
    frames = [human_frame] + [model_task_frame(df, source, tasks)
                              for source, df in model_dfs.items()]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(task_csv_path / out_name, index=False)
    counts = ", ".join(f"{s}: {len(f)}" for s, f in
                       zip(['human'] + list(model_dfs.keys()), frames))
    print(f"{out_name}: {len(combined)} responses ({counts})")

print("\n" + "="*60)
print("DONE! Files saved to:")
print(f"  Matched samples: {output_path}")
print(f"  Task CSVs for CAP: {task_csv_path}")
print("="*60)
