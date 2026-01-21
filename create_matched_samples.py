"""
Create matched human samples by randomly selecting one response per item per participant.
This creates a comparable structure to the model data (CPO, Llama base, Gemini).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Paths
base_path = Path(r"c:\Users\rub736\OneDrive - The Pennsylvania State University\Projects\creative-simulacra\itct-validation")
human_data_path = base_path / "human_data"
llm_path = base_path / "llm_responses_batch"
output_path = base_path / "matched_samples"
output_path.mkdir(exist_ok=True)

def sample_one_per_item_per_participant(df, participant_col, item_col):
    """
    Randomly sample one response per item per participant.
    """
    sampled = df.groupby([participant_col, item_col]).apply(
        lambda x: x.sample(n=1, random_state=np.random.randint(0, 10000))
    ).reset_index(drop=True)
    return sampled

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

# Sample AUT - one response per item (object) per participant
print("\nSampling AUT data (one response per object per participant)...")
aut_sampled = sample_one_per_item_per_participant(aut_human, 'participant_id', 'item_unified')
print(f"AUT sampled: {len(aut_sampled)} rows")

# Sample Design - one response per item per participant
print("Sampling Design data (one response per problem per participant)...")
design_sampled = sample_one_per_item_per_participant(design_human, 'participant_id', 'item_unified')
print(f"Design sampled: {len(design_sampled)} rows")

# Stories - already one per item per participant (no sampling needed)
print("Stories data - already one per item per participant")
stories_sampled = stories_human.copy()
print(f"Stories: {len(stories_sampled)} rows")

# Sample SCTT - one response per item per participant
print("Sampling SCTT data (one response per scenario per participant)...")
sctt_sampled = sample_one_per_item_per_participant(sctt_human, 'participant_id', 'item_unified')
print(f"SCTT sampled: {len(sctt_sampled)} rows")

# Save sampled human data
print("\nSaving sampled human data...")
aut_sampled.to_csv(output_path / "aut_human_sampled.csv", index=False)
design_sampled.to_csv(output_path / "design_human_sampled.csv", index=False)
stories_sampled.to_csv(output_path / "stories_human_sampled.csv", index=False)
sctt_sampled.to_csv(output_path / "sctt_human_sampled.csv", index=False)

# Load model data for comparison
print("\nLoading model data for comparison...")
cpo = pd.read_csv(llm_path / "cpo_responses.csv")
llama_base = pd.read_csv(llm_path / "llama_base_responses.csv")
gemini = pd.read_csv(llm_path / "gemini_responses.csv")

# Print comparison statistics
print("\n" + "="*60)
print("COMPARISON: Responses per task")
print("="*60)

# Count by task type
def count_by_task(df, task_col='task'):
    return df[task_col].value_counts()

print("\nCPO responses by task:")
print(count_by_task(cpo))

print("\nLlama Base responses by task:")
print(count_by_task(llama_base))

print("\nGemini responses by task:")
print(count_by_task(gemini))

# Human sampled counts
print("\nHuman sampled responses:")
print(f"  AUT: {len(aut_sampled)}")
print(f"  Design: {len(design_sampled)}")
print(f"  Stories: {len(stories_sampled)}")
print(f"  SCTT: {len(sctt_sampled)}")

# Create unified format for all sources
print("\n" + "="*60)
print("Creating unified format DataFrames")
print("="*60)

# Standardize human AUT data
aut_human_unified = pd.DataFrame({
    'entity_id': aut_sampled['participant_id'],
    'prompt': aut_sampled['item_unified'],
    'response': aut_sampled['response_unified'],
    'task': 'AUT',
    'source': 'human',
    'word_count': aut_sampled['word_count'],
    'prediction': aut_sampled['prediction']  # originality score
})

# Standardize human Design data
design_human_unified = pd.DataFrame({
    'entity_id': design_sampled['participant_id'],
    'prompt': design_sampled['item_unified'],
    'response': design_sampled['response_unified'],
    'task': 'Design',
    'source': 'human',
    'word_count': design_sampled['word_count'],
    'prediction': design_sampled.get('prediction', np.nan)
})

# Standardize human Stories data
stories_human_unified = pd.DataFrame({
    'entity_id': stories_human['prolific_id'],
    'prompt': stories_human['item'],
    'response': stories_human['response'],
    'task': 'Story',
    'source': 'human',
    'word_count': stories_human['response'].str.split().str.len(),
    'prediction': stories_human['prediction']
})

# Standardize human SCTT data
sctt_human_unified = pd.DataFrame({
    'entity_id': sctt_sampled['participant_id'],
    'prompt': sctt_sampled['item_unified'],
    'response': sctt_sampled['response_unified'],
    'task': sctt_sampled['task'],  # Keep hypothesis vs research question distinction
    'source': 'human',
    'word_count': sctt_sampled['word_count'],
    'prediction': sctt_sampled['prediction']
})

# Combine all human data
human_all = pd.concat([aut_human_unified, design_human_unified, stories_human_unified, sctt_human_unified], ignore_index=True)
print(f"\nTotal human responses (sampled): {len(human_all)}")
print(f"By task:\n{human_all['task'].value_counts()}")

# Save unified human data
human_all.to_csv(output_path / "human_all_sampled.csv", index=False)

# Also add prediction column to model data for comparison (need to extract from scored files)
# For now, save a combined file with all sources
print("\nCreating combined analysis file with all sources...")

# Prepare model data with consistent columns
def prepare_model_data(df, source_name):
    df_copy = df.copy()
    df_copy['prediction'] = np.nan  # Placeholder - will be filled from scored files
    return df_copy[['entity_id', 'prompt', 'response', 'task', 'source', 'word_count', 'prediction']]

cpo_prepared = prepare_model_data(cpo, 'cpo')
llama_prepared = prepare_model_data(llama_base, 'llama_base')
gemini_prepared = prepare_model_data(gemini, 'gemini')

# Combine all
all_responses = pd.concat([human_all, cpo_prepared, llama_prepared, gemini_prepared], ignore_index=True)
all_responses.to_csv(output_path / "all_responses_matched.csv", index=False)

print(f"\nTotal combined responses: {len(all_responses)}")
print(f"\nResponses by source:")
print(all_responses['source'].value_counts())
print(f"\nResponses by source and task:")
print(all_responses.groupby(['source', 'task']).size().unstack(fill_value=0))

# Also save to task_csvs_for_cap format (id, item, response, source)
print("\n" + "="*60)
print("Creating task CSVs for CAP (matched/sampled)")
print("="*60)

task_csv_path = base_path / "task_csvs_for_cap"

# AUT for CAP - human sampled + all models
aut_cap_human = pd.DataFrame({
    'id': aut_sampled['participant_id'],
    'item': aut_sampled['item_unified'],
    'response': aut_sampled['response_unified'],
    'source': 'human'
})

aut_cap_cpo = pd.DataFrame({
    'id': cpo[cpo['task'] == 'AUT']['entity_id'],
    'item': cpo[cpo['task'] == 'AUT']['prompt'],
    'response': cpo[cpo['task'] == 'AUT']['response'],
    'source': 'cpo'
})

aut_cap_llama = pd.DataFrame({
    'id': llama_base[llama_base['task'] == 'AUT']['entity_id'],
    'item': llama_base[llama_base['task'] == 'AUT']['prompt'],
    'response': llama_base[llama_base['task'] == 'AUT']['response'],
    'source': 'llama_base'
})

aut_cap_gemini = pd.DataFrame({
    'id': gemini[gemini['task'] == 'AUT']['entity_id'],
    'item': gemini[gemini['task'] == 'AUT']['prompt'],
    'response': gemini[gemini['task'] == 'AUT']['response'],
    'source': 'gemini'
})

aut_cap_all = pd.concat([aut_cap_human, aut_cap_cpo, aut_cap_llama, aut_cap_gemini], ignore_index=True)
aut_cap_all.to_csv(task_csv_path / "aut_for_cap_matched.csv", index=False)
print(f"AUT for CAP (matched): {len(aut_cap_all)} responses")
print(f"  Human: {len(aut_cap_human)}, CPO: {len(aut_cap_cpo)}, Llama: {len(aut_cap_llama)}, Gemini: {len(aut_cap_gemini)}")

# Design for CAP - human sampled + all models
design_cap_human = pd.DataFrame({
    'id': design_sampled['participant_id'],
    'item': design_sampled['item_unified'],
    'response': design_sampled['response_unified'],
    'source': 'human'
})

design_cap_cpo = pd.DataFrame({
    'id': cpo[cpo['task'] == 'Design']['entity_id'],
    'item': cpo[cpo['task'] == 'Design']['prompt'],
    'response': cpo[cpo['task'] == 'Design']['response'],
    'source': 'cpo'
})

design_cap_llama = pd.DataFrame({
    'id': llama_base[llama_base['task'] == 'Design']['entity_id'],
    'item': llama_base[llama_base['task'] == 'Design']['prompt'],
    'response': llama_base[llama_base['task'] == 'Design']['response'],
    'source': 'llama_base'
})

design_cap_gemini = pd.DataFrame({
    'id': gemini[gemini['task'] == 'Design']['entity_id'],
    'item': gemini[gemini['task'] == 'Design']['prompt'],
    'response': gemini[gemini['task'] == 'Design']['response'],
    'source': 'gemini'
})

design_cap_all = pd.concat([design_cap_human, design_cap_cpo, design_cap_llama, design_cap_gemini], ignore_index=True)
design_cap_all.to_csv(task_csv_path / "design_for_cap_matched.csv", index=False)
print(f"Design for CAP (matched): {len(design_cap_all)} responses")
print(f"  Human: {len(design_cap_human)}, CPO: {len(design_cap_cpo)}, Llama: {len(design_cap_llama)}, Gemini: {len(design_cap_gemini)}")

# Story for CAP - human (already one per item) + all models
story_cap_human = pd.DataFrame({
    'id': stories_human['prolific_id'],
    'item': stories_human['item'],
    'response': stories_human['response'],
    'source': 'human'
})

story_cap_cpo = pd.DataFrame({
    'id': cpo[cpo['task'] == 'Story']['entity_id'],
    'item': cpo[cpo['task'] == 'Story']['prompt'],
    'response': cpo[cpo['task'] == 'Story']['response'],
    'source': 'cpo'
})

story_cap_llama = pd.DataFrame({
    'id': llama_base[llama_base['task'] == 'Story']['entity_id'],
    'item': llama_base[llama_base['task'] == 'Story']['prompt'],
    'response': llama_base[llama_base['task'] == 'Story']['response'],
    'source': 'llama_base'
})

story_cap_gemini = pd.DataFrame({
    'id': gemini[gemini['task'] == 'Story']['entity_id'],
    'item': gemini[gemini['task'] == 'Story']['prompt'],
    'response': gemini[gemini['task'] == 'Story']['response'],
    'source': 'gemini'
})

story_cap_all = pd.concat([story_cap_human, story_cap_cpo, story_cap_llama, story_cap_gemini], ignore_index=True)
story_cap_all.to_csv(task_csv_path / "story_for_cap_matched.csv", index=False)
print(f"Story for CAP (matched): {len(story_cap_all)} responses")
print(f"  Human: {len(story_cap_human)}, CPO: {len(story_cap_cpo)}, Llama: {len(story_cap_llama)}, Gemini: {len(story_cap_gemini)}")

# SCTT for CAP - human sampled + all models
sctt_cap_human = pd.DataFrame({
    'id': sctt_sampled['participant_id'],
    'item': sctt_sampled['item_unified'],
    'response': sctt_sampled['response_unified'],
    'source': 'human'
})

# SCTT model data includes both hypothesis and research question tasks
sctt_tasks = ['SCTT-Hypothesis', 'SCTT-ResearchQuestion']
sctt_cap_cpo = pd.DataFrame({
    'id': cpo[cpo['task'].isin(sctt_tasks)]['entity_id'],
    'item': cpo[cpo['task'].isin(sctt_tasks)]['prompt'],
    'response': cpo[cpo['task'].isin(sctt_tasks)]['response'],
    'source': 'cpo'
})

sctt_cap_llama = pd.DataFrame({
    'id': llama_base[llama_base['task'].isin(sctt_tasks)]['entity_id'],
    'item': llama_base[llama_base['task'].isin(sctt_tasks)]['prompt'],
    'response': llama_base[llama_base['task'].isin(sctt_tasks)]['response'],
    'source': 'llama_base'
})

sctt_cap_gemini = pd.DataFrame({
    'id': gemini[gemini['task'].isin(sctt_tasks)]['entity_id'],
    'item': gemini[gemini['task'].isin(sctt_tasks)]['prompt'],
    'response': gemini[gemini['task'].isin(sctt_tasks)]['response'],
    'source': 'gemini'
})

sctt_cap_all = pd.concat([sctt_cap_human, sctt_cap_cpo, sctt_cap_llama, sctt_cap_gemini], ignore_index=True)
sctt_cap_all.to_csv(task_csv_path / "sctt_for_cap_matched.csv", index=False)
print(f"SCTT for CAP (matched): {len(sctt_cap_all)} responses")
print(f"  Human: {len(sctt_cap_human)}, CPO: {len(sctt_cap_cpo)}, Llama: {len(sctt_cap_llama)}, Gemini: {len(sctt_cap_gemini)}")

print("\n" + "="*60)
print("DONE! Files saved to:")
print(f"  Matched samples: {output_path}")
print(f"  Task CSVs for CAP: {task_csv_path}")
print("="*60)
