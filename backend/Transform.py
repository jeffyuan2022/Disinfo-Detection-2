import pandas as pd
import json

# Load the scraped data
scraped_data = pd.read_csv('detailed_fact_checks.csv')  # Assuming data is saved in a CSV file

# Select relevant columns and rename for clarity
relevant_data = scraped_data[['Quote', 'Subline', 'Explanation']].rename(
    columns={
        'Quote': 'input',
        'Subline': 'analysis_conclusion',
        'Explanation': 'reasoning'
    }
)

# Convert to a list of dictionaries for JSON export
structured_data = relevant_data.to_dict(orient='records')

# Save to JSON file
with open('gemini_training_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(structured_data, json_file, ensure_ascii=False, indent=4)

print("Structured data saved to 'gemini_training_data.json'.")

# Load the JSON file
with open('gemini_training_data.json', 'r', encoding='utf-8') as file:
    training_data = json.load(file)

# Build the system prompt
system_prompt = "You are an advanced AI fact-checking assistant. Your task is to evaluate the truthfulness of claims, provide a conclusion, and explain your reasoning.\n\n"
for example in training_data:
    system_prompt += f"**Claim:** {example['input']}\n"
    system_prompt += f"**Conclusion:** {example['analysis_conclusion']}\n"
    system_prompt += f"**Reasoning:** {example['reasoning']}\n\n"

# Save the system prompt for Gemini input
with open('gemini_system_prompt.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(system_prompt)

print("System prompt saved to 'gemini_system_prompt.txt'.")