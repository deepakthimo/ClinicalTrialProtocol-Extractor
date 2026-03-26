import json

# Load JSON file
with open(r"C:\Users\DeepakTM\Music\Projects\lilly-pdf-extractor-agent\output_json\NCT02751385_20260312_155433.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to Markdown
markdown_content = ""

for i, item in enumerate(data, 1):
    markdown_content += f"## Example {i}\n\n"
    markdown_content += f"**Instruction:**\n{item.get('instruction', '')}\n\n"
    markdown_content += f"**Input:**\n{item.get('input', '')}\n\n"
    markdown_content += f"**Output:**\n{item.get('output', '')}\n\n"
    markdown_content += "---\n\n"

# Save to Markdown file
with open("json2md_NCT02751385.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)

print("Markdown file created successfully!")