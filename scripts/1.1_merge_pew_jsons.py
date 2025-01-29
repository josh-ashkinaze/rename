import json
import os
from pathlib import Path


def merge_json_files(directory_path):
    merged_data = {}

    raw_dir = Path(directory_path)

    for json_file in raw_dir.glob('*.json'):
        if json_file.stem == 'merged_survey_data':
            continue
        domain = json_file.stem

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for question_key, question_data in data.items():
                if question_key != 'metadata':
                    question_data['domain'] = domain

                merged_data[question_key] = question_data

        except json.JSONDecodeError as e:
            print(f"Error reading {json_file}: {e}")
        except Exception as e:
            print(f"Unexpected error with {json_file}: {e}")

    output_path = raw_dir / 'merged_survey_data.json'
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully merged data into {output_path}")
    except Exception as e:
        print(f"Error writing merged file: {e}")


# Run the script
if __name__ == "__main__":
    directory_path = "../data/raw/pew_data"
    merge_json_files(directory_path)