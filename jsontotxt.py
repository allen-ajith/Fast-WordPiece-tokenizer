import json

def json_to_text(json_file, text_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(text_file, 'w', encoding='utf-8') as f:
        for line in lines:
            try:
                data = json.loads(line)
                f.write(json.dumps(data, indent=4) + "\n")
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

if __name__ == "__main__":
    json_file = input("Enter the path to the JSON file: ")
    text_file = input("Enter the path to the output text file: ")
    json_to_text(json_file, text_file)
