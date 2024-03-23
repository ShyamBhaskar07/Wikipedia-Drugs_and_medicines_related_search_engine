import os
import json

MAX_TERM_LENGTH = 15  # Set your desired maximum term length

def create_structure(node, term, posting_list):
    if not term:
        node["posting_list"] = posting_list
        return

    first_char, rest_of_term = term[0], term[1:]
    if first_char not in node:
        node[first_char] = {}

    create_structure(node[first_char], rest_of_term, posting_list)

def create_directory_structure_from_text(text_file, root):
    with open(text_file, "r") as f:
        for line in f:
            term, posting_list_str = line.strip().split(maxsplit=1)
            posting_list = eval(posting_list_str)
            
            if len(term) <= MAX_TERM_LENGTH:
                create_structure(root, term, posting_list)

def save_structure_to_disk(node, current_path=""):
    for char, child_node in node.items():
        # Sanitize the character for the directory name
        sanitized_char = "".join(c for c in char if c.isalnum() or c in ('_', '-'))
        new_path = os.path.join(current_path, sanitized_char)
        
        if char == "posting_list":
            with open(new_path + ".json", "w") as f:
                f.write(json.dumps(child_node))
        else:
            os.makedirs(new_path, exist_ok=True)
            save_structure_to_disk(child_node, new_path)

# Example usage
root = {}
text_file_path = "sorted_clean.txt"

create_directory_structure_from_text(text_file_path, root)
save_structure_to_disk(root)
