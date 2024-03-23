# Read the inverted index from a file
with open('index.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Sort the lines alphabetically
sorted_lines = sorted(lines)

# Write the sorted lines back to a new file
with open('sorted_index.txt', 'w', encoding='utf-8') as file:
    file.writelines(sorted_lines)
