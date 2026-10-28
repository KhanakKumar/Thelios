from pathlib import Path

label_dir = Path('extra\EDA\dataset\dataset\labels')
label_files = list(label_dir.glob('*.txt'))
    
for file in label_files:
    file = Path(file)
    if not file.exists():
        print(f"File not found: {file}")
        continue

    with open(file, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"[{file}] Line {idx}: Invalid format → {line.strip()}")
            continue

        try:
            _, x, y, w, h = map(float, parts)
        except ValueError:
            print(f"[{file}] Line {idx}: Non-numeric values → {line.strip()}")
            continue

        out_of_bounds = []
        for val_name, val in zip(["x", "y", "w", "h"], [x, y, w, h]):
            if not (0.0 <= val <= 1.0):
                out_of_bounds.append(f"{val_name}={val}")

        if out_of_bounds:
            print(f"[{file}] Line {idx}: Out of bounds → {', '.join(out_of_bounds)}")
