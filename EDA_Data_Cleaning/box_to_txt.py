def convert_to_yolo_format(input_lines, image_width, image_height):
    yolo_labels = []

    for line in input_lines:
        parts = line.strip().split(',')

        if len(parts) < 4:
            continue  # Skip malformed lines

        try:
            x_min, y_min = int(parts[0]), int(parts[1])
            x_max, y_max = int(parts[2]), int(parts[3])

            # Clamp coordinates to valid image dimensions
            x_min = max(0, min(x_min, image_width - 1))
            x_max = max(0, min(x_max, image_width - 1))
            y_min = max(0, min(y_min, image_height - 1))
            y_max = max(0, min(y_max, image_height - 1))

            # Skip invalid boxes (width or height ≤ 0)
            if x_max <= x_min or y_max <= y_min:
                continue

            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            class_id = 0
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_labels.append(yolo_line)

        except ValueError:
            continue

    return yolo_labels


from pathlib import Path

box_label_dir = Path('dataset/sf_week2/labels')
yolo_label_dir = Path('dataset/sf_week2/labels_txt')
yolo_label_dir.mkdir(parents=True, exist_ok=True)

for box in box_label_dir.glob('*.box'):
    with open(box, 'r') as b:
        raw_labels = b.readlines()

    yolo_output = convert_to_yolo_format(raw_labels, 1280, 720)
    yolo_file = yolo_label_dir / f'{box.stem}.txt'

    with open(yolo_file, 'w') as f:
        for line in yolo_output:
            f.write(line + '\n')