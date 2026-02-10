import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime

# Base directory
base_path = r"C:\Users\JackHu\OneDrive - Oribiotech Ltd\Desktop\Particle Photos jan2126"
image_files = [f for f in os.listdir(base_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(image_files)} image files: {image_files}")

# %% Settings
threshold_value = 90
min_area = 1
box_size = 40  # size of each box

# %% Analyze all images
results = {}

for image_file in image_files:
    print(f"\n{'='*50}")
    print(f"Analyzing: {image_file}")
    print('='*50)
    
    image_path = os.path.join(base_path, image_file)
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error reading {image_file}")
        continue
    
    h, w = frame.shape[:2]
    print(f"Image size: {w} x {h}")
    
    # Create grid of boxes (no overlap)
    step = box_size
    
    valid_boxes = []
    for y in range(0, h - box_size, step):
        for x in range(0, w - box_size, step):
            valid_boxes.append((x, y, x + box_size, y + box_size))
    
    print(f"Box size: {box_size}px (no overlap)")
    print(f"Total boxes: {len(valid_boxes)}")
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Track which boxes have particles
    boxes_with_particles = set()
    all_centroids = []
    
    for c in contours:
        if cv2.contourArea(c) > min_area:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                all_centroids.append([cx, cy])
                
                # Find which box this particle is in
                for idx, (x1, y1, x2, y2) in enumerate(valid_boxes):
                    if x1 <= cx < x2 and y1 <= cy < y2:
                        boxes_with_particles.add(idx)
    
    # Calculate spread
    filled_boxes = len(boxes_with_particles)
    total_boxes = len(valid_boxes)
    spread_percent = (filled_boxes / total_boxes * 100) if total_boxes > 0 else 0
    
    # Determine status
    if spread_percent < 30:
        status = "CONCENTRATED"
    else:
        status = "DISTRIBUTED"
    
    results[image_file] = {
        'filled_boxes': filled_boxes,
        'total_boxes': total_boxes,
        'spread_percent': spread_percent,
        'num_particles': len(all_centroids),
        'status': status
    }
    
    print(f"\nParticles detected: {len(all_centroids)}")
    print(f"Boxes with particles: {filled_boxes}/{total_boxes} ({spread_percent:.1f}%)")
    print(f"Result: {status}")
    
    # Show visualization
    display = frame.copy()
    
    # Draw grid and color boxes
    for idx, (x1, y1, x2, y2) in enumerate(valid_boxes):
        if idx in boxes_with_particles:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # Draw particles
    for cx, cy in all_centroids:
        cv2.circle(display, (cx, cy), 2, (255, 255, 0), -1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"{image_file}\nOriginal")
    axes[0].axis('off')
    
    axes[1].imshow(thresh, cmap='gray')
    axes[1].set_title(f"Thresholded (value={threshold_value})")
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Green=particles, Red=empty\n{filled_boxes}/{total_boxes} boxes ({spread_percent:.1f}%) - {status}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# %% Summary
print("\n" + "="*60)
print("SUMMARY - ALL IMAGES")
print("="*60)

for image, data in results.items():
    print(f"\n{image}:")
    print(f"  Particles: {data['num_particles']}")
    print(f"  Boxes filled: {data['filled_boxes']}/{data['total_boxes']} ({data['spread_percent']:.1f}%)")
    print(f"  Status: {data['status']}")

print("\n" + "="*60)
print("LOGIC:")
print("  <30% boxes filled = CONCENTRATED")
print("  30%+ boxes filled = DISTRIBUTED")
print("="*60)

# %% Save results to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(base_path, f"particle_results_{timestamp}.csv")

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    writer.writerow(['Image File', 'Particles', 'Boxes Filled', 'Total Boxes', 'Spread %', 'Status'])
    
    for image, data in results.items():
        writer.writerow([
            image,
            data['num_particles'],
            data['filled_boxes'],
            data['total_boxes'],
            round(data['spread_percent'], 1),
            data['status']
        ])

print(f"\nResults saved to: {csv_filename}")
print("\nDone!")