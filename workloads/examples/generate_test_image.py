# generate test images for kernel testing
# run: python3 generate_test_image.py
# spits it out in ../test_images/
import numpy as np
import cv2
import os

output_dir = "../test_images"
os.makedirs(output_dir, exist_ok=True)

# checkerboard pattern (ideal for edge detection)
def create_checkerboard(width, height, square_size):
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                img[y, x] = 255
    return img

# gradient (ideal fo testing blurs)
def create_gradient(width, height):
    img = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        img[:, x] = int((x / width) * 255)
    return img

# random noise
def create_noise(width, height):
    return np.random.randint(0, 256, (height, width), dtype=np.uint8)

# circles ('nother corner detection)
def create_circles(width, height):
    img = np.zeros((height, width), dtype=np.uint8)
    center1 = (width // 3, height // 2)
    center2 = (2 * width // 3, height // 2)
    cv2.circle(img, center1, 200, 255, 5)
    cv2.circle(img, center2, 200, 255, 5)
    return img

# generate standard test images
print("generating test images...")

# 720p images
cv2.imwrite(f"{output_dir}/checkerboard_720p.png", create_checkerboard(1280, 720, 32))
cv2.imwrite(f"{output_dir}/gradient_720p.png", create_gradient(1280, 720))
cv2.imwrite(f"{output_dir}/noise_720p.png", create_noise(1280, 720))
cv2.imwrite(f"{output_dir}/circles_720p.png", create_circles(1280, 720))

# 1080p images
cv2.imwrite(f"{output_dir}/checkerboard_1080p.png", create_checkerboard(1920, 1080, 64))
cv2.imwrite(f"{output_dir}/gradient_1080p.png", create_gradient(1920, 1080))
cv2.imwrite(f"{output_dir}/noise_1080p.png", create_noise(1920, 1080))
cv2.imwrite(f"{output_dir}/circles_1080p.png", create_circles(1920, 1080))

# lena (the png equivalent of big buck bunny)
lena_path = f"{output_dir}/lena_512.png"
if not os.path.exists(lena_path):
    print("downloading lena test image...")
    try:
        import urllib.request
        url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        urllib.request.urlretrieve(url, lena_path)
    except:
        print("  (failed to download, skipping)")

print(f"\ngenerated images in {output_dir}/:")
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.png'):
        img = cv2.imread(f"{output_dir}/{f}", cv2.IMREAD_GRAYSCALE)
        print(f"  {f}: {img.shape[1]}x{img.shape[0]}")
