import cv2
import numpy as np

# 002 and 005 abit different

# user choose 001 to 008
print("You can extract paragraphs of images from 001.png to 008.png.\n")
while (True):
    user_n = input("Please pick a number from 1 to 8: ")
    print("\n")
    try:
        if 1 <= int(user_n) <= 8:
            input_image = cv2.imread(f'00{user_n}.png')
            break
    except ValueError:
            print("Please enter an integer within 1 to 8.\n")
# Convert the image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to the grayscale image
blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

# Apply Otsu's thresholding to create a binary image
thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Create a rectangular kernel for dilation
rectangle_kernel = np.ones((3, 4), np.uint8)

# Perform dilation on the binary image
dilated_image = cv2.dilate(thresholded_image, rectangle_kernel, iterations=15)

# Find contours of connected components (paragraphs) in the dilated image
contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours based on the sum of the rounded y-coordinate and the product of the rounded x-coordinate and image height
contours = sorted(contours, key=lambda contour: round(cv2.boundingRect(contour)[1], -1) + round(cv2.boundingRect(contour)[0], -1) * input_image.shape[0])

# Iterate through the sorted contours
n = 1
for contour in contours:
    # Get bounding box coordinates and dimensions
    bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height = cv2.boundingRect(contour)

    # Extract and save each paragraph image
    paragraph_image = input_image[bounding_box_y:bounding_box_y + bounding_box_height, bounding_box_x:bounding_box_x + bounding_box_width]
    cv2.imwrite(f"paragraph_{n}.png", paragraph_image)
    n += 1

# Display useful information about the detected paragraphs
print(f"Number of paragraphs detected: {len(contours)}")
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    print(f"Paragraph {i + 1}: X={x}, Y={y}, Width={w}, Height={h}")
    
    # Display rectangle around the paragraph
    cv2.rectangle(dilated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display paragraph index
    cv2.putText(dilated_image, f"Paragraph {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# Display the dilated image with detected paragraphs
cv2.imshow('Image after Extraction', dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
