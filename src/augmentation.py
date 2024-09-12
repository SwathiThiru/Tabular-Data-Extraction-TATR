import os
import numpy as np
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

def augmentImages(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    '''# Define the ImageDataGenerator with relevant augmentation techniques
    datagen = ImageDataGenerator(
        #rotation_range=1,   # Random rotation
        brightness_range=[0.8, 1.5],  # Random brightness adjustment
        channel_shift_range=30,        # Random channel shift
        fill_mode='nearest'            # Fill mode for pixels outside the boundaries
    )
'''
    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = load_img(img_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            '''# Grayscale Image
            grayscale_img = img.convert('L')
            output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_grey.jpg')
            grayscale_img.save(output_path)'''

            # Image with Random Partial Mask
            min_mask_size = (int(x.shape[1] * 0.4), int(x.shape[2] * 0.4))

            # Randomly determine the mask size within the calculated minimum mask size range
            mask_size = (
                np.random.randint(min_mask_size[0], int(x.shape[1] * 0.8)),
                np.random.randint(min_mask_size[1], int(x.shape[2] * 0.8))
            )

            # Randomly determine the mask position
            mask_position = (
                np.random.randint(0, x.shape[1] - mask_size[0]),
                np.random.randint(0, x.shape[2] - mask_size[1])
            )

            mask_color = np.random.randint(0, 256, size=(3,))  # Random color
            mask_opacity = 0.2  # Adjust the opacity as needed

            # Create the masked image
            masked_image = x.copy()
            masked_image[:, mask_position[0]:mask_position[0] + mask_size[0],
            mask_position[1]:mask_position[1] + mask_size[1], :] = (
                                                                           1 - mask_opacity
                                                                   ) * masked_image[:,
                                                                       mask_position[0]:mask_position[0] + mask_size[0],
                                                                       mask_position[1]:mask_position[1] + mask_size[1],
                                                                       :] + (
                                                                       mask_opacity
                                                                   ) * mask_color.reshape((1, 1, 3))

            masked_image = array_to_img(masked_image[0])  # Convert back to PIL image
            output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_mask.jpg')
            masked_image.save(output_path)

            '''# Generate augmented images
            generated_images = []
            for batch in datagen.flow(x, batch_size=1):
                generated_images.append(array_to_img(batch[0]))
                if len(generated_images) == 1:
                    break

            # Save augmented images
            for i, generated_image in enumerate(generated_images):
                output_filename = os.path.splitext(filename)[0] + f'_gen.jpg'
                output_path = os.path.join(output_folder, output_filename)
                generated_image.save(output_path)'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', required=True, help="Directory for input images")
    parser.add_argument('--out_dir', required=True, help="Directory for output images")
    return parser.parse_args()

def main():
    args = get_args()
    print(args.__dict__)

    augmentImages(args.in_dir, args.out_dir)

if __name__ == "__main__":
    main()




"""import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

# Load an example image
img_path = 'Astronergy_2_0.jpg'
img = load_img(img_path)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

# Define the ImageDataGenerator with relevant augmentation techniques
datagen = ImageDataGenerator(
    rotation_range=1,   # Random rotation
    brightness_range=[0.8, 1.5],  # Random brightness adjustment
    channel_shift_range=100,        # Random channel shift
    fill_mode='nearest'            # Fill mode for pixels outside the boundaries
)

# Generate augmented images
generated_images = []
for batch in datagen.flow(x, batch_size=1):
    generated_images.append(array_to_img(batch[0]))
    if len(generated_images) == 1:
        break

# Plotting
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Original Image
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Generated Image
axes[1].imshow(generated_images[0])
axes[1].set_title('Generated Image')
axes[1].axis('off')

# Inverted Image
inverted_img = Image.fromarray((255 - x[0]).astype('uint8'))
axes[2].imshow(inverted_img)
axes[2].set_title('Inverted Image')
axes[2].axis('off')

# Image with Random Partial Mask
mask_area_threshold = 0.35  # At least 35% of the image area
img_height, img_width, _ = x.shape[1:]
min_mask_size = (int(np.sqrt(mask_area_threshold) * img_height), int(np.sqrt(mask_area_threshold) * img_width))

mask_size = (
    np.random.randint(min_mask_size[0], img_height),
    np.random.randint(min_mask_size[1], img_width)
)
mask_position = (
    np.random.randint(0, img_height - mask_size[0]),
    np.random.randint(0, img_width - mask_size[1])
)
mask_color = np.random.randint(0, 256, size=(3,))  # Random color
mask_opacity = 0.2  # Adjust the opacity as needed
masked_image = x.copy()
masked_image[:, mask_position[0]:mask_position[0] + mask_size[0], mask_position[1]:mask_position[1] + mask_size[1], :] = (
    1 - mask_opacity
) * masked_image[:, mask_position[0]:mask_position[0] + mask_size[0], mask_position[1]:mask_position[1] + mask_size[1], :] + (
    mask_opacity
) * mask_color.reshape((1, 1, 3))

axes[3].imshow(array_to_img(masked_image[0]))
axes[3].set_title('Image with Random Mask')
axes[3].axis('off')

plt.show()"""



'''import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

# Load an example image
img_path = 'DAS_DAS-NMAD9B_2_2.jpg'
img = load_img(img_path)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

# Define the ImageDataGenerator with creative augmentation techniques
datagen = ImageDataGenerator(
    rotation_range=5,            # Random rotation between -5 and 5 degrees
    width_shift_range=0.05,      # Random horizontal shift
    height_shift_range=0.05,     # Random vertical shift
    shear_range=0.1,             # Shear intensity
    zoom_range=0.05,             # Random zoom
    brightness_range=[0.9, 1.1], # Random brightness adjustment
    channel_shift_range=10,      # Random channel shift
    fill_mode='nearest',         # Fill mode for pixels outside the boundaries
)

# Generate augmented images and plot them
fig, ax = plt.subplots(3, 3, figsize=(20, 20))

# Original Image
ax[0, 0].imshow(array_to_img(x[0]))
ax[0, 0].set_title('Original')
ax[0, 0].axis('off')

for i, batch in enumerate(datagen.flow(x, batch_size=1)):
    augmented_image = array_to_img(batch[0])

    # Apply color changes manually
    hue_factor = np.random.uniform(0.9, 1.1)
    saturation_factor = np.random.uniform(0.8, 1.2)
    augmented_image = augmented_image.convert('HSV')
    augmented_image = augmented_image.point(lambda p: p * hue_factor)
    augmented_image = augmented_image.convert('RGB')

    # Plot augmented images
    ax[(i + 1) // 3, (i + 1) % 3].imshow(augmented_image)
    ax[(i + 1) // 3, (i + 1) % 3].set_title(f'Augmented {i + 1}')
    ax[(i + 1) // 3, (i + 1) % 3].axis('off')

    if i == 7:
        break

# Inverted Image
inverted_img = Image.fromarray((255 - x[0]).astype('uint8'))
ax[2, 0].imshow(inverted_img)
ax[2, 0].set_title('Inverted Image')
ax[2, 0].axis('off')

# Grayscale Image
grayscale_img = img.convert('L')
ax[2, 1].imshow(grayscale_img, cmap='gray')
ax[2, 1].set_title('Grayscale Image')
ax[2, 1].axis('off')

# Image with Random Partial Mask
mask_size = (int(x.shape[1] * 0.2), int(x.shape[2] * 0.2))
mask_position = (np.random.randint(0, x.shape[1] - mask_size[0]), np.random.randint(0, x.shape[2] - mask_size[1]))
masked_image = x.copy()
masked_image[:, mask_position[0]:mask_position[0] + mask_size[0], mask_position[1]:mask_position[1] + mask_size[1], :] = 0
ax[2, 2].imshow(array_to_img(masked_image[0]))
ax[2, 2].set_title('Image with Random Mask')
ax[2, 2].axis('off')

plt.show()

'''


'''import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Load an example image
img_path = 'Aoli_AL-G1M158_1_0.jpg'
img = load_img(img_path)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

# Define the ImageDataGenerator with creative augmentation techniques
datagen = ImageDataGenerator(
    rotation_range=4,  # Random rotation between -5 and 5 degrees
    width_shift_range=0.05,  # Random horizontal shift
    height_shift_range=0.05,  # Random vertical shift
    shear_range=0.1,  # Shear intensity
    zoom_range=0.05,  # Random zoom
    brightness_range=[0.9, 1.1],  # Random brightness adjustment
    channel_shift_range=50,  # Random channel shift
    fill_mode='nearest',  # Fill mode for pixels outside the boundaries
)

# Generate augmented images and plot them
fig, ax = plt.subplots(2, 2, figsize=(18, 18))
for i, batch in enumerate(datagen.flow(x, batch_size=1)):
    augmented_image = array_to_img(batch[0])

    # Apply color changes manually
    hue_factor = np.random.uniform(0.9, 1.1)
    saturation_factor = np.random.uniform(0.8, 1.2)
    augmented_image = augmented_image.convert('HSV')
    augmented_image = augmented_image.point(lambda p: p * hue_factor)
    augmented_image = augmented_image.convert('RGB')

    ax[i // 2, i % 2].imshow(augmented_image)
    ax[i // 2, i % 2].axis('off')

    if i == 3:
        break

plt.show()'''

