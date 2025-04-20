import cv2 as cv
import numpy as np

#image dimensions:
image_height = 100
image_width = 100
num_channels = 3

def bits_to_image(bit_list):
    print("Started picture reconstruction")
    # Step 0: Check for access data:
    bit_num = image_height*image_width*num_channels*8
    print("bit_num:",bit_num)
    print("bit_list:", len(bit_list))
    if len(bit_list) != bit_num:
        if len(bit_list) > bit_num:
            for i in range(len(bit_list) - bit_num):
                bit_list.pop()
        else:
            print("Missing Data!")
            return
    print(len(bit_list))

    # Step 1: Group bits into bytes (8 bits per byte)
    byte_list = []
    for i in range(0, len(bit_list), 8):
        byte = bit_list[i:i + 8]
        byte_value = int(''.join(map(str, byte)), 2)
        byte_list.append(byte_value)

    # Step 2: Convert byte list to numpy array
    img_array = np.array(byte_list, dtype=np.uint8)

    # Step 3: Reshape into image shape
    try:
        img = img_array.reshape((image_height, image_width, num_channels))
    except ValueError:
        print("Error: Bit list size doesn't match expected image dimensions.")
        return

    # Step 4: Convert BGR to RGB (if needed)
    #img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Step 4.5: Resize Image:
    max_width = 800
    max_height = 800
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_size = (int(w * scale), int(h * scale))
    resized_img = cv.resize(img, new_size, interpolation=cv.INTER_AREA)

    # Step 5: Display the image
    cv.imshow('Reconstructed Image', resized_img)
    cv.waitKey(0)
    cv.destroyAllWindows()