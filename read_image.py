#My file path: r"C:\Users\adibl\Pictures\IronMan.jpg"
import cv2 as cv

#image dimantions:
image_height = 100
image_width = 100
num_channels = 3

image_path = "Emogi.png"
def image_to_bits(image_path):
    # Step 1: Read the image (BGR format by default)
    img = cv.imread(image_path)

    # Step 2: Flatten the image to 1D array of bytes
    flat_bytes = img.flatten()

    # Step 3: Convert each byte (0-255) to 8-bit binary and split into bits
    bit_list = []
    for byte in flat_bytes:
        bits = [int(bit) for bit in format(byte, '08b')]  # convert to 8-bit string
        bit_list.extend(bits)
    print("Finished image deconstruction")
    return bit_list

if __name__ == '__main__':
    data = image_to_bits("IronMan.jpg")
    print(data[:100])
    if len(data) == (image_height*image_width*num_channels*8):
        print("All Good!")