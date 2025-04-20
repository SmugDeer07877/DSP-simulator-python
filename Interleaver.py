import numpy as np
#Assumes that data is divisible by 3.

def interleaver (code):
    data, leftover = devis_by_3(code)
    arr1 = data[0:int(len(data)/3)]
    arr2 = data[int(len(data)/3):int(len(data)*2/3)]
    arr3 = data[int(len(data)*2/3):]
    arr_tuple = (arr1, arr2, arr3)
    interleaved = np.vstack(arr_tuple).reshape((-1), order = 'F').tolist()
    interleaved.extend(leftover)
    return interleaved

def deinterleaver (code):
    data, leftover = devis_by_3(code)
    channels = 3
    frames = np.array(data)
    deinterleaved = [frames[idx::channels] for idx in range(channels)]
    deinterleaved = np.concatenate(deinterleaved).tolist()
    deinterleaved.extend(leftover)
    return deinterleaved

def devis_by_3 (data):
    leftover = []
    if len(data) % 3 != 0:
        for i in range(len(data) % 3):
            leftover.append(data.pop())
    return data, leftover


if __name__ == '__main__':
    data = [1,1,1,0,0,0,1,1,1]
    interleaved_data = interleaver(data)
    print(interleaved_data)
    new_data = deinterleaver(interleaved_data)
    print(new_data)
