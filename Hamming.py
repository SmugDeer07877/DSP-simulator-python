#Requires going over to remove print() and out of function operations
#Add check for 2 bit error

def hamming (data):
    #preferably improve code to a similar syntax as dehamming function
    j = 0
    k = 0
    new_data = []
    new_data.append(0)
    for i in range(1, 16):
        if i == 2 ** j:
            new_data.append(0)
            j += 1
        else:
            new_data.append(data[k])
            k += 1

    #print("data with added parity:", new_data)
    # rows:
    q1 = 0
    for i in range(9, 16):
        q1 += new_data[i]
    if q1 % 2 != 0:
        new_data[8] = 1
    q2 = 0
    for i in range(5, 8):
        q2 += new_data[i]
    for i in range(12,16):
        q2 += new_data[i]
    if q2 % 2 != 0:
        new_data[4] = 1

    # columns:
    q3 = 0
    for i in range(3, 16, 4):
        q3 += new_data[i]
    for i in range(6,15,4):
        q3 += new_data[i]
    if q3 % 2 != 0:
        new_data[2] = 1
    q4 = 0
    for i in range(5, 14, 4):
        q4 += new_data[i]
    for i in range(3, 16, 4):
        q4 += new_data[i]
    if q4 % 2 != 0:
        new_data[1] = 1

    #parrity:
    total = 0
    for i in range(1,16):
        total += new_data[i]
    if total % 2 != 0:
        new_data[0] = 1
    return new_data


def dehamming(code):
    #add a check for 2 bits error using the parity bit
    #add striping of parity bits
    #First index is for YES MISTAKE, Second index is for NO MISTAKE
    quadrants = {1:[3,5,7,9,11,13,15,5,10],
                 2:[3,6,7,10,11,14,15,3,12],
                 3:[5,6,7,12,13,14,15,5,10],
                 4:[9,10,11,12,13,14,15,3,12]}
    cases = {}
    error = 0
    sum = 0
    for s in range(1, len(code)):
        sum += code[s]
    for i in range(1,5):
        total = 0
        for n in range(7):
            total += code[quadrants[i][n]]
        if total % 2 != code[2**(i-1)]:
            cases[f"q{i}"] = quadrants[i][7]
            error +=1
        else:
            cases[f"q{i}"] = quadrants[i][8]
        #print(f"q{i}:",total)
    #print(cases)
    if error > 0 and (sum % 2 == code[0]):
        print("Two Bit Error Detected")
    elif error > 0:
        error_index = 4*(cases["q1"] & cases["q2"]) | (cases["q3"] & cases["q4"])
        if error_index > len(code):
            print("Unfixable Error")
        #print("error index:", error_index)
        else:
            if code[error_index] == 0:
                code[error_index] = 1
            else:
                code[error_index] = 0
    else:
        print("No error detected")
    #print("corrected code:",code)

    #Remove parity bits
    for i in range (3,-1,-1):
        code.pop(2**i)
    code.pop(0)
    return code

if __name__ == '__main__':
    data = [1,0,1,0,1,0,1,0,1,0,1]
    data1 = [1,1,0,0,1,0,1,1,0,1,1]
    data2 = [0,1,0,1,1,1,0,1,0,0,0]
    corrupted = [1,1,1,0,1,1,0,0,1,1,1,1,1,0,0,1]
    new_data = hamming(data1)
    print("hammed_data:", new_data)
    corrected_code = dehamming(corrupted)
    print("striped code:", corrected_code)