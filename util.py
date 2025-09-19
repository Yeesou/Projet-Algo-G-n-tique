def gen_names(n):
        noms = []
        i = 0
        while len(noms) < n:
            noms.append(num_to_name(i))
            i += 1
        return noms

def num_to_name(num):
    name = ""
    while True:
        num, r = divmod(num, 26)
        name = chr(65 + r) + name
        if num == 0:
            break
        num -= 1
    return name