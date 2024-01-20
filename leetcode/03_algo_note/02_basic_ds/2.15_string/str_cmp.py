""" 字符串比较
"""

def strcmp(str1:str, str2:str):
    """ 字符串比较
    """
    idx1 = 0
    idx2 = 0
    # ord 返回字符的 ASCII 数值
    while idx1 < len(str1) and idx2 < len(str2):
        if ord(str1[idx1]) == ord(str2[idx2]):
            idx1 += 1
            idx2 += 1
        elif ord(str1[idx1]) < ord(str2[idx2]):
            return -1
        else:
            return 1
        
    if len(str1) < len(str2):
        return -1
    elif len(str1) > len(str2):
        return 1
    else:
        return 0


