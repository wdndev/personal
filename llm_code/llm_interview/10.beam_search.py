
class BeamSearchNode:
    def __init__(self, sequence, score):
        self.sequence = sequence    # 生成的序列
        self.score = score          # 分数（概率）

def simple_next_word_probs(sequence):
    """ 示例：下一个token的概率函数，简单使用固定概率
    """
    if sequence[-1] == "<end>":
        return {}
    return {"apple": 0.3, "like": 0.35, "peach": 0.2, "banana": 0.15}

def beam_search(initial_sequence, next_word_probs_func, num_beams, max_sequence_length):
    # 初始化初始节点，且分数为1
    initial_node = BeamSearchNode(sequence=initial_sequence, score=1.0)
    candidates = [initial_node]

    final_candidates = []  # 最终的候选序列
    # 只要候选节点列表不为空，且 final_candidates 中的候选节点数量还没有达到指定的束宽度，就继续进行搜索
    while candidates and len(final_candidates) < num_beams:
        # 候选节点排序
        candidates.sort(key=lambda x: -x.score)
        current_node = candidates.pop(0)
        # 当节点序列末尾生成结束符号（如"<end>"），或者当生成的序列长度达到最大限制时终止节点的扩展
        if current_node.sequence[-1] == "<end>" or len(current_node.sequence) >= max_sequence_length:
            final_candidates.append(current_node)
        else:
            # 获取下一个token的概率，我们的例子返回的是固定的概率
            next_words_probs = next_word_probs_func(current_node.sequence) 
            # 生成新的候选序列，并计算分数           
            for next_word, next_word_prob in next_words_probs.items():
                new_sequence = current_node.sequence + [next_word]
                new_score = current_node.score * next_word_prob
                new_node = BeamSearchNode(sequence=new_sequence, score=new_score)
                candidates.append(new_node)

    return [candidate.sequence for candidate in final_candidates]

initial_sequence = ["<start>", "I"]
num_beams = 3
max_sequence_length = 3
result = beam_search(initial_sequence, simple_next_word_probs, num_beams, max_sequence_length)

for idx, sequence in enumerate(result):
    print(f"Sentence {idx + 1}: {' '.join(sequence)}")