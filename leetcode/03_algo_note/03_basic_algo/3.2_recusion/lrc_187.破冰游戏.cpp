// 社团共有 num 为成员参与破冰游戏，编号为 0 ~ num-1。
// 成员们按照编号顺序围绕圆桌而坐。社长抽取一个数字 target，
// 从 0 号成员起开始计数，排在第 target 位的成员离开圆桌，
// 且成员离开后从下一个成员开始计数。请返回游戏结束时最后一位成员的编号。


// 示例 1：
// 输入：num = 7, target = 4
// 输出：1

// 示例 2：
// 输入：num = 12, target = 5
// 输出：0


// 长度为 num 的序列会先删除第 `target % num` 个元素，
// 然后剩下一个长度为 `num - 1` 的序列。那么，
// 可以递归地求解 `f(num - 1, target)`，就可以知道对于剩下的 `num - 1 `个元素，
// 最终会留下第几个元素，设答案为 `x = f(num - 1, target)`。

// 由于删除了第 `target % num` 个元素，将序列的长度变为 `num - 1`。
// 当知道了` f(num - 1, target)` 对应的答案 x 之后，也就可以知道，
// 长度为 num 的序列最后一个删除的元素，应当是从 `target % num` 开始数的第 x 个元素。
// 因此有 `f(num, target) = (target % num + x) % num = (target + x) % num`。

class Solution {
public:
    int iceBreakingGame(int num, int target) {
        return this->f(num, target);
    }

    int f(int num, int target) {
        if (num == 1) {
            return 0;
        }
        int x = this->f(num - 1, target);
        return (target + x) % num;
    }
};