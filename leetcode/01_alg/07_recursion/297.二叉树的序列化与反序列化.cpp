// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem297.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=297 lang=cpp
 *
 * [297] 二叉树的序列化与反序列化
 */

// @lc code=start
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

// 序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，
// 同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

// 请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，
// 你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

// 提示: 输入输出格式与 LeetCode 目前使用的方式一致，详情请参阅 LeetCode 序列化二叉树的格式。
// 你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。

// 示例 1：
// 输入：root = [1,2,3,null,null,4,5]
// 输出：[1,2,3,null,null,4,5]

// 示例 2：
// 输入：root = []
// 输出：[]

// 示例 3：
// 输入：root = [1]
// 输出：[1]

// 示例 4：
// 输入：root = [1,2]
// 输出：[1,2]


class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        std::string ans;
        this->pre_order(root, ans);
        return ans;
    }

    // 先序遍历输出
    void pre_order(TreeNode* root, std::string& str) {
        if (root == nullptr) {
            str += "None,";
            return;
        }
        str += std::to_string(root->val) + ",";
        this->pre_order(root->left, str);
        this->pre_order(root->right, str);

    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        std::list<std::string> data_array;
        std::string str;
        // 将字符串分割，存在list中
        for (auto& ch : data) {
            if (ch == ',') {
                data_array.push_back(str);
                str.clear();
            } else {
                str.push_back(ch);
            }
        }

        // 处理最后一个
        if (!str.empty()) {
            data_array.push_back(str);
            str.clear();
        }

        // 先序遍历创建树
        return this->pre_create_tree(data_array);
    }

    // 先序遍历创建树，创建一棵树
    TreeNode* pre_create_tree(std::list<std::string>& data_array) {
        // 如果碰到 None 返回
        if (data_array.front() == "None") {
            data_array.erase(data_array.begin());
            return nullptr;
        }

        // 新建结点
        TreeNode* root = new TreeNode(std::stoi(data_array.front().c_str()));
        data_array.erase(data_array.begin());

        // 创建左子树
        root->left = this->pre_create_tree(data_array);
        // 创建右子树
        root->right = this->pre_create_tree(data_array);

        return root;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec ser, deser;
// TreeNode* ans = deser.deserialize(ser.serialize(root));
// @lc code=end

