/*
 * @lc app=leetcode.cn id=590 lang=cpp
 *
 * [590] N 叉树的后序遍历
 *
 * https://leetcode.cn/problems/n-ary-tree-postorder-traversal/description/
 *
 * algorithms
 * Easy (78.75%)
 * Likes:    298
 * Dislikes: 0
 * Total Accepted:    134.6K
 * Total Submissions: 170.9K
 * Testcase Example:  '[1,null,3,2,4,null,5,6]'
 *
 * 给定一个 n 叉树的根节点 root ，返回 其节点值的 后序遍历 。
 * 
 * n 叉树 在输入中按层序遍历进行序列化表示，每组子节点由空值 null 分隔（请参见示例）。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：root = [1,null,3,2,4,null,5,6]
 * 输出：[5,6,3,2,4,1]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 
 * 
 * 输入：root =
 * [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
 * 输出：[2,6,14,11,7,3,12,8,4,13,9,10,5,1]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 节点总数在范围 [0, 10^4] 内
 * 0 <= Node.val <= 10^4
 * n 叉树的高度小于或等于 1000
 * 
 * 
 * 
 * 
 * 进阶：递归法很简单，你可以使用迭代法完成此题吗?
 * 
 */

// @lc code=start
/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};
*/

class Solution {
public:
    vector<int> postorder(Node* root) {
        std::vector<int> ans;
        this->postorder_recu(root, ans);

        return ans;
    }

private:
    void postorder_recu(Node* root, std::vector<int>& ans) {
        if (root == nullptr) {
            return;
        }

        for (auto& child : root->children) {
            this->postorder_recu(child, ans);
        }

        ans.push_back(root->val);
    }
};
// @lc code=end

