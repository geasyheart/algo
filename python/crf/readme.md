## 
bilstm_crf.py讲解了neg_loglikehood的计算和viterbi算法的前向运算

其中 self.transitions 一般行表示前一状态，列表示后一状态，而这里相反，看得时候需要注意

另外发现另一个crf的实现repo，也可以一起混合着看，交叉理解。

> https://github.com/geasyheart/linear-chiain-crf