# LaTeX 常用笔记
## 宏包
- 开头缩进  
  \usepackage{indentfirst}   
  \setlength{\parindent}{2em}
- 插入图片  
  \usepackage{graphicx}
- 页边距  
  \usepackage[margin=1.5cm]{geometry}
## 列表
- 无序列表  
  ```tex
  \begin{itemize}
    \item A
        \begin{itemize}
            \item a_1
            \item a_2
        \end{itemize}
    \item B
        \begin{itemize}
            \item b_1
            \item b_2
        \end{itemize}
  \end{itemize}
  ```
- 表格
  ```tex
  \textbf{标题1}~~\\{\bfseries 标题2}
			\begin{table}[h] \footnotesize %--设置字体大小
				\begin{tabular}{|p{3.5cm}|p{5cm}|p{4cm}|p{4cm}|}
					\hline
					a & b & c & d \\
					\hline
					a & b & c & d \\
					\hline
					a & b & c & d \\
					\hline
					a & b & c & d\\
					\hline
					a & b & c & d \\
					\hline
				\end{tabular}	
			\end{table}
  ```
