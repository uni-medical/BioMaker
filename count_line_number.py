# -*- coding: utf-8 -*-
"""
该脚本用于分析 tree_plot.html 文件，提取其中的树状结构数据，
并统计代码行数的变化。

功能:
1. 使用 BeautifulSoup 解析 HTML 文件。
2. 从 <script> 标签中稳健地提取 'treeStructData' JavaScript 对象。
3. 将提取的字符串解析为 Python 字典。
4. 获取所有代码节点的行数。
5. 将行数升序排列，并绘制图表展示代码行数随步骤增加的增长情况。
6. 显示图表并保存为 'line_number_analysis.png'。
"""

import argparse
import json
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

def parse_html_and_extract_data(html_file_path):
    """
    解析HTML文件并提取treeStructData
    
    :param html_file_path: HTML文件的路径
    :return: 包含树结构数据的字典，如果找不到则返回None
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError:
        print(f"错误：文件 '{html_file_path}' 未找到。")
        return None

    soup = BeautifulSoup(html_content, 'html.parser')
    scripts = soup.find_all('script')

    data_str = None
    for script in scripts:
        if not script.string:
            continue
        
        script_content = script.string
        start_marker = 'let treeStructData = '
        start_index = script_content.find(start_marker)
        
        if start_index != -1:
            # 找到数据定义的起始位置
            obj_start_index = script_content.find('{', start_index)
            if obj_start_index == -1:
                continue

            # 使用花括号平衡来找到对象的结束位置，这比正则表达式更健壮
            brace_level = 1
            search_index = obj_start_index + 1
            
            # 遍历字符串以找到匹配的闭合花括号
            while search_index < len(script_content) and brace_level > 0:
                char = script_content[search_index]
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                search_index += 1

            if brace_level == 0:
                # 成功找到匹配的括号
                obj_end_index = search_index
                data_str = script_content[obj_start_index:obj_end_index]
                break  # 找到了数据，跳出循环
            else:
                # 未找到匹配的括号，可能数据格式有问题
                continue
    
    if not data_str:
        print("错误：在HTML文件中找不到 'treeStructData' 或其格式不完整。")
        return None

    try:
        # 将提取到的JS对象字符串解析为Python字典
        data = json.loads(data_str)
        return data
    except json.JSONDecodeError as e:
        print(f"错误：解析JSON数据失败 - {e}")
        print("--- 提取到的数据字符串 (前500字符) ---")
        print(data_str[:500])
        print("------------------------------------")
        return None

def analyze_and_visualize_data(data):
    """
    分析数据并生成可视化图表
    
    :param data: 包含树结构数据的字典
    """
    if 'code' not in data:
        print("Error: Data format is incorrect, missing 'code' key.")
        return

    codes = data['code']

    # 1. 获取所有node的code line number
    line_counts = [len(c.splitlines()) for c in codes]
    print(f"Successfully analyzed {len(line_counts)} code nodes.")

    # 2. 升序排列
    sorted_line_counts = sorted(line_counts)
    print('\n'.join([f"{count}" for i, count in enumerate(sorted_line_counts)]))
    
    # 步骤即为被分析的节点总数，从 1 到 N
    steps = range(1, len(sorted_line_counts) + 1)
    
    # 3. 给出英文title的图
    plt.style.use('seaborn-v0_8-whitegrid')
    # 创建单个图表
    plt.figure(figsize=(10, 6))

    plt.plot(steps, sorted_line_counts, marker='o', linestyle='-', color='c', label='Line Count per Node (Sorted)')
    
    # 使用英文标题
    plt.title('Growth of Code Line Numbers Across All Steps', fontsize=16)
    plt.xlabel('Step (Nodes Sorted by Line Count)', fontsize=12)
    plt.ylabel('Number of Code Lines', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    output_filename = 'line_number_analysis.png'
    plt.savefig(output_filename)
    print(f"Analysis plot saved as '{output_filename}'")
    
    # 显示图表
    plt.show()


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="分析 tree_plot.html 文件，追踪代码行数的变化。"
    )
    parser.add_argument(
        "html_file", 
        help="需要分析的 tree_plot.html 文件的路径。"
    )
    args = parser.parse_args()

    # 执行主流程
    tree_data = parse_html_and_extract_data(args.html_file)
    if tree_data:
        analyze_and_visualize_data(tree_data)

