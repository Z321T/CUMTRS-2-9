def convert_format(input_file, output_file):
    """
    将逗号分隔格式转换为制表符分隔格式
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 去除行末换行符并按逗号分割
                line = line.strip()
                if line:  # 确保不是空行
                    # 将逗号替换为制表符
                    converted_line = line.replace(',', '\t')
                    outfile.write(converted_line + '\n')

# 使用示例
if __name__ == "__main__":
    input_filename = r"result\recommendation_cf_20251017_210334.txt"  # 使用原始字符串
    output_filename = "result/output.txt"  # 输出文件名

    try:
        convert_format(input_filename, output_filename)
        print(f"格式转换完成！结果已保存到 {output_filename}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_filename}")
    except Exception as e:
        print(f"转换过程中发生错误：{e}")
