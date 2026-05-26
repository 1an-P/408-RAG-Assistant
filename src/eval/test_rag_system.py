import json
import os
from tqdm import tqdm
import sys

# 添加项目根目录到Python路径
current_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)
great_grandparent_dir = os.path.dirname(grandparent_dir)
sys.path.append(great_grandparent_dir)

from src.rag.rag_main import RAGSystem


def test_rag_system_with_alpha(alpha=0.5):
    """使用RAG系统测试400题数据，计算正确率"""
    # 定义项目根目录
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 定义数据路径
    persist_directory = os.path.join(project_dir, "data_base/vector_db/408.db")
    test_data_path = os.path.join(project_dir, "data/test_data/questions_400.json")
    output_folder = os.path.join(project_dir, "output/400_question")
    os.makedirs(output_folder, exist_ok=True)

    # 初始化RAG系统
    rag_system = RAGSystem(persist_dir=persist_directory, strategy="chapter", alpha=alpha)

    # 加载测试数据
    with open(test_data_path, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    # 测试结果
    results = []
    correct_count = 0
    total_count = len(test_data)

    # 运行测试
    print(f"开始测试，alpha={alpha}，共{total_count}题")
    for i, item in tqdm(enumerate(test_data, start=1)):
        question = item["question"]
        correct_answer = item["answer"]
        options = item["options"]
        
        # 构建完整问题（包含选项）
        full_question = f"{question}\n选项："
        for key, value in options.items():
            full_question += f"{key}. {value}\n"
        full_question += "请选择正确答案，只需输出选项字母。"
        
        try:
            # 使用RAG系统回答（测试模式，只输出答案）
            answer = rag_system.query(full_question, alpha=alpha, test_mode=True)
            
            # 提取答案（尝试从回答中提取选项字母）
            extracted_answer = extract_answer(answer)
            
            # 检查答案是否正确
            is_correct = extracted_answer == correct_answer
            if is_correct:
                correct_count += 1
            
            # 保存结果
            results.append({
                "序号": i,
                "问题": question,
                "选项": options,
                "正确答案": correct_answer,
                "模型结果": extracted_answer,
                "模型回答": answer,
                "测试结果": is_correct
            })
            
            print(f"第{i}题：{'正确' if is_correct else '错误'}，模型答案：{extracted_answer}，正确答案：{correct_answer}")
            
        except Exception as e:
            print(f"第{i}题处理失败：{str(e)}")
            results.append({
                "序号": i,
                "问题": question,
                "选项": options,
                "正确答案": correct_answer,
                "模型结果": "错误",
                "模型回答": str(e),
                "测试结果": False
            })
    
    # 计算正确率
    accuracy = correct_count / total_count
    print(f"测试完成，alpha={alpha}，正确数：{correct_count}，总题数：{total_count}，正确率：{accuracy:.4f}")
    
    # 保存结果
    result_file = os.path.join(output_folder, f"rag_alpha_{alpha}_400_question_result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        result_dic = {
            "alpha": alpha,
            "正确个数": correct_count,
            "总题数": total_count,
            "正确率": accuracy,
            "详细结果": results
        }
        json.dump(result_dic, f, ensure_ascii=False, indent=4)
    
    return accuracy


def extract_answer(answer):
    """从模型回答中提取选项字母"""
    # 尝试从回答中提取选项字母（A、B、C、D）
    import re
    match = re.search(r'[ABCD]', answer)
    if match:
        return match.group(0)
    # 如果没有找到，返回空
    return ""


if __name__ == "__main__":
    # 测试不同的alpha值
    alpha_values = [0.3, 0.5, 0.7]
    # 先测试10题，验证脚本是否正常工作
    test_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/test_data/questions_400.json")
    with open(test_data_path, "r", encoding="utf-8") as file:
        test_data = json.load(file)
    print(f"总题数：{len(test_data)}")
    print("测试前5题，验证脚本是否正常工作...")
    
    # 临时修改test_rag_system_with_alpha函数，只测试前5题
    def test_rag_system_with_alpha_test(alpha=0.5):
        """使用RAG系统测试前5题数据，验证脚本是否正常工作"""
        # 定义项目根目录
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 定义数据路径
        persist_directory = os.path.join(project_dir, "data_base/vector_db/408.db")
        test_data_path = os.path.join(project_dir, "data/test_data/questions_400.json")
        output_folder = os.path.join(project_dir, "output/400_question")
        os.makedirs(output_folder, exist_ok=True)

        # 初始化RAG系统
        rag_system = RAGSystem(persist_dir=persist_directory, strategy="chapter", alpha=alpha)

        # 加载测试数据
        with open(test_data_path, "r", encoding="utf-8") as file:
            test_data = json.load(file)
        # 只测试前10题
        test_data = test_data[:10]

        # 测试结果
        results = []
        correct_count = 0
        total_count = len(test_data)

        # 运行测试
        print(f"开始测试，alpha={alpha}，共{total_count}题")
        for i, item in enumerate(test_data, start=1):
            question = item["question"]
            correct_answer = item["answer"]
            options = item["options"]
            
            # 构建完整问题（包含选项）
            full_question = f"{question}\n选项："
            for key, value in options.items():
                full_question += f"{key}. {value}\n"
            full_question += "请选择正确答案，只需输出选项字母。"
            
            try:
                # 使用RAG系统回答（测试模式，只输出答案）
                answer = rag_system.query(full_question, alpha=alpha, test_mode=True)
                
                # 提取答案（尝试从回答中提取选项字母）
                extracted_answer = extract_answer(answer)
                
                # 检查答案是否正确
                is_correct = extracted_answer == correct_answer
                if is_correct:
                    correct_count += 1
                
                # 保存结果
                results.append({
                    "序号": i,
                    "问题": question,
                    "选项": options,
                    "正确答案": correct_answer,
                    "模型结果": extracted_answer,
                    "模型回答": answer,
                    "测试结果": is_correct
                })
                
                print(f"第{i}题：{'正确' if is_correct else '错误'}，模型答案：{extracted_answer}，正确答案：{correct_answer}")
                
            except Exception as e:
                print(f"第{i}题处理失败：{str(e)}")
                results.append({
                    "序号": i,
                    "问题": question,
                    "选项": options,
                    "正确答案": correct_answer,
                    "模型结果": "错误",
                    "模型回答": str(e),
                    "测试结果": False
                })
        
        # 计算正确率
        accuracy = correct_count / total_count
        print(f"测试完成，alpha={alpha}，正确数：{correct_count}，总题数：{total_count}，正确率：{accuracy:.4f}")
        
        return accuracy
    
    # 测试前5题
    for alpha in alpha_values:
        test_rag_system_with_alpha_test(alpha)
    
    print("\n测试脚本验证完成，如需测试完整400题，请修改main函数。")
