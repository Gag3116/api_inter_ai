from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import spacy
from spacy.matcher import PhraseMatcher
import json

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 确保 spaCy 模型已安装
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 加载药品数据
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
medicine_file_path = os.path.join(BASE_DIR, "medicine.json")
try:
    with open(medicine_file_path, "r") as f:
        medicine_data = json.load(f)["medications"]
except FileNotFoundError:
    raise FileNotFoundError(f"{medicine_file_path} 文件未找到，请确保文件路径正确。")

# 从药品数据提取症状关键词
symptom_keywords = list({symptom for med in medicine_data for symptom in med["symptoms"]})
matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp.make_doc(symptom) for symptom in symptom_keywords]
matcher.add("SYMPTOMS", patterns)

# 否定检测函数
def is_negated(token):
    """
    检查 token 是否被否定修饰，支持直接否定、间接否定和双重否定。
    确保否定词唯一性，避免重复计数。
    """
    negation_words = {"no", "not", "n't", "don't", "doesn't", "never", "without", "lack"}
    negation_tokens = set()  # 使用集合存储否定词，确保唯一性

    # 1. 检查父节点的子节点是否有否定修饰
    for child in token.head.children:
        if child.dep_ == "neg" or child.lower_ in negation_words:
            negation_tokens.add(child.text)

    # 2. 检查路径上的祖先节点是否有否定修饰
    for ancestor in token.ancestors:
        if ancestor.lower_ in negation_words:
            negation_tokens.add(ancestor.text)
        for child in ancestor.children:
            if child.dep_ == "neg" or child.lower_ in negation_words:
                negation_tokens.add(child.text)

    # 3. 处理间接否定：检查症状词的整个子树中是否包含否定词
    for descendant in token.subtree:
        if descendant.lower_ in negation_words:
            negation_tokens.add(descendant.text)

    # # 4. 输出调试信息
    # print(f"[调试] 检测症状 '{token.text}'")
    # print(f"[调试] 否定词数量: {len(negation_tokens)}")
    # print(f"[调试] 否定词列表: {list(negation_tokens)}")

    # 5. 根据否定词数量判断是否为否定
    return len(negation_tokens) % 2 == 1  # 奇数为否定，偶数为肯定（双重否定）

# 检查对比连词后的状态
def check_contrast_and_status(doc):
    contrast_words = {"but", "however"}
    fine_words = {"fine", "better", "well", "okay", "recovered"}
    for token in doc:
        if token.text.lower() in contrast_words:
            for descendant in token.subtree:
                if descendant.text.lower() in fine_words:
                    return True
    return False

# 时态检测函数
def get_tense(token):
    verb = token.head
    if verb.tag_ in {"VBD", "VBN"}:
        return "past"
    elif verb.tag_ in {"VBZ", "VBP"}:
        return "present"
    elif verb.text in {"will", "going"}:
        return "future"
    return "unknown"

# 解析用户输入
def parse_input_function(user_input):
    doc = nlp(user_input.lower())
    detected_symptoms = set()
    processed_tokens = set()
    resolved_to_fine = check_contrast_and_status(doc)

    matches = matcher(doc)
    for match_id, start, end in matches:
        symptom = doc[start:end].text
        symptom_root = doc[start:end].root

        if symptom_root in processed_tokens:
            continue

        negated = is_negated(symptom_root)
        tense = get_tense(symptom_root)
        current = tense != "past"

        if resolved_to_fine:
            current = False

        if current and not negated:
            detected_symptoms.add(symptom)

        processed_tokens.add(symptom_root)

    return list(detected_symptoms)

# 药品推荐函数
def recommend_medications(symptoms):
    """
    根据症状推荐药品，并将症状整合到对应药品中
    Args:
        symptoms (list): 症状列表
    Returns:
        list: 推荐的药品及其所有对应症状
    """
    recommendations = {}

    for symptom in symptoms:
        for med in medicine_data:
            if symptom in med["symptoms"]:
                if med["name"] not in recommendations:
                    # 如果药品未添加到结果中，初始化
                    recommendations[med["name"]] = {"medication": med["name"], "symptoms": []}
                # 将症状添加到药品的症状列表中
                recommendations[med["name"]]["symptoms"].append(symptom)

    # 转换字典为列表
    return list(recommendations.values())

# 健康检查接口
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# 综合解析与推荐接口
@app.route('/parse_and_recommend', methods=['POST'])
def parse_and_recommend():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    user_input = data.get("input", "")

    if not user_input:
        return jsonify({"error": "Input field is empty"}), 400

    symptoms = parse_input_function(user_input)
    recommendations = recommend_medications(symptoms)

    return jsonify({
        "recommendations": recommendations
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
