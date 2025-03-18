import numpy as np
import pandas as pd
from faker import Faker
import random
import os
from langchain_deepseek import ChatDeepSeek
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 设置中文语言环境的Faker
fake = Faker('zh_CN')
np.random.seed(42)
random.seed(42)

def generate_chinese_financial_data(num_users):
    # 生成省份列表并过滤掉特别行政区
    provinces = []
    for _ in range(num_users):
        province = fake.province()
        while "香港特别行政区" in province or "澳门特别行政区" in province:
            province = fake.province()
        provinces.append(province)
    
    data = {
        '用户ID': [i for i in range(1, num_users + 1)],
        '年收入': np.random.uniform(60000, 600000, num_users).round(2),
        '年支出': np.random.uniform(30000, 300000, num_users).round(2),
        '医疗保险': np.random.uniform(0, 10000, num_users).round(2),
        '住房贷款': np.random.uniform(0, 60000, num_users).round(2),
        '商业养老保险': np.random.uniform(0, 20000, num_users).round(2),
        '企业年金': np.random.uniform(0, 12000, num_users).round(2),
        '住房公积金': np.random.uniform(0, 24000, num_users).round(2),
        '房租支出': np.random.uniform(0, 60000, num_users).round(2),
        '上年缴税金额': np.random.uniform(0, 50000, num_users).round(2),
        '省份': provinces,
        '婚姻状态': [random.choice(['未婚', '已婚', '离异']) for _ in range(num_users)],
        '税收抵免': np.random.uniform(0, 10000, num_users).round(2),
        '子女教育支出': np.random.uniform(0, 36000, num_users).round(2),
        '继续教育支出': np.random.uniform(0, 8000, num_users).round(2),
        '大病医疗支出': np.random.uniform(0, 40000, num_users).round(2),
        '赡养老人支出': np.random.uniform(0, 24000, num_users).round(2)
    }

    # 对某些字段随机设置为0
    for column in ['医疗保险', '住房贷款', '商业养老保险', '企业年金', '住房公积金', '房租支出', 
                   '子女教育支出', '继续教育支出', '大病医疗支出', '赡养老人支出']:
        data[column] = [value if random.random() > 0.5 else 0 for value in data[column]]

    df = pd.DataFrame(data)
    return df


def generate_chinese_tax_regulations():
    # 中国个人所得税累进税率表（2024年标准）
    tax_brackets = ['3% - ¥0至¥36,000', '10% - ¥36,001至¥144,000', '20% - ¥144,001至¥300,000',
                    '25% - ¥300,001至¥420,000', '30% - ¥420,001至¥660,000', '35% - ¥660,001至¥960,000',
                    '45% - ¥960,001以上']
    
    # 中国标准扣除额（每月5000元，这里显示年度总额）
    standard_deductions = [60000] * len(tax_brackets)
    
    # 中国专项附加扣除（假设值，实际根据个人情况不同）
    tax_deductions = [0, 12000, 24000, 36000, 48000, 60000, 72000]

    regulations = {
        '税率级别': tax_brackets,
        '基本减除费用(年)': standard_deductions,
        '专项附加扣除上限(年)': tax_deductions
    }
    
    df = pd.DataFrame(regulations)
    return df



# 应用中国税务规定到财务数据
def apply_chinese_tax_regulations(financial_df, regulations_df):
    # 简化的中国个人所得税计算模型
    def calculate_chinese_tax(income, deductions, standard_deduction):
        # 计算应纳税所得额 = 年收入 - 标准扣除额 - 专项扣除 - 专项附加扣除
        taxable_income = max(income - standard_deduction - deductions, 0)
        
        # 按照中国累进税率计算税额 (含速算扣除)
        if taxable_income <= 36000:
            tax = taxable_income * 0.03
        elif taxable_income <= 144000:
            tax = taxable_income * 0.1 - 2520
        elif taxable_income <= 300000:
            tax = taxable_income * 0.2 - 16920
        elif taxable_income <= 420000:
            tax = taxable_income * 0.25 - 31920
        elif taxable_income <= 660000:
            tax = taxable_income * 0.3 - 52920
        elif taxable_income <= 960000:
            tax = taxable_income * 0.35 - 85920
        else:
            tax = taxable_income * 0.45 - 181920
            
        return max(tax, 0)  # 确保税额不为负数

    # 获取标准扣除额
    standard_deduction = regulations_df['基本减除费用(年)'].iloc[0]

    # 计算每个用户的专项扣除总额
    financial_df['专项扣除总额'] = financial_df['医疗保险'] + financial_df['住房公积金']
    
    # 计算每个用户的专项附加扣除总额
    financial_df['专项附加扣除总额'] = financial_df[['子女教育支出', '继续教育支出', '大病医疗支出', 
                                     '住房贷款', '房租支出', '赡养老人支出']].sum(axis=1)
    
    # 计算每个用户的应缴税额
    financial_df['应缴税额'] = financial_df.apply(
        lambda row: calculate_chinese_tax(
            row['年收入'], 
            row['专项扣除总额'] + row['专项附加扣除总额'], 
            standard_deduction
        ),
        axis=1
    )
    
    # 计算实际缴税额（考虑税收抵免）
    financial_df['实际缴税额'] = financial_df.apply(
        lambda row: max(row['应缴税额'] - row['税收抵免'], 0),
        axis=1
    )
    
    # 计算税后收入
    financial_df['税后收入'] = financial_df['年收入'] - financial_df['实际缴税额']
    
    # 计算税负率
    financial_df['税负率'] = (financial_df['实际缴税额'] / financial_df['年收入'] * 100).round(2)
    
    return financial_df




import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 为LangChain准备文档
def prepare_documents_for_chroma(financial_data_with_taxes):
    documents = []
    for _, row in financial_data_with_taxes.iterrows():
        content = (f"用户ID: {row['用户ID']}, 年收入: {row['年收入']}, 年支出: {row['年支出']}, "
                  f"医疗保险: {row['医疗保险']}, 住房贷款: {row['住房贷款']}, "
                  f"商业养老保险: {row['商业养老保险']}, 企业年金: {row['企业年金']}, 住房公积金: {row['住房公积金']}, "
                  f"房租支出: {row['房租支出']}, 上年缴税金额: {row['上年缴税金额']}, 省份: {row['省份']}, "
                  f"婚姻状态: {row['婚姻状态']}, 税收抵免: {row['税收抵免']}, 子女教育支出: {row['子女教育支出']}, "
                  f"继续教育支出: {row['继续教育支出']}, 大病医疗支出: {row['大病医疗支出']}, 赡养老人支出: {row['赡养老人支出']}, "
                  f"专项扣除总额: {row['专项扣除总额']}, 专项附加扣除总额: {row['专项附加扣除总额']}, "
                  f"应缴税额: {row['应缴税额']}, 实际缴税额: {row['实际缴税额']}, 税后收入: {row['税后收入']}, 税负率: {row['税负率']}")

        metadata = {
            "用户ID": str(row['用户ID']),
            "年收入": float(row['年收入']),
            "省份": row['省份'],
            "婚姻状态": row['婚姻状态'],
            "税负率": float(row['税负率'])
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

def create_chroma_db(chinese_financial_data_with_taxes, model_name='moka-ai/m3e-base'):
    # 使用中文模型作为嵌入模型，例如M3E
    hg_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    persist_directory = './chinese_financial_chroma_db'
    
    # 检查数据库是否已经存在
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        try:
            # 尝试加载现有的数据库
            langchain_chroma = Chroma(
                collection_name="chinese_financial_data",
                embedding_function=hg_embeddings,
                persist_directory=persist_directory
            )
            print("已加载现有的Chroma向量数据库")
            return langchain_chroma
        except Exception as e:
            print(f"加载现有数据库失败: {e}")
            print("将创建新的数据库...")
    
    # 准备文档
    documents = prepare_documents_for_chroma(chinese_financial_data_with_taxes)
    
    # 创建并持久化Chroma数据库
    langchain_chroma = Chroma.from_documents(
        documents=documents,
        collection_name="chinese_financial_data",
        embedding=hg_embeddings,
        persist_directory=persist_directory
    )
    
    print(f"已创建包含{len(documents)}条财务数据记录的Chroma向量数据库")
    return langchain_chroma

def query_chroma_db(langchain_chroma, query_text, k=5):
    """
    根据查询文本搜索相似的财务数据记录
    
    参数:
        langchain_chroma: Chroma数据库实例
        query_text: 查询文本
        k: 返回的结果数量
    
    返回:
        相似的文档及其相似度分数
    """
    results = langchain_chroma.similarity_search_with_score(query_text, k=k)
    return results


# Function to remove duplicates from retrieved documents
def remove_duplicates(documents):
    seen = set()
    unique_docs = []
    for doc in documents:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)
    return unique_docs



def get_tax_optimization_recommendations(query):
    # Retrieve documents
    raw_docs = retriever.get_relevant_documents(query)

    # Remove duplicates
    unique_docs = remove_duplicates(raw_docs)

    # Prepare the context for the prompt
    context = " ".join([doc.page_content for doc in unique_docs])

    # Use the QA chain to get the response
    result = qa_chain({"context": context, "query": query})
    return result



if __name__ == "__main__":
    # 生成财务数据
    num_users = 1000
    chinese_financial_data = generate_chinese_financial_data(num_users)
    chinese_tax_regulations = generate_chinese_tax_regulations()
    chinese_financial_data_with_taxes = apply_chinese_tax_regulations(chinese_financial_data, chinese_tax_regulations)

    chinese_financial_data_with_taxes.to_csv('chinese_financial_data_with_taxes.csv', index=False)

    # 创建Chroma向量数据库
    chroma_db = create_chroma_db(chinese_financial_data_with_taxes)
    
    # 用户输入查询文本和k值，保持默认值
    default_query = "高收入已婚人士的税负情况"
    default_k = 3
    
    user_query = input(f"请输入查询文本 (默认: '{default_query}'): ")
    if not user_query.strip():
        user_query = default_query
    
    try:
        user_k = input(f"请输入返回结果数量k (默认: {default_k}): ")
        if not user_k.strip():
            user_k = default_k
        else:
            user_k = int(user_k)
    except ValueError:
        print(f"输入的k值无效，使用默认值: {default_k}")
        user_k = default_k
    
    # 执行查询
    query_results = query_chroma_db(chroma_db, user_query, k=user_k)
    
    # 显示查询结果
    print("\n查询结果:")
    for doc, score in query_results:
        print(f"相似度分数: {score}")
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}\n")


    if not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "sk-7c87ef2add054e439095db9b18c921e9"

    llm = ChatDeepSeek(
        model="deepseek-reasoner",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    template = """
    根据以下财务数据和税收法规，为当前查询用户分析并提供个性化的节税建议：
    其他用户财务数据：{question}
    当前查询用户信息{context}
    回答：
        """
    PROMPT = PromptTemplate(input_variables=["context", "query"], template=template)

    # Set up retriever
    retriever = chroma_db.as_retriever(search_kwargs={"k": 10})

    # Set up the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever, chain_type_kwargs={"prompt": PROMPT}
    )

    default_tax_query = "分析 - 当前查询用户: 年收入: 287569.26, 年支出: 179052.74, 医疗保险: 4128.43, 住房贷款: 0, 商业养老保险: 商业养老保险, 企业年金: 0, 住房公积金: 6928.65, 房租支出: 1870.99, 上年缴税金额: 36409.44, 省份: 甘肃省, 婚姻状态: 离异, 税收抵免: 3609.74, 子女教育支出: 0, 继续教育支出: 5644.6, 大病医疗支出: 37770.66, 赡养老人支出: 0.0, 专项扣除总额: 6928.65, 专项附加扣除总额:45286.25, 应缴税额: 18150.872, 实际缴税额: 14541.132, 税后收入: 273028.128, 税负率: 5.06"


    # 获取用户输入的查询
    user_tax_query = input(f"请输入需要分析的财务数据 (默认: '{default_tax_query}'): ")
    if not user_tax_query.strip():
        user_tax_query = default_tax_query

    # 使用用户输入或默认值获取税务优化建议
    response = get_tax_optimization_recommendations(user_tax_query)
    print("\n税务优化建议:")
    print(response['result'])
