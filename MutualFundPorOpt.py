import pandas as pd
import numpy as np
# 使用Hugging Face Embeddings和ChromaDB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document




if __name__ == "__main__":

    # 参数
    num_funds = 10
    start_date = pd.to_datetime('2025-01-01')

    # 生成数据
    data = []

    for fund_id in range(1, num_funds + 1):
        # 为每只基金生成汇总数据
        avg_nav = np.round(np.random.uniform(10, 100), 2)
        avg_return = np.round(np.random.normal(0.05, 0.02), 2)
        risk_level = np.random.choice(['低', '中', '高'])
        sector_exposure = {
            '科技': np.round(np.random.uniform(10, 50), 2),
            '医疗': np.round(np.random.uniform(10, 50), 2),
            '金融': np.round(np.random.uniform(10, 50), 2),
            '能源': np.round(np.random.uniform(5, 20), 2)
        }
        avg_interest_rate = np.round(np.random.uniform(0.5, 5), 2)
        avg_inflation_rate = np.round(np.random.uniform(1, 4), 2)

        data.append({
            '基金编号': fund_id,
            '日期': start_date,
            '平均净值': avg_nav,
            '平均收益率_%': avg_return,
            '风险等级': risk_level,
            '科技占比_%': sector_exposure['科技'],
            '医疗占比_%': sector_exposure['医疗'],
            '金融占比_%': sector_exposure['金融'],
            '能源占比_%': sector_exposure['能源'],
            '平均利率_%': avg_interest_rate,
            '平均通胀率_%': avg_inflation_rate
        })

    # 转换为DataFrame
    df = pd.DataFrame(data)
    

    # 为LangChain准备文档
    documents = []
    for _, row in df.iterrows():
        content = f"基金编号: {row['基金编号']}, 日期: {row['日期']}, 净值: {row['平均净值']}, " \
                f"收益率: {row['平均收益率_%']}, 风险等级: {row['风险等级']}, " \
                f"科技占比: {row['科技占比_%']}, " \
                f"医疗占比: {row['医疗占比_%']}, " \
                f"金融占比: {row['金融占比_%']}, " \
                f"能源占比: {row['能源占比_%']}, " \
                f"利率: {row['平均利率_%']}, 通胀率: {row['平均通胀率_%']}"

        documents.append(Document(page_content=content))



    hg_embeddings = HuggingFaceEmbeddings()
    persist_directory = '/content/'

    # 初始化ChromaDB
    langchain_chroma = Chroma.from_documents(
        documents=documents,
        collection_name="mutual_fund_optimization",
        embedding=hg_embeddings,
        persist_directory=persist_directory
    )

