import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 使用Hugging Face Embeddings和ChromaDB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from langchain_deepseek import ChatDeepSeek
from IPython.display import display, Markdown

import os
import tempfile
import urllib.request

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

if __name__ == "__main__":
    # 参数
    num_funds = 10
    start_date = pd.to_datetime('2025-01-01')
    csv_path = 'mutual_fund_data.csv'
    persist_directory = './chroma_db/'  # 修改为本地可写目录
    collection_name = "mutual_fund_optimization"

    # 检查CSV文件是否已存在
    if not os.path.exists(csv_path):
        print(f"CSV文件不存在，正在生成 {csv_path}...")
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
        df.to_csv(csv_path, index=False)
        print("CSV文件已生成")
    else:
        print(f"CSV文件已存在，正在加载 {csv_path}...")

    # 加载CSV文件
    df = pd.read_csv(csv_path)

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

    # 指定模型名称以避免警告
    hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 检查ChromaDB集合是否已存在
    try:
        # 尝试加载现有集合
        existing_db = Chroma(
            collection_name=collection_name,
            embedding_function=hg_embeddings,
            persist_directory=persist_directory
        )
        
        # 检查集合是否有数据
        if existing_db._collection.count() > 0:
            print(f"ChromaDB集合 '{collection_name}' 已存在，使用现有集合")
            langchain_chroma = existing_db
        else:
            print(f"ChromaDB集合 '{collection_name}' 存在但为空，创建新集合")
            langchain_chroma = Chroma.from_documents(
                documents=documents,
                collection_name=collection_name,
                embedding=hg_embeddings,
                persist_directory=persist_directory
            )
    except Exception as e:
        print(f"ChromaDB集合不存在或发生错误，创建新集合: {e}")
        # 初始化ChromaDB
        langchain_chroma = Chroma.from_documents(
            documents=documents,
            collection_name=collection_name,
            embedding=hg_embeddings,
            persist_directory=persist_directory
        )

    # Define the prompt template
    template = """
    基于以下共同基金数据，分析并仅提供每只基金各百块的百分比优化建议。
    共同基金信息：{question}
    上下文：{context}
    回答：
    """
    PROMPT = PromptTemplate(input_variables=["context", "query"], template=template)

    # Set up retriever
    retriever = langchain_chroma.as_retriever(search_kwargs={"k": 10})

    # Function to remove duplicates from retrieved documents
    def remove_duplicates(documents):
        seen = set()
        unique_docs = []
        for doc in documents:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)
        return unique_docs
    
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
    # Set up the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever, chain_type_kwargs={"prompt": PROMPT}
    )

    def get_optimized_recommendations(query):
        # Retrieve documents
        raw_docs = retriever.get_relevant_documents(query)

        # Remove duplicates
        unique_docs = remove_duplicates(raw_docs)

        # Prepare the context for the prompt
        context = " ".join([doc.page_content for doc in unique_docs])

        # Use the QA chain to get the response
        result = qa_chain({"context": context, "query": query})
        return result

    # Example query
    query = "分析并提供每个基金板块的优化推荐百分比。并按照r'基金编号: (\d+)\s+优化建议：科技 (\d+)%，医疗 (\d+)%，金融 (\d+)%，能源 (\d+)%'的格式返回结果。"
    response = get_optimized_recommendations(query)
    print(Markdown(response['result']))

    # 解析优化建议
    def parse_optimization_result(result_text):
        fund_pattern = r"基金编号: (\d+)\s+优化建议：科技 (\d+)%，医疗 (\d+)%，金融 (\d+)%，能源 (\d+)%"
        matches = re.findall(fund_pattern, result_text)
        
        optimized_data = []
        for match in matches:
            fund_id, tech, medical, finance, energy = match
            optimized_data.append({
                '基金编号': int(fund_id),
                '科技占比_优化_%': float(tech),
                '医疗占比_优化_%': float(medical),
                '金融占比_优化_%': float(finance),
                '能源占比_优化_%': float(energy)
            })
        
        return pd.DataFrame(optimized_data)

    # 解析模型返回的优化结果
    optimization_result = parse_optimization_result(response['result'])
    
    # 将优化结果合并到原始数据中
    merged_df = pd.merge(df, optimization_result, on='基金编号', how='inner')

    merged_df.to_csv('merged_fund_data.csv', index=False)
    

    # 获取一个可用的中文字体
    def get_chinese_font():
        # 尝试常见的中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'AR PL UMing CN', 
                    'WenQuanYi Zen Hei', 'Hiragino Sans GB', 'Noto Sans CJK SC', 
                    'Source Han Sans CN', 'Source Han Sans SC', 'PingFang SC']
        
        existing_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in chinese_fonts:
            if font in existing_fonts:
                return FontProperties(fname=fm.findfont(font))
        
        # 如果找不到中文字体，尝试使用默认字体
        return FontProperties()

    # 获取中文字体
    chinese_font = get_chinese_font()

    # 方法2：如果上述方法不行，尝试下载并使用免费的中文字体

    def download_chinese_font():
        # 下载文泉驿微米黑字体(开源中文字体)
        temp_dir = tempfile.gettempdir()
        font_path = os.path.join(temp_dir, 'wqy-microhei.ttc')
        
        if not os.path.exists(font_path):
            print("下载中文字体...")
            url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/wqy-microhei.ttc"
            urllib.request.urlretrieve(url, font_path)
            print(f"字体已下载到: {font_path}")
        
        return font_path

    # 如果方法1失败，使用方法2
    if chinese_font.get_name() == 'DejaVu Sans':
        print("找不到系统中文字体，正在下载中文字体...")
        font_path = download_chinese_font()
        chinese_font = FontProperties(fname=font_path)

    # 设置全局字体参数
    plt.rcParams['font.family'] = chinese_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 修正负号显示

    # 现在修改你的绘图代码，在每个标题和标签中明确指定字体
    def create_comparison_charts(merged_df):
        # 设置图表样式
        sns.set(style="whitegrid")
        
        # 处理的基金编号列表
        fund_ids = sorted(merged_df['基金编号'].unique())
        num_funds = len(fund_ids)
        
        # 创建横向布局的网格，两行多列
        fig = plt.figure(figsize=(14, 8))  # 调整为更适合横向布局的尺寸
        gs = GridSpec(2, num_funds, figure=fig)
        
        # 定义共享的颜色和标签
        labels = ['科技', '医疗', '金融', '能源']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # 第一行：所有基金的原始分配
        for col, fund_id in enumerate(fund_ids):
            fund_data = merged_df[merged_df['基金编号'] == fund_id]
            
            # 原始分配饼图
            ax1 = fig.add_subplot(gs[0, col])
            original_data = [
                fund_data['科技占比_%'].values[0],
                fund_data['医疗占比_%'].values[0],
                fund_data['金融占比_%'].values[0],
                fund_data['能源占比_%'].values[0]
            ]
            
            ax1.pie(original_data, autopct='%1.1f%%', startangle=90, colors=colors)
            ax1.axis('equal')
            ax1.set_title(f'基金 {fund_id} - 原始分配', fontproperties=chinese_font, fontsize=12)
        
        # 第二行：所有基金的优化分配
        for col, fund_id in enumerate(fund_ids):
            fund_data = merged_df[merged_df['基金编号'] == fund_id]
            
            # 优化分配饼图
            ax2 = fig.add_subplot(gs[1, col])
            optimized_data = [
                fund_data['科技占比_优化_%'].values[0],
                fund_data['医疗占比_优化_%'].values[0],
                fund_data['金融占比_优化_%'].values[0],
                fund_data['能源占比_优化_%'].values[0]
            ]
            
            ax2.pie(optimized_data, autopct='%1.1f%%', startangle=90, colors=colors)
            ax2.axis('equal')
            ax2.set_title(f'基金 {fund_id} - 优化分配', fontproperties=chinese_font, fontsize=12)
        
        # 添加总标题
        plt.suptitle('基金板块分配 - 原始分配与优化分配对比(模拟结果)', fontproperties=chinese_font, fontsize=16)
        
        # 创建图例的补丁
        patches = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(labels))]
        
        # 在图表右侧添加一个共享的图例
        fig.legend(patches, labels, loc='center right', 
                bbox_to_anchor=(0.98, 0.5), prop=chinese_font, fontsize=12)
        
        plt.tight_layout()
        # 为图例和标题腾出空间
        plt.subplots_adjust(top=0.9, right=0.92)
        
        # 保存图像
        plt.savefig('fund_allocation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 调用修改后的函数
    create_comparison_charts(merged_df)