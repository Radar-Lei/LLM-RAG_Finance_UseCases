import sys
import os
import numpy as np
import pandas as pd
from faker import Faker
import random
import markdown
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QTextEdit, QTextBrowser,
                           QLineEdit, QSpinBox, QGroupBox, QFileDialog,
                           QMessageBox, QGridLayout)
from PyQt6.QtGui import QPixmap, QColor, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize

# Original imports from TaxOpt.py
from langchain_deepseek import ChatDeepSeek
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# 定义常量
APP_TITLE = "个税优化AI助手"
THEME_COLOR = QColor(172, 52, 32, 230)  # #AC3420 with 90% opacity
THEME_COLOR_STR = "#AC3420"
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
LOGO_PATH = "figs/四川大学商学院.png"
AUTHOR = "作者：雷达"
ACCENT_COLOR = "#D9534F"  # 辅助颜色，比主题色更柔和
BACKGROUND_COLOR = "#FFFFFF"  # 背景色
TEXT_COLOR = "#333333"  # 文字颜色

# 从原始TaxOpt.py文件导入的函数
# 设置中文语言环境的Faker
fake = Faker('zh_CN')

def generate_chinese_financial_data(num_users):
    np.random.seed(42)
    random.seed(42)
    
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


# 后台线程用于数据生成和处理
class DataGenerationThread(QThread):
    update_signal = pyqtSignal(pd.DataFrame)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    
    def __init__(self, num_users):
        super().__init__()
        self.num_users = num_users
        
    def run(self):
        try:
            self.progress_signal.emit("正在生成财务数据...")
            financial_data = generate_chinese_financial_data(self.num_users)
            
            self.progress_signal.emit("正在生成税务规定...")
            tax_regulations = generate_chinese_tax_regulations()
            
            self.progress_signal.emit("正在应用税务规定到财务数据...")
            financial_data_with_taxes = apply_chinese_tax_regulations(financial_data, tax_regulations)
            
            # 保存数据
            financial_data_with_taxes.to_csv('chinese_financial_data_with_taxes.csv', index=False)
            
            self.update_signal.emit(financial_data_with_taxes)
            
        except Exception as e:
            self.error_signal.emit(f"数据生成失败: {str(e)}")


class ChromaDbCreationThread(QThread):
    update_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    
    def __init__(self, financial_data_with_taxes):
        super().__init__()
        self.financial_data_with_taxes = financial_data_with_taxes
        
    def run(self):
        try:
            self.progress_signal.emit("正在创建Chroma向量数据库...")
            chroma_db = create_chroma_db(self.financial_data_with_taxes)
            self.progress_signal.emit("Chroma向量数据库创建完成")
            self.update_signal.emit(chroma_db)
            
        except Exception as e:
            self.error_signal.emit(f"创建Chroma数据库失败: {str(e)}")


class QueryThread(QThread):
    update_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    
    def __init__(self, chroma_db, query_text, k):
        super().__init__()
        self.chroma_db = chroma_db
        self.query_text = query_text
        self.k = k
        
    def run(self):
        try:
            self.progress_signal.emit("正在查询数据库...")
            results = query_chroma_db(self.chroma_db, self.query_text, self.k)
            self.progress_signal.emit("查询完成")
            self.update_signal.emit(results)
            
        except Exception as e:
            self.error_signal.emit(f"查询失败: {str(e)}")


class OptimizationThread(QThread):
    update_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    
    def __init__(self, chroma_db, query_text, api_key):
        super().__init__()
        self.chroma_db = chroma_db
        self.query_text = query_text
        self.api_key = api_key
        
    def run(self):
        try:
            if not os.getenv("DEEPSEEK_API_KEY"):
                os.environ["DEEPSEEK_API_KEY"] = self.api_key
                
            self.progress_signal.emit("正在初始化LLM...")
            llm = ChatDeepSeek(
                model="deepseek-reasoner",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            
            template = """
            根据以下财务数据和税收法规，为当前查询用户分析并提供个性化的节税建议：
            其他用户财务数据：{context}
            当前查询用户信息：{question}
            回答：
            """
            PROMPT = PromptTemplate(input_variables=["context", "query"], template=template)
            
            # 设置检索器
            self.progress_signal.emit("正在设置检索器...")
            retriever = self.chroma_db.as_retriever(search_kwargs={"k": 10})
            
            # 设置QA链
            qa_chain = RetrievalQA.from_chain_type(
                llm, retriever=retriever, chain_type_kwargs={"prompt": PROMPT}
            )
            
            # 检索文档
            self.progress_signal.emit("正在检索相关文档...")
            raw_docs = retriever.get_relevant_documents(self.query_text)
            
            # 去除重复
            unique_docs = remove_duplicates(raw_docs)
            
            # 为提示准备上下文
            context = " ".join([doc.page_content for doc in unique_docs])
            
            # 使用QA链获取响应
            self.progress_signal.emit("正在生成税务优化建议...")
            result = qa_chain({"context": context, "query": self.query_text})
            self.update_signal.emit(result['result'])
            
        except Exception as e:
            self.error_signal.emit(f"获取税务优化建议失败: {str(e)}")


class TaxOptApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.financial_data = None
        self.chroma_db = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(APP_TITLE)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # 设置样式表
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BACKGROUND_COLOR};
                border: 2px solid {THEME_COLOR_STR};
            }}
            QWidget {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
            }}
            QPushButton {{
                background-color: {ACCENT_COLOR};
                color: {BACKGROUND_COLOR};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {THEME_COLOR_STR};
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #888888;
            }}
            QLineEdit, QTextEdit, QSpinBox {{
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 4px;
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 10px;
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
            }}
        """)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部布局 - 包含标题和徽标
        top_layout = QHBoxLayout()
        
        # 添加徽标（如果存在）
        if os.path.exists(LOGO_PATH):
            logo_label = QLabel()
            logo_pixmap = QPixmap(LOGO_PATH)
            logo_size = QSize(200, 60)
            scaled_logo = logo_pixmap.scaled(logo_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
            logo_label.setFixedSize(logo_size)
            top_layout.addWidget(logo_label, 0, Qt.AlignmentFlag.AlignLeft)
        
        # 添加标题
        title_label = QLabel(APP_TITLE)
        title_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {THEME_COLOR_STR};
        """)
        top_layout.addWidget(title_label, 0, Qt.AlignmentFlag.AlignHCenter)
        
        # 添加作者标签
        author_label = QLabel(AUTHOR)
        author_label.setStyleSheet("""
            font-size: 14px;
            font-style: italic;
        """)
        top_layout.addWidget(author_label, 0, Qt.AlignmentFlag.AlignRight)
        
        # 添加顶部布局到主布局
        main_layout.addLayout(top_layout)
        
        # 设置数据生成组
        data_group = QGroupBox("财务数据生成")
        data_layout = QGridLayout(data_group)
        
        # 用户数量选择
        data_layout.addWidget(QLabel("生成用户数量:"), 0, 0)
        self.num_users_spinbox = QSpinBox()
        self.num_users_spinbox.setRange(10, 10000)
        self.num_users_spinbox.setValue(1000)  # 默认值与原代码一致
        data_layout.addWidget(self.num_users_spinbox, 0, 1)
        
        # 生成按钮
        self.generate_button = QPushButton("生成数据")
        self.generate_button.clicked.connect(self.generate_data)
        data_layout.addWidget(self.generate_button, 0, 2)
        
        # 创建数据库按钮
        self.create_db_button = QPushButton("创建数据库")
        self.create_db_button.clicked.connect(self.create_database)
        self.create_db_button.setEnabled(False)  # 初始禁用，直到生成数据
        data_layout.addWidget(self.create_db_button, 0, 3)
        
        main_layout.addWidget(data_group)
        
        # 设置查询组
        query_group = QGroupBox("财务数据查询")
        query_layout = QGridLayout(query_group)
        
        # 查询文本输入
        query_layout.addWidget(QLabel("查询文本:"), 0, 0)
        self.query_input = QLineEdit("高收入已婚人士的税负情况")  # 默认值与原代码一致
        query_layout.addWidget(self.query_input, 0, 1, 1, 2)
        
        # 结果数量选择
        query_layout.addWidget(QLabel("返回结果数量:"), 1, 0)
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setRange(1, 20)
        self.k_spinbox.setValue(3)  # 默认值与原代码一致
        query_layout.addWidget(self.k_spinbox, 1, 1)
        
        # 查询按钮
        self.query_button = QPushButton("查询")
        self.query_button.clicked.connect(self.query_database)
        self.query_button.setEnabled(False)  # 初始禁用，直到创建数据库
        query_layout.addWidget(self.query_button, 1, 2)
        
        main_layout.addWidget(query_group)
        
        # 设置税务优化组
        optimization_group = QGroupBox("税务优化建议")
        optimization_layout = QGridLayout(optimization_group)
        
        # 用户信息输入
        optimization_layout.addWidget(QLabel("用户财务信息:"), 0, 0)
        self.optimization_input = QTextEdit()
        self.optimization_input.setPlaceholderText("请输入需要分析的用户财务信息...")
        default_tax_query = "当前查询用户: 年收入: 287569.26, 年支出: 179052.74, 医疗保险: 4128.43, 住房贷款: 0, 商业养老保险: 0, 企业年金: 0, 住房公积金: 6928.65, 房租支出: 1870.99, 上年缴税金额: 36409.44, 省份: 甘肃省, 婚姻状态: 离异, 税收抵免: 3609.74, 子女教育支出: 0, 继续教育支出: 5644.6, 大病医疗支出: 37770.66, 赡养老人支出: 0.0"
        self.optimization_input.setText(default_tax_query)  # 使用原代码中的示例
        optimization_layout.addWidget(self.optimization_input, 1, 0, 1, 4)
        
        # API密钥输入
        optimization_layout.addWidget(QLabel("DeepSeek API 密钥:"), 2, 0)
        self.api_key_input = QLineEdit("sk-7c87ef2add054e439095db9b18c921e9")  # 使用原代码中的默认值
        optimization_layout.addWidget(self.api_key_input, 2, 1, 1, 2)
        
        # 优化按钮
        self.optimize_button = QPushButton("获取税务优化建议")
        self.optimize_button.clicked.connect(self.get_optimization)
        self.optimize_button.setEnabled(False)  # 初始禁用，直到创建数据库
        optimization_layout.addWidget(self.optimize_button, 2, 3)
        
        main_layout.addWidget(optimization_group)
        
        # 添加输出区域
        output_group = QGroupBox("结果输出")
        output_layout = QVBoxLayout(output_group)
        
        self.output_browser = QTextBrowser()
        self.output_browser.setOpenExternalLinks(True)  # 允许打开外部链接
        output_layout.addWidget(self.output_browser)
        
        main_layout.addWidget(output_group)
        
        # 添加状态栏
        self.statusBar().showMessage("就绪")
        self.statusBar().setStyleSheet(f"""
            background-color: {THEME_COLOR_STR};
            color: white;
            padding: 5px;
        """)
        
        # 检查缓存的财务数据是否存在
        if os.path.exists('chinese_financial_data_with_taxes.csv'):
            try:
                self.financial_data = pd.read_csv('chinese_financial_data_with_taxes.csv')
                self.statusBar().showMessage(f"已加载缓存的财务数据: {len(self.financial_data)} 条记录")
                self.create_db_button.setEnabled(True)
                
                # 尝试加载现有的Chroma数据库
                if os.path.exists('./chinese_financial_chroma_db') and os.path.isdir('./chinese_financial_chroma_db'):
                    try:
                        hg_embeddings = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base')
                        self.chroma_db = Chroma(
                            collection_name="chinese_financial_data",
                            embedding_function=hg_embeddings,
                            persist_directory='./chinese_financial_chroma_db'
                        )
                        self.statusBar().showMessage("已加载现有的Chroma向量数据库")
                        self.query_button.setEnabled(True)
                        self.optimize_button.setEnabled(True)
                    except Exception as e:
                        self.statusBar().showMessage(f"加载现有Chroma数据库失败: {str(e)}")
            except Exception as e:
                self.statusBar().showMessage(f"加载缓存的财务数据失败: {str(e)}")
    
    def generate_data(self):
        # 获取用户数量
        num_users = self.num_users_spinbox.value()
        
        # 禁用按钮并显示状态
        self.generate_button.setEnabled(False)
        self.statusBar().showMessage("正在生成数据...")
        self.output_browser.clear()
        self.output_browser.append("正在生成财务数据，请稍候...\n")
        
        # 创建并启动线程
        self.data_thread = DataGenerationThread(num_users)
        self.data_thread.update_signal.connect(self.update_data)
        self.data_thread.error_signal.connect(self.show_error)
        self.data_thread.progress_signal.connect(self.show_progress)
        self.data_thread.finished.connect(lambda: self.generate_button.setEnabled(True))
        self.data_thread.start()
    
    def update_data(self, df):
        self.financial_data = df
        self.statusBar().showMessage(f"已生成 {len(df)} 条财务数据")
        
        # 显示数据摘要
        self.output_browser.clear()
        self.output_browser.append(f"已生成 {len(df)} 条财务数据\n")
        
        if not df.empty:
            sample_size = min(5, len(df))
            self.output_browser.append(f"\n数据样本 (显示前 {sample_size} 条):\n")
            
            for idx, row in df.head(sample_size).iterrows():
                self.output_browser.append(f"用户ID: {row['用户ID']}")
                self.output_browser.append(f"年收入: {row['年收入']:.2f}")
                self.output_browser.append(f"年支出: {row['年支出']:.2f}")
                self.output_browser.append(f"省份: {row['省份']}")
                self.output_browser.append(f"婚姻状态: {row['婚姻状态']}")
                if '税负率' in row:
                    self.output_browser.append(f"税负率: {row['税负率']:.2f}%")
                self.output_browser.append("----------\n")
        
        # 启用创建数据库按钮
        self.create_db_button.setEnabled(True)
    
    def create_database(self):
        if self.financial_data is None or self.financial_data.empty:
            QMessageBox.warning(self, "错误", "没有可用的财务数据")
            return
        
        # 禁用按钮并显示状态
        self.create_db_button.setEnabled(False)
        self.statusBar().showMessage("正在创建Chroma向量数据库...")
        self.output_browser.clear()
        self.output_browser.append("正在创建Chroma向量数据库，这可能需要几分钟时间...\n")
        
        # 创建并启动线程
        self.db_thread = ChromaDbCreationThread(self.financial_data)
        self.db_thread.update_signal.connect(self.update_db)
        self.db_thread.error_signal.connect(self.show_error)
        self.db_thread.progress_signal.connect(self.show_progress)
        self.db_thread.finished.connect(lambda: self.create_db_button.setEnabled(True))
        self.db_thread.start()
    
    def update_db(self, chroma_db):
        self.chroma_db = chroma_db
        self.statusBar().showMessage("Chroma向量数据库创建完成")
        
        # 显示数据库创建成功信息
        self.output_browser.append("Chroma向量数据库创建成功\n")
        
        # 启用查询和优化按钮
        self.query_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
    
    def query_database(self):
        if self.chroma_db is None:
            QMessageBox.warning(self, "错误", "没有可用的Chroma数据库")
            return
        
        # 获取查询参数
        query_text = self.query_input.text().strip()
        if not query_text:
            QMessageBox.warning(self, "错误", "请输入查询文本")
            return
        
        k = self.k_spinbox.value()
        
        # 禁用按钮并显示状态
        self.query_button.setEnabled(False)
        self.statusBar().showMessage("正在查询数据库...")
        self.output_browser.clear()
        self.output_browser.append("正在查询数据库，请稍候...\n")
        
        # 创建并启动线程
        self.query_thread = QueryThread(self.chroma_db, query_text, k)
        self.query_thread.update_signal.connect(self.update_query_results)
        self.query_thread.error_signal.connect(self.show_error)
        self.query_thread.progress_signal.connect(self.show_progress)
        self.query_thread.finished.connect(lambda: self.query_button.setEnabled(True))
        self.query_thread.start()
    
    def update_query_results(self, results):
        self.statusBar().showMessage(f"查询完成，找到 {len(results)} 条结果")
        
        # 显示查询结果
        self.output_browser.clear()
        self.output_browser.append(f"查询结果 (共 {len(results)} 条):\n")
        
        for i, (doc, score) in enumerate(results):
            self.output_browser.append(f"结果 {i+1}:")
            self.output_browser.append(f"相似度分数: {score}")
            self.output_browser.append(f"内容: {doc.page_content}")
            self.output_browser.append(f"元数据: {doc.metadata}\n")
    
    def get_optimization(self):
        if self.chroma_db is None:
            QMessageBox.warning(self, "错误", "没有可用的Chroma数据库")
            return
        
        # 获取用户信息和API密钥
        query_text = self.optimization_input.toPlainText().strip()
        if not query_text:
            QMessageBox.warning(self, "错误", "请输入需要分析的用户财务信息")
            return
        
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "错误", "请输入DeepSeek API密钥")
            return
        
        # 禁用按钮并显示状态
        self.optimize_button.setEnabled(False)
        self.statusBar().showMessage("正在获取税务优化建议...")
        self.output_browser.clear()
        self.output_browser.append("正在获取税务优化建议，这可能需要几分钟时间...\n")
        
        # 创建并启动线程
        self.optimization_thread = OptimizationThread(self.chroma_db, query_text, api_key)
        self.optimization_thread.update_signal.connect(self.update_optimization_results)
        self.optimization_thread.error_signal.connect(self.show_error)
        self.optimization_thread.progress_signal.connect(self.show_progress)
        self.optimization_thread.finished.connect(lambda: self.optimize_button.setEnabled(True))
        self.optimization_thread.start()
    
    def update_optimization_results(self, result):
        self.statusBar().showMessage("税务优化建议生成完成")
        
        # 显示优化建议
        self.output_browser.clear()
        
        # 将markdown转换为HTML并显示
        html_content = markdown.markdown(result, extensions=['tables'])
        
        # 添加CSS样式，使HTML内容更美观
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 10px; }}
                h1, h2, h3, h4, h5, h6 {{ color: {THEME_COLOR_STR}; margin-top: 20px; }}
                h3 {{ border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                ul, ol {{ margin-left: 20px; }}
                li {{ margin: 5px 0; }}
                code {{ background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h3>税务优化建议:</h3>
            {html_content}
        </body>
        </html>
        """
        
        self.output_browser.setHtml(styled_html)
    
    def show_error(self, error_message):
        # 显示错误信息
        QMessageBox.critical(self, "错误", error_message)
        self.statusBar().showMessage(f"发生错误: {error_message}")
        self.output_browser.append(f"错误: {error_message}")
    
    def show_progress(self, progress_message):
        # 显示进度信息
        self.output_browser.append(progress_message)
        self.statusBar().showMessage(progress_message)


# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 应用程序样式
    app.setStyle("Fusion")
    
    # 显示应用程序
    main_window = TaxOptApp()
    main_window.show()
    
    sys.exit(app.exec())
