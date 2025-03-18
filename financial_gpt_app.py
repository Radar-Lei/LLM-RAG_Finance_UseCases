import sys
import os
import pandas as pd
import markdown
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QTextEdit, QTextBrowser,
                           QLineEdit, QComboBox, QSpinBox, QTabWidget, 
                           QGridLayout, QGroupBox, QCheckBox, QFileDialog,
                           QMessageBox, QSplashScreen, QScrollArea)
from PyQt6.QtGui import QPixmap, QIcon, QColor, QPalette, QFont, QTextCursor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize

# Original imports
import requests
from newsapi import NewsApiClient
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 定义常量
APP_TITLE = "金融资讯分析AI助手"
THEME_COLOR = QColor(172, 52, 32, 230)  # #AC3420 with 90% opacity
THEME_COLOR_STR = "#AC3420"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
LOGO_PATH = "figs/四川大学商学院.png"
AUTHOR = "作者：雷达"
ACCENT_COLOR = "#D9534F"  # 辅助颜色，比主题色更柔和
BACKGROUND_COLOR = "#FFFFFF"  # 背景色
TEXT_COLOR = "#333333"  # 文字颜色

# 沿用原始代码的函数
def fetch_news(query, from_date, to_date, language='en', sort_by='relevancy', page_size=30, api_key='YOUR_API_KEY'):
    # Initialize the NewsAPI client
    newsapi = NewsApiClient(api_key=api_key)
    query = query.replace(' ','&')
    # Fetch all articles matching the query
    all_articles = newsapi.get_everything(
        q=query,
        from_param=from_date,
        to=to_date,
        language=language,
        sort_by=sort_by,
        page_size=page_size
    )

    # Extract articles
    articles = all_articles.get('articles', [])

    # Convert to DataFrame
    if articles:
        df = pd.DataFrame(articles)
        return df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no articles are found

def preprocess_news_data(df):
    # Convert publishedAt to datetime
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df = df[~df['author'].isna()]
    
    # Keep the keyword column if it exists
    columns_to_keep = ['author', 'title', 'description', 'publishedAt']
    if 'keyword' in df.columns:
        columns_to_keep.append('keyword')
    
    df = df[columns_to_keep]
    return df

def build_prompt_for_summary(news_df, question):
    prompt = f"你是一位金融分析师，负责提供与金融行业相关的最新新闻文章的洞察。以下是一些关于{question}最新的新闻文章：\n\n"
    
    for index, row in news_df.iterrows():
        title = row['title']
        description = row['description']
        published_time = row['publishedAt']
        keyword = row.get('keyword', '')  # Get the keyword if it exists
        
        if 'keyword' in news_df.columns:
            prompt += f"**关键词:** {keyword}, **新闻标题:** {title}, **新闻内容:** {description}, **新闻发布时间:** {published_time}\n\n"
        else:
            prompt += f"**新闻标题:** {title}, **新闻内容:** {description}, **新闻发布时间:** {published_time}\n\n"
    
    prompt += f"请分析这些新闻，并提供关于这些新闻关于{question}可能对金融行业产生的影响，按照不同的关键词分别进行分析，注意标注新闻来源及时间"
    
    return prompt

# 后台线程用于处理长时间运行的任务
class FetchNewsThread(QThread):
    update_signal = pyqtSignal(pd.DataFrame)
    error_signal = pyqtSignal(str)
    
    def __init__(self, keywords, from_date, to_date, api_key):
        super().__init__()
        self.keywords = keywords
        self.from_date = from_date
        self.to_date = to_date
        self.api_key = api_key
        
    def run(self):
        try:
            all_dfs = []
            for keyword in self.keywords:
                df = fetch_news(keyword, self.from_date, self.to_date, api_key=self.api_key)
                if not df.empty:
                    df['keyword'] = keyword
                    all_dfs.append(df)
            
            if all_dfs:
                merged_df = pd.concat(all_dfs, ignore_index=True)
                if 'source' in merged_df.columns:
                    merged_df = merged_df.drop("source", axis=1)
                preprocessed_df = preprocess_news_data(merged_df)
                self.update_signal.emit(preprocessed_df)
            else:
                self.error_signal.emit("没有找到相关新闻")
        except Exception as e:
            self.error_signal.emit(f"获取新闻失败: {str(e)}")

class AnalyzeNewsThread(QThread):
    update_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, news_df, keywords, api_key):
        super().__init__()
        self.news_df = news_df
        self.keywords = keywords
        self.api_key = api_key
        
    def run(self):
        try:
            if os.environ.get("DEEPSEEK_API_KEY") != self.api_key:
                os.environ["DEEPSEEK_API_KEY"] = self.api_key

            llm = ChatDeepSeek(
                model="deepseek-reasoner",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )
            
            # 构建提示词
            all_keywords = ", ".join(self.keywords)
            prompt_news_summary = build_prompt_for_summary(self.news_df, all_keywords)
            
            self.update_signal.emit("正在分析新闻...\n")
            news_summary_response = llm.invoke(prompt_news_summary)
            self.update_signal.emit(news_summary_response.content)
            
        except Exception as e:
            self.error_signal.emit(f"分析新闻失败: {str(e)}")

class VerifyNewsThread(QThread):
    update_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, query, csv_path, api_key):
        super().__init__()
        self.query = query
        self.csv_path = csv_path
        self.api_key = api_key
        
    def run(self):
        try:
            if os.environ.get("DEEPSEEK_API_KEY") != self.api_key:
                os.environ["DEEPSEEK_API_KEY"] = self.api_key
                
            llm = ChatDeepSeek(
                model="deepseek-reasoner",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )
            
            self.update_signal.emit("正在加载数据...\n")
            
            # 加载新闻数据
            loader_news = CSVLoader(self.csv_path)
            documents_news = loader_news.load()
            
            # 文本分割
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=5)
            texts_news = text_splitter.split_documents(documents_news)
            
            # 向量嵌入
            self.update_signal.emit("正在构建向量数据库...\n")
            hg_embeddings = HuggingFaceEmbeddings()
            
            persist_directory = 'docs/chroma_rag/'
            
            # 检查 Chroma 数据库是否存在
            if os.path.exists(persist_directory):
                self.update_signal.emit("加载现有向量数据库...\n")
                economic_langchain_chroma = Chroma(
                    collection_name="economic_data",
                    embedding_function=hg_embeddings,
                    persist_directory=persist_directory
                )
            else:
                self.update_signal.emit("创建新的向量数据库...\n")
                economic_langchain_chroma = Chroma.from_documents(
                    documents=texts_news,
                    collection_name="economic_data",
                    embedding=hg_embeddings,
                    persist_directory=persist_directory
                )
            
            # 设置检索器
            retriever_eco = economic_langchain_chroma.as_retriever(search_kwargs={"k": 10})
            
            # 构建提示模板
            template = """你是一名金融信息专家。
                      请仅根据这些信息 {context} 并回答有关该公司的信息 {question}是否为真，并准确标注信息来源及时间"""
            
            PROMPT = PromptTemplate(input_variables=["context","question"], template=template)
            
            self.update_signal.emit("正在验证信息...\n")
            qa_with_sources = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff",
                chain_type_kwargs={"prompt": PROMPT}, 
                retriever=retriever_eco, 
                return_source_documents=True
            )
            
            llm_response = qa_with_sources({"query": self.query})
            self.update_signal.emit(llm_response['result'])
            
        except Exception as e:
            self.error_signal.emit(f"验证信息失败: {str(e)}")

# 主窗口
class FinancialGptApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.news_df = None
        self.csv_path = "preprocessed_news.csv"
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(APP_TITLE)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # 设置应用程序调色板
        app_palette = QPalette()
        app_palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND_COLOR))
        app_palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_COLOR))
        app_palette.setColor(QPalette.ColorRole.Base, QColor(BACKGROUND_COLOR))
        app_palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_COLOR))
        app_palette.setColor(QPalette.ColorRole.Button, QColor(BACKGROUND_COLOR))
        app_palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_COLOR))
        app_palette.setColor(QPalette.ColorRole.Highlight, THEME_COLOR)
        app_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(BACKGROUND_COLOR))
        self.setPalette(app_palette)
        
        # 应用样式表
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BACKGROUND_COLOR};
                border: 2px solid {THEME_COLOR_STR};
            }}
            QWidget {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
            }}
            QTabWidget::pane {{
                border: 1px solid #cccccc;
                background-color: {BACKGROUND_COLOR};
            }}
            QTabBar::tab {{
                background-color: #f0f0f0;
                color: {TEXT_COLOR};
                padding: 8px 20px;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {THEME_COLOR_STR};
                color: {BACKGROUND_COLOR};
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
            QLineEdit, QTextEdit, QComboBox, QSpinBox {{
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
        
        # 添加徽标
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
        
        # 创建选项卡部件
        self.tabs = QTabWidget()
        
        # 创建新闻获取和分析选项卡
        self.tab_news = QWidget()
        self.tab_verification = QWidget()
        
        self.tabs.addTab(self.tab_news, "新闻获取与分析")
        self.tabs.addTab(self.tab_verification, "金融信息验证")
        
        # 设置新闻获取和分析选项卡
        self.setup_news_tab()
        
        # 设置金融信息验证选项卡
        self.setup_verification_tab()
        
        # 添加选项卡到主布局
        main_layout.addWidget(self.tabs)
        
        # 添加状态栏
        self.statusBar().showMessage("就绪")
        self.statusBar().setStyleSheet(f"""
            background-color: {THEME_COLOR_STR};
            color: white;
            padding: 5px;
        """)
        
        # 检查缓存的新闻数据是否存在
        if os.path.exists(self.csv_path):
            try:
                self.news_df = pd.read_csv(self.csv_path)
                if 'publishedAt' in self.news_df.columns:
                    self.news_df['publishedAt'] = pd.to_datetime(self.news_df['publishedAt'])
                self.statusBar().showMessage(f"已加载缓存的新闻数据: {len(self.news_df)} 条记录")
            except Exception as e:
                self.statusBar().showMessage(f"加载缓存的新闻数据失败: {str(e)}")
    
    def setup_news_tab(self):
        # 创建新闻获取和分析选项卡的布局
        layout = QVBoxLayout(self.tab_news)
        
        # 参数设置组
        params_group = QGroupBox("新闻获取参数")
        params_layout = QGridLayout(params_group)
        
        # 关键词输入
        params_layout.addWidget(QLabel("关键词 (例如: Apple(AAPL), Tesla(TSLA)):"), 0, 0)
        self.keywords_input = QLineEdit("Alibaba(BABA)")
        params_layout.addWidget(self.keywords_input, 0, 1)
        
        # 天数选择
        params_layout.addWidget(QLabel("获取过去几天的新闻:"), 1, 0)
        self.days_spinbox = QSpinBox()
        self.days_spinbox.setRange(1, 30)
        self.days_spinbox.setValue(29)  # 默认值与原代码一致
        params_layout.addWidget(self.days_spinbox, 1, 1)
        
        # API密钥输入
        params_layout.addWidget(QLabel("News API 密钥:"), 2, 0)
        self.news_api_key_input = QLineEdit("61792ba8adbd40b7bc86b5563eb41f87")  # 使用原代码中的默认值
        params_layout.addWidget(self.news_api_key_input, 2, 1)
        
        params_layout.addWidget(QLabel("DeepSeek API 密钥:"), 3, 0)
        self.deepseek_api_key_input = QLineEdit("sk-7c87ef2add054e439095db9b18c921e9")  # 使用原代码中的默认值
        params_layout.addWidget(self.deepseek_api_key_input, 3, 1)
        
        # 按钮
        buttons_layout = QHBoxLayout()
        
        self.fetch_button = QPushButton("获取新闻")
        self.fetch_button.clicked.connect(self.fetch_news)
        buttons_layout.addWidget(self.fetch_button)
        
        self.analyze_button = QPushButton("分析新闻")
        self.analyze_button.clicked.connect(self.analyze_news)
        self.analyze_button.setEnabled(False)  # 初始禁用，直到获取了新闻
        buttons_layout.addWidget(self.analyze_button)
        
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)  # 初始禁用，直到有结果
        buttons_layout.addWidget(self.save_button)
        
        # 添加参数组和按钮到布局
        layout.addWidget(params_group)
        layout.addLayout(buttons_layout)
        
        # 添加输出区域
        output_group = QGroupBox("分析结果")
        output_layout = QVBoxLayout(output_group)
        
        self.news_output = QTextBrowser()
        self.news_output.setOpenExternalLinks(True)
        output_layout.addWidget(self.news_output)
        
        layout.addWidget(output_group)
        
        # 如果有缓存的新闻数据，启用分析按钮
        if os.path.exists(self.csv_path):
            self.analyze_button.setEnabled(True)
    
    def setup_verification_tab(self):
        # 创建金融信息验证选项卡的布局
        layout = QVBoxLayout(self.tab_verification)
        
        # 参数设置组
        params_group = QGroupBox("信息验证")
        params_layout = QVBoxLayout(params_group)
        
        # 信息输入
        params_layout.addWidget(QLabel("输入需要验证的金融信息:"))
        self.verification_input = QTextEdit()
        self.verification_input.setPlaceholderText("例如: 阿里巴巴集团控股有限公司已承诺投资超过3800亿元人民币（约530亿美元）用于人工智能基础设施，如数据中心等...")
        self.verification_input.setText("阿里巴巴集团控股有限公司已承诺投资超过3800亿元人民币（约530亿美元）用于人工智能基础设施，如数据中心等...")  # 使用原代码中的示例
        params_layout.addWidget(self.verification_input)
        
        # API密钥输入
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("DeepSeek API 密钥:"))
        self.verify_api_key_input = QLineEdit("sk-7c87ef2add054e439095db9b18c921e9")  # 使用原代码中的默认值
        key_layout.addWidget(self.verify_api_key_input)
        params_layout.addLayout(key_layout)
        
        # 验证按钮
        self.verify_button = QPushButton("验证信息")
        self.verify_button.clicked.connect(self.verify_news)
        params_layout.addWidget(self.verify_button)
        
        # 添加参数组到布局
        layout.addWidget(params_group)
        
        # 添加输出区域
        output_group = QGroupBox("验证结果")
        output_layout = QVBoxLayout(output_group)
        
        self.verification_output = QTextBrowser()
        self.verification_output.setOpenExternalLinks(True)
        output_layout.addWidget(self.verification_output)
        
        layout.addWidget(output_group)
    
    def fetch_news(self):
        # 获取输入参数
        keywords_text = self.keywords_input.text().strip()
        if not keywords_text:
            QMessageBox.warning(self, "错误", "请输入至少一个关键词")
            return
        
        keywords = [k.strip() for k in keywords_text.split(',')]
        days = self.days_spinbox.value()
        news_api_key = self.news_api_key_input.text().strip()
        
        if not news_api_key:
            QMessageBox.warning(self, "错误", "请输入News API密钥")
            return
        
        # 设置日期范围
        current_time = datetime.now()
        from_date = current_time - timedelta(days=days)
        
        # 禁用按钮并显示状态
        self.fetch_button.setEnabled(False)
        self.statusBar().showMessage("正在获取新闻...")
        self.news_output.clear()
        self.news_output.append("正在获取新闻，请稍候...\n")
        
        # 创建并启动线程
        self.fetch_thread = FetchNewsThread(keywords, from_date, current_time, news_api_key)
        self.fetch_thread.update_signal.connect(self.update_news_data)
        self.fetch_thread.error_signal.connect(self.show_error)
        self.fetch_thread.finished.connect(lambda: self.fetch_button.setEnabled(True))
        self.fetch_thread.start()
    
    def update_news_data(self, df):
        self.news_df = df
        
        # 保存到CSV
        try:
            self.news_df.to_csv(self.csv_path, index=False)
            self.statusBar().showMessage(f"已获取 {len(df)} 条新闻并保存到 {self.csv_path}")
        except Exception as e:
            self.statusBar().showMessage(f"保存新闻数据失败: {str(e)}")
        
        # 显示新闻摘要
        self.news_output.clear()
        self.news_output.append(f"已获取 {len(df)} 条新闻\n")
        
        if not df.empty:
            sample_size = min(5, len(df))
            self.news_output.append(f"\n新闻样本 (显示前 {sample_size} 条):\n")
            
            for idx, row in df.head(sample_size).iterrows():
                self.news_output.append(f"标题: {row['title']}")
                self.news_output.append(f"发布时间: {row['publishedAt']}")
                self.news_output.append(f"作者: {row['author']}")
                self.news_output.append(f"内容摘要: {row['description']}")
                self.news_output.append("----------\n")
        
        # 启用分析按钮
        self.analyze_button.setEnabled(True)
    
    def analyze_news(self):
        if self.news_df is None or self.news_df.empty:
            QMessageBox.warning(self, "错误", "没有可分析的新闻数据")
            return
        
        # 获取API密钥
        api_key = self.deepseek_api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "错误", "请输入DeepSeek API密钥")
            return
        
        # 获取关键词
        keywords_text = self.keywords_input.text().strip()
        keywords = [k.strip() for k in keywords_text.split(',')]
        
        # 禁用按钮并显示状态
        self.analyze_button.setEnabled(False)
        self.statusBar().showMessage("正在分析新闻...")
        self.news_output.clear()
        self.news_output.append("正在分析新闻，这可能需要几分钟时间...\n")
        
        # 创建并启动线程
        self.analyze_thread = AnalyzeNewsThread(self.news_df, keywords, api_key)
        self.analyze_thread.update_signal.connect(self.update_analysis)
        self.analyze_thread.error_signal.connect(self.show_error)
        self.analyze_thread.finished.connect(lambda: self.analyze_button.setEnabled(True))
        self.analyze_thread.start()
    
    def update_analysis(self, result):
        # 将Markdown内容转换为HTML
        if result.startswith("正在分析新闻..."):
            # 初始消息保持为纯文本
            self.news_output.clear()
            self.news_output.setPlainText(result)
        else:
            # 分析结果转换为HTML显示
            html_content = markdown.markdown(result)
            self.news_output.setHtml(html_content)
        
        # 滚动到底部
        self.news_output.verticalScrollBar().setValue(self.news_output.verticalScrollBar().maximum())
        
        # 启用保存按钮
        self.save_button.setEnabled(True)
        
        # 更新状态
        self.statusBar().showMessage("新闻分析完成")
    
    def verify_news(self):
        # 获取输入信息
        query = self.verification_input.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "错误", "请输入需要验证的信息")
            return
        
        # 获取API密钥
        api_key = self.verify_api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "错误", "请输入DeepSeek API密钥")
            return
        
        # 检查新闻数据是否存在
        if not os.path.exists(self.csv_path):
            QMessageBox.warning(self, "错误", "没有找到新闻数据文件，请先在'新闻获取与分析'选项卡中获取新闻")
            return
        
        # 禁用按钮并显示状态
        self.verify_button.setEnabled(False)
        self.statusBar().showMessage("正在验证信息...")
        self.verification_output.clear()
        self.verification_output.append("正在验证信息，这可能需要几分钟时间...\n")
        
        # 创建并启动线程
        self.verify_thread = VerifyNewsThread(query, self.csv_path, api_key)
        self.verify_thread.update_signal.connect(self.update_verification)
        self.verify_thread.error_signal.connect(self.show_error)
        self.verify_thread.finished.connect(lambda: self.verify_button.setEnabled(True))
        self.verify_thread.start()
    
    def update_verification(self, result):
        # 将Markdown内容转换为HTML
        if result.startswith("正在") and result.endswith("...\n"):
            # 初始消息保持为纯文本
            self.verification_output.setPlainText(result)
        else:
            # 验证结果转换为HTML显示
            html_content = markdown.markdown(result)
            self.verification_output.setHtml(html_content)
        
        # 滚动到底部
        self.verification_output.verticalScrollBar().setValue(self.verification_output.verticalScrollBar().maximum())
        
        # 更新状态
        self.statusBar().showMessage("信息验证完成")
    
    def save_results(self):
        # 保存分析结果到文件
        content = self.news_output.toPlainText()
        if not content:
            QMessageBox.warning(self, "错误", "没有可保存的分析结果")
            return
        
        # 打开文件对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存分析结果", 
            "news_analysis_result.txt", 
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                self.statusBar().showMessage(f"分析结果已保存到 {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存文件失败: {str(e)}")
    
    def show_error(self, error_message):
        # 显示错误信息
        QMessageBox.critical(self, "错误", error_message)
        self.statusBar().showMessage(f"发生错误: {error_message}")

# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 应用程序样式
    app.setStyle("Fusion")
    
    # 显示应用程序
    main_window = FinancialGptApp()
    main_window.show()
    
    sys.exit(app.exec())
