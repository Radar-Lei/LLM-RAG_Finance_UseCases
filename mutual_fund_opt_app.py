import sys
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from datetime import datetime
import tempfile
import urllib.request

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QTextEdit, QTextBrowser,
                           QLineEdit, QComboBox, QSpinBox, QTabWidget, 
                           QGridLayout, QGroupBox, QCheckBox, QFileDialog,
                           QMessageBox, QSplashScreen, QScrollArea, QDateEdit,
                           QSizePolicy)
from PyQt6.QtGui import QPixmap, QIcon, QColor, QPalette, QFont, QTextCursor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QDate

# 导入原始脚本的相关库
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from IPython.display import display, Markdown

# 定义常量
APP_TITLE = "基金组合优化AI助手"
THEME_COLOR = QColor(172, 52, 32, 230)  # 红色主题 #AC3420 with 90% opacity
THEME_COLOR_STR = "#AC3420"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
LOGO_PATH = "figs/四川大学商学院.png"
AUTHOR = "作者：雷达"
ACCENT_COLOR = "#D9534F"  # 辅助颜色，比主题色更柔和
BACKGROUND_COLOR = "#FFFFFF"  # 背景色
TEXT_COLOR = "#333333"  # 文字颜色

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

# 下载中文字体如果需要
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

# 创建图表的画布部件
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, 
                                  QSizePolicy.Policy.Expanding, 
                                  QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        # 设置字体
        self.chinese_font = get_chinese_font()
        if self.chinese_font.get_name() == 'DejaVu Sans':
            font_path = download_chinese_font()
            self.chinese_font = FontProperties(fname=font_path)

# 后台线程用于数据生成
class DataGenerationThread(QThread):
    update_signal = pyqtSignal(pd.DataFrame)
    progress_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, num_funds, start_date, csv_path):
        super().__init__()
        self.num_funds = num_funds
        self.start_date = start_date
        self.csv_path = csv_path
        
    def run(self):
        try:
            self.progress_signal.emit("正在生成基金数据...")
            
            # 生成数据
            data = []
            
            for fund_id in range(1, self.num_funds + 1):
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
                    '日期': self.start_date,
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
            df.to_csv(self.csv_path, index=False)
            self.progress_signal.emit(f"基金数据已生成并保存到 {self.csv_path}")
            self.update_signal.emit(df)
        
        except Exception as e:
            self.error_signal.emit(f"生成数据失败: {str(e)}")

# 后台线程用于优化基金配置
class OptimizationThread(QThread):
    update_signal = pyqtSignal(pd.DataFrame, str)
    progress_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, df, persist_directory, collection_name, api_key):
        super().__init__()
        self.df = df
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.api_key = api_key
        
    def run(self):
        try:
            self.progress_signal.emit("正在准备数据...")
            
            # 为LangChain准备文档
            documents = []
            for _, row in self.df.iterrows():
                content = f"基金编号: {row['基金编号']}, 日期: {row['日期']}, 净值: {row['平均净值']}, " \
                        f"收益率: {row['平均收益率_%']}, 风险等级: {row['风险等级']}, " \
                        f"科技占比: {row['科技占比_%']}, " \
                        f"医疗占比: {row['医疗占比_%']}, " \
                        f"金融占比: {row['金融占比_%']}, " \
                        f"能源占比: {row['能源占比_%']}, " \
                        f"利率: {row['平均利率_%']}, 通胀率: {row['平均通胀率_%']}"
                
                documents.append(Document(page_content=content))
            
            self.progress_signal.emit("初始化嵌入模型...")
            
            # 指定模型名称以避免警告
            hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # 检查ChromaDB集合是否已存在
            try:
                # 尝试加载现有集合
                self.progress_signal.emit("检查现有向量数据库...")
                existing_db = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=hg_embeddings,
                    persist_directory=self.persist_directory
                )
                
                # 检查集合是否有数据
                if existing_db._collection.count() > 0:
                    self.progress_signal.emit(f"使用现有向量数据库集合: '{self.collection_name}'")
                    langchain_chroma = existing_db
                else:
                    self.progress_signal.emit(f"创建新的向量数据库集合: '{self.collection_name}'")
                    langchain_chroma = Chroma.from_documents(
                        documents=documents,
                        collection_name=self.collection_name,
                        embedding=hg_embeddings,
                        persist_directory=self.persist_directory
                    )
            except Exception as e:
                self.progress_signal.emit(f"创建新的向量数据库集合: '{self.collection_name}'")
                # 初始化ChromaDB
                langchain_chroma = Chroma.from_documents(
                    documents=documents,
                    collection_name=self.collection_name,
                    embedding=hg_embeddings,
                    persist_directory=self.persist_directory
                )
            
            # 设置API密钥
            if not os.getenv("DEEPSEEK_API_KEY"):
                os.environ["DEEPSEEK_API_KEY"] = self.api_key
            
            self.progress_signal.emit("初始化LLM模型...")
            
            # 初始化LLM
            llm = ChatDeepSeek(
                model="deepseek-reasoner",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            
            # 定义提示模板
            template = """
            基于以下共同基金数据，分析并仅提供每只基金各百块的百分比优化建议。
            共同基金信息：{question}
            上下文：{context}
            回答：
            """
            PROMPT = PromptTemplate(input_variables=["context", "query"], template=template)
            
            # 设置检索器
            retriever = langchain_chroma.as_retriever(search_kwargs={"k": 10})
            
            # 设置QA链
            qa_chain = RetrievalQA.from_chain_type(
                llm, retriever=retriever, chain_type_kwargs={"prompt": PROMPT}
            )
            
            self.progress_signal.emit("正在发送优化请求...")
            
            # 获取优化建议
            def get_optimized_recommendations(query):
                # 获取相关文档
                raw_docs = retriever.get_relevant_documents(query)
                
                # 移除重复文档
                seen = set()
                unique_docs = []
                for doc in raw_docs:
                    if doc.page_content not in seen:
                        unique_docs.append(doc)
                        seen.add(doc.page_content)
                
                # 准备上下文
                context = " ".join([doc.page_content for doc in unique_docs])
                
                # 使用QA链获取响应
                result = qa_chain({"context": context, "query": query})
                return result
            
            # 示例查询
            query = "分析并提供每个基金板块的优化推荐百分比。并按照r'基金编号: (\\d+)\\s+优化建议：科技 (\\d+)%，医疗 (\\d+)%，金融 (\\d+)%，能源 (\\d+)%'的格式返回结果。"
            
            self.progress_signal.emit("正在等待优化结果...")
            response = get_optimized_recommendations(query)
            
            self.progress_signal.emit("解析优化结果...")
            
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
            merged_df = pd.merge(self.df, optimization_result, on='基金编号', how='inner')
            merged_df.to_csv('merged_fund_data.csv', index=False)
            
            self.progress_signal.emit("优化完成！")
            self.update_signal.emit(merged_df, response['result'])
            
        except Exception as e:
            self.error_signal.emit(f"优化失败: {str(e)}")

# 主窗口
class MutualFundOptApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 初始化变量
        self.df = None
        self.merged_df = None
        self.csv_path = 'mutual_fund_data.csv'
        self.persist_directory = './chroma_db/'
        self.collection_name = "mutual_fund_optimization"
        
        # 获取中文字体
        self.chinese_font = get_chinese_font()
        if self.chinese_font.get_name() == 'DejaVu Sans':
            self.font_path = download_chinese_font()
            self.chinese_font = FontProperties(fname=self.font_path)
        
        # 设置全局字体参数
        plt.rcParams['font.family'] = self.chinese_font.get_name()
        plt.rcParams['axes.unicode_minus'] = False  # 修正负号显示
        
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
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDateEdit {{
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
        if os.path.exists(LOGO_PATH):
            logo_pixmap = QPixmap(LOGO_PATH)
            logo_size = QSize(200, 60)
            scaled_logo = logo_pixmap.scaled(logo_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
            logo_label.setFixedSize(logo_size)
        else:
            logo_label.setText("找不到徽标")
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
        
        # 创建选项卡
        self.tab_data = QWidget()
        self.tab_optimization = QWidget()
        self.tab_visualization = QWidget()
        
        self.tabs.addTab(self.tab_data, "数据管理")
        self.tabs.addTab(self.tab_optimization, "优化配置")
        self.tabs.addTab(self.tab_visualization, "结果可视化")
        
        # 设置各选项卡
        self.setup_data_tab()
        self.setup_optimization_tab()
        self.setup_visualization_tab()
        
        # 添加选项卡到主布局
        main_layout.addWidget(self.tabs)
        
        # 添加状态栏
        self.statusBar().showMessage("就绪")
        self.statusBar().setStyleSheet(f"""
            background-color: {THEME_COLOR_STR};
            color: white;
            padding: 5px;
        """)
        
        # 检查缓存的数据是否存在
        if os.path.exists(self.csv_path):
            try:
                self.df = pd.read_csv(self.csv_path)
                self.update_data_summary()
                self.statusBar().showMessage(f"已加载基金数据: {len(self.df)} 条记录")
                
                # 检查是否存在合并数据
                if os.path.exists('merged_fund_data.csv'):
                    self.merged_df = pd.read_csv('merged_fund_data.csv')
                    self.tabs.setCurrentIndex(2)  # 切换到可视化选项卡
                    self.update_visualization()
            except Exception as e:
                self.statusBar().showMessage(f"加载数据失败: {str(e)}")
    
    def setup_data_tab(self):
        # 创建数据管理选项卡的布局
        layout = QVBoxLayout(self.tab_data)
        
        # 参数设置组
        params_group = QGroupBox("数据生成参数")
        params_layout = QGridLayout(params_group)
        
        # 基金数量
        params_layout.addWidget(QLabel("基金数量:"), 0, 0)
        self.num_funds_spinbox = QSpinBox()
        self.num_funds_spinbox.setRange(1, 100)
        self.num_funds_spinbox.setValue(10)  # 默认值与原代码一致
        params_layout.addWidget(self.num_funds_spinbox, 0, 1)
        
        # 开始日期
        params_layout.addWidget(QLabel("开始日期:"), 1, 0)
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setDate(QDate(2025, 1, 1))  # 默认值与原代码一致
        self.start_date_edit.setCalendarPopup(True)
        params_layout.addWidget(self.start_date_edit, 1, 1)
        
        # 数据文件路径
        params_layout.addWidget(QLabel("数据文件路径:"), 2, 0)
        self.csv_path_input = QLineEdit(self.csv_path)
        params_layout.addWidget(self.csv_path_input, 2, 1)
        
        # 数据操作按钮
        buttons_layout = QHBoxLayout()
        
        self.generate_button = QPushButton("生成新数据")
        self.generate_button.clicked.connect(self.generate_data)
        buttons_layout.addWidget(self.generate_button)
        
        self.load_button = QPushButton("加载现有数据")
        self.load_button.clicked.connect(self.load_data)
        buttons_layout.addWidget(self.load_button)
        
        # 添加参数组和按钮到布局
        layout.addWidget(params_group)
        layout.addLayout(buttons_layout)
        
        # 添加数据预览区域
        preview_group = QGroupBox("数据预览")
        preview_layout = QVBoxLayout(preview_group)
        
        self.data_summary = QTextBrowser()
        preview_layout.addWidget(self.data_summary)
        
        layout.addWidget(preview_group)
    
    def setup_optimization_tab(self):
        # 创建优化配置选项卡的布局
        layout = QVBoxLayout(self.tab_optimization)
        
        # 参数设置组
        params_group = QGroupBox("优化参数")
        params_layout = QGridLayout(params_group)
        
        # 向量存储目录
        params_layout.addWidget(QLabel("向量存储目录:"), 0, 0)
        self.persist_dir_input = QLineEdit(self.persist_directory)
        params_layout.addWidget(self.persist_dir_input, 0, 1)
        
        # 集合名称
        params_layout.addWidget(QLabel("集合名称:"), 1, 0)
        self.collection_name_input = QLineEdit(self.collection_name)
        params_layout.addWidget(self.collection_name_input, 1, 1)
        
        # DeepSeek API密钥
        params_layout.addWidget(QLabel("DeepSeek API密钥:"), 2, 0)
        self.api_key_input = QLineEdit("sk-7c87ef2add054e439095db9b18c921e9")  # 默认使用原始代码中的密钥
        params_layout.addWidget(self.api_key_input, 2, 1)
        
        # 优化操作按钮
        self.optimize_button = QPushButton("开始优化")
        self.optimize_button.clicked.connect(self.start_optimization)
        
        # 添加组件到布局
        layout.addWidget(params_group)
        layout.addWidget(self.optimize_button)
        
        # 添加优化结果区域
        results_group = QGroupBox("优化结果")
        results_layout = QVBoxLayout(results_group)
        
        self.optimization_output = QTextBrowser()
        results_layout.addWidget(self.optimization_output)
        
        layout.addWidget(results_group)
    
    def setup_visualization_tab(self):
        # 创建结果可视化选项卡的布局
        layout = QVBoxLayout(self.tab_visualization)
        
        # 图表容器
        chart_group = QGroupBox("基金配置对比图")
        chart_layout = QVBoxLayout(chart_group)
        
        # 创建画布容器
        self.plot_canvas_container = QScrollArea()
        self.plot_canvas_container.setWidgetResizable(True)
        
        # 创建一个容器Widget来放置画布
        canvas_container = QWidget()
        self.plot_canvas_layout = QVBoxLayout(canvas_container)
        
        # 初始化创建画布，而不是等到update_visualization方法
        self.plot_canvas = MatplotlibCanvas(width=10, height=8, dpi=100)
        self.plot_canvas_layout.addWidget(self.plot_canvas)
        
        self.plot_canvas_container.setWidget(canvas_container)
        chart_layout.addWidget(self.plot_canvas_container)
        
        # 添加按钮
        buttons_layout = QHBoxLayout()
        
        self.save_chart_button = QPushButton("保存图表")
        self.save_chart_button.clicked.connect(self.save_chart)
        buttons_layout.addWidget(self.save_chart_button)
        
        self.save_data_button = QPushButton("导出优化数据")
        self.save_data_button.clicked.connect(self.export_data)
        buttons_layout.addWidget(self.save_data_button)
        
        chart_layout.addLayout(buttons_layout)
        
        # 添加到主布局
        layout.addWidget(chart_group)
    
    def generate_data(self):
        # 获取参数
        num_funds = self.num_funds_spinbox.value()
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        csv_path = self.csv_path_input.text().strip()
        
        self.statusBar().showMessage("正在生成数据...")
        self.generate_button.setEnabled(False)
        
        # 创建并启动线程
        self.data_thread = DataGenerationThread(num_funds, start_date, csv_path)
        self.data_thread.update_signal.connect(self.update_data)
        self.data_thread.progress_signal.connect(self.update_progress)
        self.data_thread.error_signal.connect(self.show_error)
        self.data_thread.finished.connect(lambda: self.generate_button.setEnabled(True))
        self.data_thread.start()
    
    def load_data(self):
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "加载基金数据", 
            "", 
            "CSV文件 (*.csv);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.csv_path = file_path
                self.csv_path_input.setText(file_path)
                self.update_data_summary()
                self.statusBar().showMessage(f"已加载基金数据: {len(self.df)} 条记录")
            except Exception as e:
                self.show_error(f"加载数据失败: {str(e)}")
    
    def update_data(self, df):
        self.df = df
        self.update_data_summary()
        
        # 启用优化按钮
        if not self.df.empty:
            self.tabs.setCurrentIndex(1)  # 切换到优化选项卡
            self.statusBar().showMessage(f"已生成 {len(df)} 条基金数据")
    
    def update_data_summary(self):
        if self.df is None or self.df.empty:
            self.data_summary.setText("没有可用的基金数据")
            return
        
        # 准备摘要信息
        summary = f"基金数据摘要 (共 {len(self.df)} 条记录):\n\n"
        
        # 基本统计信息
        summary += "== 净值统计 ==\n"
        summary += f"平均净值范围: {self.df['平均净值'].min():.2f} - {self.df['平均净值'].max():.2f}\n"
        summary += f"平均净值均值: {self.df['平均净值'].mean():.2f}\n\n"
        
        summary += "== 收益率统计 ==\n"
        summary += f"收益率范围: {self.df['平均收益率_%'].min():.2f}% - {self.df['平均收益率_%'].max():.2f}%\n"
        summary += f"收益率均值: {self.df['平均收益率_%'].mean():.2f}%\n\n"
        
        summary += "== 风险等级分布 ==\n"
        risk_counts = self.df['风险等级'].value_counts()
        for risk, count in risk_counts.items():
            summary += f"{risk}: {count} 只基金\n"
        
        summary += "\n== 行业占比平均值 ==\n"
        summary += f"科技: {self.df['科技占比_%'].mean():.2f}%\n"
        summary += f"医疗: {self.df['医疗占比_%'].mean():.2f}%\n"
        summary += f"金融: {self.df['金融占比_%'].mean():.2f}%\n"
        summary += f"能源: {self.df['能源占比_%'].mean():.2f}%\n"
        
        # 显示前5条记录
        if len(self.df) > 0:
            summary += "\n== 前 5 条记录预览 ==\n"
            preview_df = self.df.head().copy()
            
            # 格式化显示
            for idx, row in preview_df.iterrows():
                summary += f"基金编号: {row['基金编号']}, 净值: {row['平均净值']:.2f}, "
                summary += f"收益率: {row['平均收益率_%']:.2f}%, 风险等级: {row['风险等级']}\n"
                summary += f"行业占比: 科技 {row['科技占比_%']:.2f}%, 医疗 {row['医疗占比_%']:.2f}%, "
                summary += f"金融 {row['金融占比_%']:.2f}%, 能源 {row['能源占比_%']:.2f}%\n"
                summary += f"利率: {row['平均利率_%']:.2f}%, 通胀率: {row['平均通胀率_%']:.2f}%\n\n"
        
        self.data_summary.setText(summary)
    
    def start_optimization(self):
        # 检查是否有基金数据
        if self.df is None or self.df.empty:
            self.show_error("没有可用的基金数据，请先生成或加载数据")
            return
        
        # 获取参数
        persist_directory = self.persist_dir_input.text().strip()
        collection_name = self.collection_name_input.text().strip()
        api_key = self.api_key_input.text().strip()
        
        if not api_key:
            self.show_error("请输入DeepSeek API密钥")
            return
            
        # 禁用按钮并显示状态
        self.optimize_button.setEnabled(False)
        self.statusBar().showMessage("正在优化...")
        self.optimization_output.clear()
        
        # 创建并启动线程
        self.opt_thread = OptimizationThread(self.df, persist_directory, collection_name, api_key)
        self.opt_thread.update_signal.connect(self.update_optimization_result)
        self.opt_thread.progress_signal.connect(self.update_progress)
        self.opt_thread.error_signal.connect(self.show_error)
        self.opt_thread.finished.connect(lambda: self.optimize_button.setEnabled(True))
        self.opt_thread.start()
    
    def update_optimization_result(self, merged_df, result_text):
        # 保存优化结果
        self.merged_df = merged_df
        
        # 显示结果
        self.optimization_output.setPlainText(result_text)
        
        # 更新可视化
        self.update_visualization()
        
        # 切换到可视化选项卡
        self.tabs.setCurrentIndex(2)
        
        # 更新状态
        self.statusBar().showMessage("优化完成")
    
    def update_visualization(self):
        if self.merged_df is None or self.merged_df.empty:
            return
        
        # 清除并重置画布
        self.plot_canvas.fig.clear()
        
        # 创建比较图表
        self.create_comparison_charts()
        
        # 刷新画布
        self.plot_canvas.draw()
    
    def create_comparison_charts(self):
        if self.merged_df is None or self.merged_df.empty:
            return
        
        # 设置图表样式
        sns.set(style="whitegrid")
        
        # 处理的基金编号列表
        fund_ids = sorted(self.merged_df['基金编号'].unique())
        num_funds = len(fund_ids)
        
        # 创建横向布局的网格
        gs = GridSpec(2, num_funds, figure=self.plot_canvas.fig)
        
        # 定义共享的颜色和标签
        labels = ['科技', '医疗', '金融', '能源']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # 第一行：所有基金的原始分配
        for col, fund_id in enumerate(fund_ids):
            fund_data = self.merged_df[self.merged_df['基金编号'] == fund_id]
            
            # 原始分配饼图
            ax1 = self.plot_canvas.fig.add_subplot(gs[0, col])
            original_data = [
                fund_data['科技占比_%'].values[0],
                fund_data['医疗占比_%'].values[0],
                fund_data['金融占比_%'].values[0],
                fund_data['能源占比_%'].values[0]
            ]
            
            ax1.pie(original_data, autopct='%1.1f%%', startangle=90, colors=colors)
            ax1.axis('equal')
            ax1.set_title(f'基金 {fund_id} - 原始分配', fontproperties=self.plot_canvas.chinese_font, fontsize=12)
        
        # 第二行：所有基金的优化分配
        for col, fund_id in enumerate(fund_ids):
            fund_data = self.merged_df[self.merged_df['基金编号'] == fund_id]
            
            # 优化分配饼图
            ax2 = self.plot_canvas.fig.add_subplot(gs[1, col])
            optimized_data = [
                fund_data['科技占比_优化_%'].values[0],
                fund_data['医疗占比_优化_%'].values[0],
                fund_data['金融占比_优化_%'].values[0],
                fund_data['能源占比_优化_%'].values[0]
            ]
            
            ax2.pie(optimized_data, autopct='%1.1f%%', startangle=90, colors=colors)
            ax2.axis('equal')
            ax2.set_title(f'基金 {fund_id} - 优化分配', fontproperties=self.plot_canvas.chinese_font, fontsize=12)
        
        # 添加总标题
        self.plot_canvas.fig.suptitle('基金板块分配 - 原始分配与优化分配对比', 
                             fontproperties=self.plot_canvas.chinese_font, fontsize=16)
        
        # 创建图例的补丁
        patches = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(labels))]
        
        # 在图表右侧添加一个共享的图例
        self.plot_canvas.fig.legend(patches, labels, loc='center right', 
                          bbox_to_anchor=(0.98, 0.5), prop=self.plot_canvas.chinese_font, fontsize=12)
        
        # 调整布局
        self.plot_canvas.fig.tight_layout()
        # 为图例和标题腾出空间
        plt.subplots_adjust(top=0.9, right=0.92)
        
        # 刷新画布
        self.plot_canvas.draw()
    
    def save_chart(self):
        if self.plot_canvas is None:
            self.show_error("没有可保存的图表")
            return
        
        # 打开文件对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存图表", 
            "fund_allocation_comparison.png", 
            "PNG图像 (*.png);;JPG图像 (*.jpg);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                self.plot_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.statusBar().showMessage(f"图表已保存到 {file_path}")
            except Exception as e:
                self.show_error(f"保存图表失败: {str(e)}")
    
    def export_data(self):
        if self.merged_df is None or self.merged_df.empty:
            self.show_error("没有可导出的优化数据")
            return
        
        # 打开文件对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "导出优化数据", 
            "optimized_fund_data.csv", 
            "CSV文件 (*.csv);;Excel文件 (*.xlsx);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    self.merged_df.to_excel(file_path, index=False)
                else:
                    self.merged_df.to_csv(file_path, index=False)
                self.statusBar().showMessage(f"数据已导出到 {file_path}")
            except Exception as e:
                self.show_error(f"导出数据失败: {str(e)}")
    
    def update_progress(self, message):
        if self.tabs.currentIndex() == 1:  # 优化选项卡
            self.optimization_output.append(message)
            # 滚动到底部
            cursor = self.optimization_output.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.optimization_output.setTextCursor(cursor)
        self.statusBar().showMessage(message)
    
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
    main_window = MutualFundOptApp()
    main_window.show()
    
    sys.exit(app.exec())
