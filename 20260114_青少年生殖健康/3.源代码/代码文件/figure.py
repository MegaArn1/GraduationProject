import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from matplotlib import rcParams
from matplotlib import font_manager

# 设置全局字体，支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 'SimHei' for black, 'Microsoft YaHei' for elegant black
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

myfont = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")  # 以微软雅黑为例
# ==============================
# 1. 读取 Excel 数据
# ==============================
# 假设表格文件名为 "bilibili_data.xlsx"
# 请根据实际文件路径修改
df = pd.read_excel("ex_ernie-speed-video_crb_filtered_classify_1 to 5113_v1.3.xlsx")
print(df.columns.tolist())
# ==============================
# 2. Search_item 统计并生成词云
# ==============================
search_item_counts = df['Search_item'].value_counts().to_dict()

font_path = "C:/Windows/Fonts/msyh.ttc"  # 确保字体路径正确

wc_search = WordCloud( 
    font_path=font_path,
    width=800,
    height=400,
    background_color="white",
    colormap="viridis"
).generate_from_frequencies(search_item_counts)

plt.figure(figsize=(10,6))
plt.imshow(wc_search, interpolation="bilinear")
plt.axis("off")
plt.title("Search_item 关键词词云", fontsize=14)
plt.tight_layout()
plt.savefig("search_item_wordcloud.png", dpi=300)
plt.show()

# ==============================
# 3. Tag 统计并画柱状图
# ==============================
tag_counts_df = df['Tag'].value_counts().reset_index()
tag_counts_df.columns = ['Tag', 'Count']

plt.figure(figsize=(10,6))
sns.barplot(
    x='Tag', 
    y='Count',
    data=tag_counts_df,
    palette="Set2"
)
# plt.title("Tag 分类类别柱状图")
plt.xlabel("分类")
plt.ylabel("数量")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("tag_bar.png", dpi=300)
plt.show()
