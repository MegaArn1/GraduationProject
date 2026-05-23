'''
03/05
本程序用于：弹幕表、评论表 数据库 按关键词查找 --> 生成查找结果表格（新增两列："级别","关键字"。）
青少年生殖健康新进展

'''

import pymysql
import pandas as pd
from sqlalchemy import create_engine


# **数据库连接信息**
DB_CONFIG = {
    "host": "172.16.135.10",
    "port": 3306,
    "user": "zzx",
    "password": "20040505zzx",
    "database": "vd_comment",
    "charset": "utf8mb4"
}

# **创建 SQLAlchemy 连接**
engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset=utf8mb4")

# **关键词列表**
KEYWORDS = [
    "生育", "想生孩子", "想生娃", "想生育", "想生男孩", "想生女孩",
    "要孩子", "要娃", "要生孩子", "要生娃", "要生男孩", "要生女孩",
    "打算生孩子", "打算生娃", "想结婚", "要结婚", "打算结婚", "计划结婚",
    "想成家", "要成家", "打算成家", "领结婚证", "丁克", "绝后", "躺平",
    "自洽", "养老", "自由", "双职工", "断后", "彩礼", "恐婚", "不婚",
    "孤寡", "恋爱脑", "催婚", "民政局", "相亲", "分娩", "妊娠纹",
    "漏尿", "吞金兽", "带娃", "产后", "育儿", "宫缩"
]

# **查询数据**
def fetch_data(query):
    """使用 SQLAlchemy 连接数据库并执行 SQL 查询"""
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

# **匹配关键字**
def match_keywords(text):
    """匹配文本中的关键字，并返回所有匹配的关键字（以逗号分隔）"""
    if pd.isnull(text):  # 处理空值
        return None
    
    matched_keywords = []
    for keyword in KEYWORDS:
        if keyword in text:
            matched_keywords.append(keyword)
    
    if matched_keywords:
        return "、".join(matched_keywords)  # 用中文顿号连接多个关键字
    return None  # 没匹配到

# **查询 `video_bulletchats_v2`**
query_bulletchats = """
SELECT id, Search_item, URL, Video_ID, Title, Bullets_Count, Bullet_Videotime, 
       Bullet_Content, Bullet_Posttime, Tag 
FROM video_bulletchats_v2;
"""
df_bulletchats = fetch_data(query_bulletchats)

# **匹配关键字**
df_bulletchats["关键字"] = df_bulletchats["Bullet_Content"].apply(match_keywords)

# **过滤匹配到关键字的数据**
df_bulletchats = df_bulletchats.dropna(subset=["关键字"])

# **查询 `video_comments_all_v2`**
query_comments_all = """
SELECT id, Video_ID, Title, Search_item, Tag, Comment_ID, Commentator_Name, 
       Commentator_Level, Comment_Content, Comment_Time, Comment_Likes, Comment_Length, 
       Sub_C_Reply_ID, Sub_C_Replier_Name, Sub_C_Replier_Level, Sub_C_Reply_Content, 
       Sub_C_Reply_Time, Sub_C_Reply_Likes, Reply_Length
FROM video_comments_all_v2;
"""
df_comments_all = fetch_data(query_comments_all)

# **查询 `video_comments_appended`**
query_comments_appended = """
SELECT id, Video_ID, Title, Search_item, Tag, Comment_ID, Commentator_Name, 
       Commentator_Level, Comment_Content, Comment_Time, Comment_Likes, Comment_Length, 
       Sub_C_Reply_ID, Sub_C_Replier_Name, Sub_C_Replier_Level, Sub_C_Reply_Content, 
       Sub_C_Reply_Time, Sub_C_Reply_Likes, Reply_Length
FROM video_comments_appended;
"""
df_comments_appended = fetch_data(query_comments_appended)

# **匹配关键字**
df_comments_all["关键字"] = df_comments_all["Comment_Content"].apply(match_keywords)
df_comments_appended["关键字"] = df_comments_appended["Comment_Content"].apply(match_keywords)

df_comments_all_replies = df_comments_all.copy()
df_comments_appended_replies = df_comments_appended.copy()

df_comments_all_replies["关键字"] = df_comments_all_replies["Sub_C_Reply_Content"].apply(match_keywords)
df_comments_appended_replies["关键字"] = df_comments_appended_replies["Sub_C_Reply_Content"].apply(match_keywords)

# **过滤匹配到关键字的数据**
df_comments_all = df_comments_all.dropna(subset=["关键字"])
df_comments_appended = df_comments_appended.dropna(subset=["关键字"])
df_comments_all_replies = df_comments_all_replies.dropna(subset=["关键字"])
df_comments_appended_replies = df_comments_appended_replies.dropna(subset=["关键字"])

# **写入 Excel**
with pd.ExcelWriter("comment_bullet_keywords_extraction.xlsx") as writer:
    df_bulletchats.to_excel(writer, sheet_name="bulletchat", index=False)
    
    df_comments_all[
        ["id", "Video_ID", "Title", "Search_item", "Tag", "Comment_ID", "Commentator_Name",
         "Commentator_Level", "Comment_Content", "Comment_Time", "Comment_Likes", "Comment_Length", "关键字"]
    ].to_excel(writer, sheet_name="comment_all", index=False)
    
    df_comments_all_replies[
        ["Sub_C_Reply_ID", "Sub_C_Replier_Name", "Sub_C_Replier_Level", "Sub_C_Reply_Content",
         "Sub_C_Reply_Time", "Sub_C_Reply_Likes", "Reply_Length", "关键字"]
    ].to_excel(writer, sheet_name="comment_all_reply", index=False)

    df_comments_appended[
        ["id", "Video_ID", "Title", "Search_item", "Tag", "Comment_ID", "Commentator_Name",
         "Commentator_Level", "Comment_Content", "Comment_Time", "Comment_Likes", "Comment_Length", "关键字"]
    ].to_excel(writer, sheet_name="comment_appended", index=False)

    df_comments_appended_replies[
        ["Sub_C_Reply_ID", "Sub_C_Replier_Name", "Sub_C_Replier_Level", "Sub_C_Reply_Content",
         "Sub_C_Reply_Time", "Sub_C_Reply_Likes", "Reply_Length", "关键字"]
    ].to_excel(writer, sheet_name="comment_appended_reply", index=False)

print("数据提取完成，保存至 `comment_bullet_keywords_extraction.xlsx` 🎉")