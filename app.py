import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. 页面设置
st.set_page_config(page_title="AI-PRS预测与推荐器")
st.title("🌊 海岛公园AI-PRS预测与推荐器")
st.write("输入使用者特征，预测PRS分数并给出改善建议。")

# 2. 读取你的CSV并训练模型
@st.cache_data
def load_and_train():
    # 你的具体路径
    file_path = "model_data.csv"
    
    df = pd.read_csv(file_path)
    
    # 这里严格对照你截图的表头写死，绝对不报错
    input_cols = [
        '您的性别', 
        '您在本城市居住的时间', 
        '您选择来这个公园的原因', 
        '您到公园的出行方式', 
        '您对公园的使用频率', 
        '您每次在公园内停留的平均时间'
    ]
    
    # 输出变量：你需要确保CSV里有 'total' 这一列
    target_col = 'total'
    
    # 剔除缺失值，防止报错
    df_model = df[input_cols + [target_col]].dropna()
    
    X = df_model[input_cols]
    y = df_model[target_col]
    
    model = LinearRegression()
    model.fit(X, y)
    return model, input_cols

model, features = load_and_train()

# 3. 构建网页输入界面
st.sidebar.header("📋 请填写使用者信息")

# 这里的选项数值和你的问卷、SPSS数据完全一致
gender = st.sidebar.selectbox("性别", [1, 2], format_func=lambda x: "男" if x==1 else "女")
years = st.sidebar.selectbox("居住年限", [1, 2, 3, 4, 5, 6], format_func=lambda x: {1:"1年以下", 2:"1-2年", 3:"3-5年", 4:"6-10年", 5:"11-20年", 6:"20年以上"}[x])
reason = st.sidebar.selectbox("来园原因", [1, 2, 3, 4, 5], format_func=lambda x: {1:"离家或公司近", 2:"路过就过来了", 3:"出名大家都说好", 4:"风景优美", 5:"其他"}[x])
walk = st.sidebar.selectbox("出行方式", [1, 2, 3, 4], format_func=lambda x: {1:"步行", 2:"公共交通", 3:"开车", 4:"其他"}[x])
frequency = st.sidebar.selectbox("使用频率", [1, 2, 3, 4, 5], format_func=lambda x: {1:"几乎每天", 2:"一周三次", 3:"经常", 4:"偶尔", 5:"几乎不"}[x])
stay = st.sidebar.selectbox("停留时长", [1, 2, 3, 4], format_func=lambda x: {1:"少于20分钟", 2:"20-40分钟", 3:"40-60分钟", 4:"超过60分钟"}[x])

# 4. 预测与推荐逻辑
if st.button("🚀 开始预测与推荐"):
    input_data = pd.DataFrame([[gender, years, reason, walk, frequency, stay]], columns=features)
    predicted_score = model.predict(input_data)[0]
    
    st.success(f"### 🎯 预测的PRS复愈性分数为：**{predicted_score:.3f}**")
    st.info("💡 **四维度短板提示**：海岛公园的**延展性**（空间边界感）通常是相对短板，建议在活动组织上予以弥补。")
    st.subheader("📌 推荐行动与管理建议")
    
    # 根据“使用频率”判断用户画像
    if frequency >= 4:
        st.warning("⚠️ **用户画像：低频-路过型**")
        st.write("- **推荐入口**：建议增设靠近居民区的**微型出入口**或便捷步道。")
        st.write("- **推荐时段**：午间（12:00-14:00）或傍晚（18:00-20:00），顺路停留。")
        st.write("- **区域推荐**：建议前往**静憩区**，以短时休息为主，降低体力负担。")
        st.write("- **管理建议**：该类用户卡在‘最后一公里’，建议排查步行15分钟覆盖盲区，改善沿途遮荫。")
    else:
        st.success("✅ **用户画像：高频-深度型**")
        st.write("- **推荐入口**：主入口，可无障碍通达。")
        st.write("- **推荐时段**：任意时段，尤其是清晨和周末。")
        st.write("- **区域推荐**：**活动区**（高活力区），适合社交与慢跑。")
        st.write("- **管理建议**：保护现有秩序，控制噪音，维护其复愈体验。")

    if reason != 4:
        st.write("- **认知唤醒建议**：该用户非景观导向，建议在入口处增加‘健康步道’标识，将其注意力转移到公园的疗愈功能上。")

st.caption("注：本工具基于调研数据的多元回归模型，预测结果仅作参考，不作临床诊断。")