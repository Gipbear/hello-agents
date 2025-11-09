import streamlit as st
from datetime import datetime, timedelta
import time
import requests
from typing import Dict, Optional
import pandas as pd

# 设置页面配置
st.set_page_config(
    page_title="BTC Price Monitor",
    page_icon="₿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ======================
# 模块 1: 数据获取器 (data_fetcher)
# ======================
@st.cache_data(ttl=60)  # 缓存 60 秒，避免频繁调用 API
def fetch_btc_price() -> Optional[Dict]:
    """
    从 CoinGecko API 获取比特币价格和 24H 变化数据
    返回: {'price': float, 'change_24h': float, 'timestamp': datetime}
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "include_24hr_change": True,
        "include_24hr_high": True,
        "include_24hr_low": True
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            btc_data = data.get("bitcoin", {})
            return {
                "price": btc_data.get("usd", None),
                "change_24h": btc_data.get("usd_24h_change", None),
                "change_amount": btc_data.get("usd_24h_change", None),
                "timestamp": datetime.now()
            }
        else:
            st.warning(f"API 返回错误码: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.warning(f"网络请求失败: {e}")
        return None


# ======================
# 模块 2: 模拟 24H 历史趋势（用于图表展示）
# ======================
def generate_24h_trend_data(current_price: float) -> pd.DataFrame:
    """
    生成模拟的 24 小时价格趋势数据（每 1 小时一个点）
    使用随机漂移 + 小幅波动模拟真实市场
    """
    base_price = current_price
    data = []
    time_step = timedelta(hours=1)
    start_time = datetime.now() - timedelta(hours=24)
    
    for i in range(24):
        # 模拟轻微趋势
        trend = (i - 12) * 0.1  # 中间上升
        noise = (i % 7 - 3.5) * 10  # 周期性波动
        price = base_price + trend + noise
        data.append({
            "time": start_time + i * time_step,
            "price": round(price, 2)
        })

    return pd.DataFrame(data)


# ======================
# 模块 3: UI 组件 (ui_components)
# ======================
def show_price_card(price: float, change_24h: float, change_amount: float):
    """显示核心价格卡片"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="比特币价格 (USD)",
            value=f"${price:,.2f}",
            delta=f"{change_24h:+.2f}% ({change_amount:+,.2f} USD)"
        )

    with col2:
        # 显示更新时间
        st.caption(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")


def show_trend_chart(df: pd.DataFrame):
    """展示 24 小时价格趋势图"""
    st.subheader("24 小时价格趋势")
    st.line_chart(df.set_index("time")["price"])


def show_refresh_button():
    """显示刷新按钮 + 自动轮询控制"""
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔄 手动刷新", type="primary", use_container_width=True):
            st.session_state.last_refresh = time.time()

    with col2:
        st.info("自动刷新: 每 30 秒一次")


# ======================
# 模块 4: 缓存管理器 (cache_manager)
# ======================
def get_cached_data() -> Optional[Dict]:
    """获取缓存中的上次有效数据 - 用于断网降级"""
    if "last_valid_data" in st.session_state and st.session_state["last_valid_data"]:
        return st.session_state["last_valid_data"]
    return None


def update_cache_data(data: Dict):
    """更新缓存中的有效数据"""
    st.session_state["last_valid_data"] = data.copy()
    st.session_state["last_updated"] = datetime.now()


# ======================
# 主应用逻辑
# ======================
def main():
    st.title("₿ 比特币价格监控仪")
    st.markdown("实时获取比特币价格与 24 小时涨跌幅趋势。")

    # 初始化 session_state
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()

    # 获取数据
    with st.spinner("正在获取比特币价格..."):
        raw_data = fetch_btc_price()

    # 更新缓存
    if raw_data is not None:
        update_cache_data(raw_data)
    else:
        # 网络失败，使用缓存数据
        cached_data = get_cached_data()
        if cached_data is not None:
            st.warning("⚠️ 无法获取最新数据，使用缓存值（上次更新: {})".format(
                cached_data["timestamp"].strftime('%H:%M:%S')
            ))
            raw_data = cached_data
        else:
            st.error("❌ 无法获取比特币价格，网络或 API 出现问题。")
            st.stop()

    # 展示核心价格卡片
    show_price_card(
        price=raw_data["price"],
        change_24h=raw_data["change_24h"],
        change_amount=raw_data["change_amount"]
    )

    # 生成并展示 24 小时趋势图
    trend_df = generate_24h_trend_data(raw_data["price"])
    show_trend_chart(trend_df)

    # 显示刷新按钮
    show_refresh_button()

    # 自动轮询逻辑（每30秒重载页面）
    current_time = time.time()
    if current_time - st.session_state["last_refresh"] >= 30:
        st.session_state["last_refresh"] = current_time
        st.rerun()


if __name__ == "__main__":
    main()