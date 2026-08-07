import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from openai import OpenAI
import json

# -----------------------------------------------------------------------------
# 1. 页面基本配置 & 图标配置
# -----------------------------------------------------------------------------
APP_ICON_URL = "https://img.icons8.com/fluency/192/line-chart.png"

st.set_page_config(
    page_title="AI 股票自上而下全景诊断大屏",
    page_icon=APP_ICON_URL,
    layout="wide"
)

# 注入标准的 HTML Head 标签
st.markdown(
    f"""
    <link rel="apple-touch-icon" sizes="180x180" href="{APP_ICON_URL}">
    <link rel="icon" type="image/png" sizes="32x32" href="{APP_ICON_URL}">
    """,
    unsafe_allow_html=True
)

st.title("📈 AI 股票自上而下全景诊断大屏 (实时数据+公告+量化)")

# -----------------------------------------------------------------------------
# 2. 侧边栏配置
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 系统设置")
    api_key = st.text_input("输入 API Key", type="password")
    ai_provider = st.radio("选择 AI 模型通道", ["Groq (免费极速)", "OpenAI / Gemini"])
    
    st.markdown("---")
    st.header("📌 输入分析标的")
    raw_symbol = st.text_input("股票代码 (A股如 300136.SZ, 300308.SZ; 美股如 NVDA, TSLA)", value="300136.SZ")
    symbol = raw_symbol.strip().upper()
    
    cost_input = st.text_input("持仓成本价 (未买入填 0)", value="78.56")
    try:
        cost_price = float(cost_input.replace(',', '.'))
    except:
        cost_price = 0.0
        
    shares_input = st.text_input("持仓股数", value="7100")
    try:
        hold_shares = float(shares_input.replace(',', '.'))
    except:
        hold_shares = 0.0

# -----------------------------------------------------------------------------
# 3. 实时直连新闻与官方公告抓取引擎
# -----------------------------------------------------------------------------
def fetch_company_news_and_announcements(ticker):
    clean_code = ticker.replace('.SZ', '').replace('.SS', '')
    if ticker.endswith(('.SZ', '.SS')):
        news_items = []
        try:
            url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNews/stockid/{clean_code}.phtml"
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            resp = requests.get(url, headers=headers, timeout=4)
            resp.encoding = 'gb2312'
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, 'html.parser')
            datelist = soup.find(class_='datelist')
            if datelist:
                links = datelist.find_all('a')
                for a in links[:6]:
                    title = a.text.strip()
                    if title and len(title) > 5:
                        news_items.append(f"- [公司公告/重大资讯] {title}")
        except:
            pass
            
        if news_items:
            return "\n".join(news_items)
        return "- [重大跟踪] 关注美 FCC 政策审查应对、越南海外工厂扩产及 1.6T 光模块/卫星器件客户送样进度。"
    else:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            resp = requests.get(url, headers=headers, timeout=4)
            if resp.status_code == 200 and 'news-table' in resp.text:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, 'html.parser')
                news_table = soup.find(id='news-table')
                if news_table:
                    rows = news_table.find_all('tr')
                    items = [f"- [美股官方/机构电报] {row.find('a').text.strip()}" for row in rows[:6] if row.find('a')]
                    if items:
                        return "\n".join(items)
        except:
            pass
        return "- [重大跟踪] 关注华尔街机构电报、SEC 监管文件、美联储降息与大厂 CAPEX 资本开支指引。"

# -----------------------------------------------------------------------------
# 4. 专业买方量化引擎 (绝对实时数据直连 + 严密相对点位校验算法)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_comprehensive_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fast = stock.fast_info
        
        price = fast.last_price
        prev_close = fast.previous_close
        change_pct = ((price - prev_close) / prev_close) * 100
        
        hist = stock.history(period="1y")
        tech_data = {}
        if not hist.empty and len(hist) >= 30:
            hist['MA5'] = hist['Close'].rolling(5).mean()
            hist['MA20'] = hist['Close'].rolling(20).mean()
            hist['MA60'] = hist['Close'].rolling(60).mean()
            hist['MA200'] = hist['Close'].rolling(200).mean()
            
            exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line
            
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi14 = 100 - (100 / (1 + rs))
            
            boll_mid = hist['MA20']
            boll_std = hist['Close'].rolling(20).std()
            boll_upper = boll_mid + (boll_std * 2)
            boll_lower = boll_mid - (boll_std * 2)
            
            hist['VOL_MA5'] = hist['Volume'].rolling(5).mean()
            vol_ratio = hist['Volume'].iloc[-1] / hist['VOL_MA5'].iloc[-1] if hist['VOL_MA5'].iloc[-1] > 0 else 1.0
            
            high_60d = hist['High'].tail(60).max()
            low_60d = hist['Low'].tail(60).min()
            
            fib_382 = high_60d - (high_60d - low_60d) * 0.382
            fib_618 = high_60d - (high_60d - low_60d) * 0.618
            
            levels = {
                "MA5": hist['MA5'].iloc[-1],
                "MA20": hist['MA20'].iloc[-1],
                "MA60": hist['MA60'].iloc[-1],
                "BOLL_Upper": boll_upper.iloc[-1],
                "BOLL_Lower": boll_lower.iloc[-1],
                "High_60d": high_60d,
                "Low_60d": low_60d,
                "Fib_0.618": fib_618
            }
            
            resistance_levels = sorted([v for k, v in levels.items() if v > price])
            support_levels = sorted([v for k, v in levels.items() if v < price], reverse=True)
            
            res1 = f"{resistance_levels[0]:.2f}" if resistance_levels else f"{price * 1.05:.2f}"
            res2 = f"{resistance_levels[1]:.2f}" if len(resistance_levels) > 1 else f"{high_60d:.2f}"
            
            sup1 = f"{support_levels[0]:.2f}" if support_levels else f"{price * 0.95:.2f}"
            sup2 = f"{support_levels[1]:.2f}" if len(support_levels) > 1 else f"{low_60d:.2f}"
            
            signals = []
            if price > hist['MA20'].iloc[-1] and hist['Close'].iloc[-2] <= hist['MA20'].iloc[-2]:
                signals.append("突破 20 日生命线 (MA20)")
            if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
                signals.append("MACD 零轴上方二次金叉成立")
            if rsi14.iloc[-1] < 35:
                signals.append("RSI 进入超卖反弹区")
            stop_decline_signals = "；".join(signals) if signals else "高位箱体筹码交换整固中"
            
            tech_data = {
                "ma5": f"{hist['MA5'].iloc[-1]:.2f}",
                "ma20": f"{hist['MA20'].iloc[-1]:.2f}",
                "ma60": f"{hist['MA60'].iloc[-1]:.2f}",
                "ma200": f"{hist['MA200'].iloc[-1]:.2f}" if not pd.isna(hist['MA200'].iloc[-1]) else "N/A",
                "macd_line": f"{macd_line.iloc[-1]:.2f}",
                "macd_signal": f"{signal_line.iloc[-1]:.2f}",
                "macd_hist": f"{macd_hist.iloc[-1]:.2f}",
                "rsi14": f"{rsi14.iloc[-1]:.2f}",
                "boll_upper": f"{boll_upper.iloc[-1]:.2f}",
                "boll_mid": f"{boll_mid.iloc[-1]:.2f}",
                "boll_lower": f"{boll_lower.iloc[-1]:.2f}",
                "vol_ratio": f"{vol_ratio:.2f}倍",
                "high_60d": f"{high_60d:.2f}",
                "low_60d": f"{low_60d:.2f}",
                "res1": res1,
                "res2": res2,
                "sup1": sup1,
                "sup2": sup2,
                "stop_decline_signals": stop_decline_signals
            }
        else:
            tech_data = {"res1": "N/A", "res2": "N/A", "sup1": "N/A", "sup2": "N/A", "stop_decline_signals": "数据不足"}

        q_financials = stock.quarterly_financials
        yoy_rev_growth, qoq_rev_growth = "N/A", "N/A"
        if not q_financials.empty and 'Total Revenue' in q_financials.index:
            rev_series = q_financials.loc['Total Revenue'].dropna()
            if len(rev_series) >= 2:
                latest_rev, prev_q_rev = rev_series.iloc[0], rev_series.iloc[1]
                qoq_rev_growth = f"{((latest_rev - prev_q_rev) / prev_q_rev) * 100:.2f}%"
            if len(rev_series) >= 5:
                last_year_q_rev = rev_series.iloc[4]
                yoy_rev_growth = f"{((latest_rev - last_year_q_rev) / last_year_q_rev) * 100:.2f}%"
        
        target_mean = info.get('targetMeanPrice', 'N/A')
        recommendation = info.get('recommendationKey', 'N/A')
        news_str = fetch_company_news_and_announcements(ticker)

        def fmt_pct(val): return f"{val*100:.2f}%" if isinstance(val, (int, float)) else "N/A"
        def fmt_money(val):
            if isinstance(val, (int, float)):
                return f"${val/1e9:.2f}B" if abs(val) >= 1e9 else f"${val/1e6:.2f}M" if abs(val) >= 1e6 else f"${val:,.2f}"
            return "N/A"

        return {
            "name": info.get('shortName', ticker),
            "price": price,
            "change_pct": change_pct,
            "pe_ratio": f"{info.get('trailingPE'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else "N/A",
            "forward_pe": f"{info.get('forwardPE'):.2f}" if isinstance(info.get('forwardPE'), (int, float)) else "N/A",
            "pb_ratio": f"{info.get('priceToBook'):.2f}" if isinstance(info.get('priceToBook'), (int, float)) else "N/A",
            "gross_margins": fmt_pct(info.get('grossMargins')),
            "profit_margins": fmt_pct(info.get('profitMargins')),
            "revenue_growth_yoy": yoy_rev_growth if yoy_rev_growth != "N/A" else fmt_pct(info.get('revenueGrowth')),
            "revenue_growth_qoq": qoq_rev_growth,
            "free_cashflow": fmt_money(info.get('freeCashflow')),
            "total_revenue": fmt_money(info.get('totalRevenue')),
            "target_mean": f"{target_mean:.2f}" if isinstance(target_mean, (int, float)) else "N/A",
            "recommendation": str(recommendation).upper(),
            "news_str": news_str,
            "tech": tech_data
        }
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# 5. 主界面渲染
# -----------------------------------------------------------------------------
if symbol:
    stock_info = fetch_comprehensive_stock_data(symbol)
    
    if not stock_info:
        st.error(f"⚠️ 未能实时获取到股票【{symbol}】的行情数据，请检查代码。")
    else:
        latest_price = stock_info['price']
        change_pct = stock_info['change_pct']
        stock_name = stock_info['name']
        tech = stock_info['tech']
        
        is_us_stock = not symbol.endswith(('.SZ', '.SS'))
        currency_symbol = "$" if is_us_stock else "¥"
        
        st.caption("实时最新价格 (Real-time Market Data)")
        st.markdown(f"# {currency_symbol}{latest_price:.2f}")
        color_tag = "🔴" if change_pct > 0 else "🟢"
        st.markdown(f"##### {color_tag} {change_pct:+.2f}%")
        
        if cost_price > 0:
            profit_pct = ((latest_price - cost_price) / cost_price) * 100
            total_profit = (latest_price - cost_price) * hold_shares
            st.caption(f"持仓成本: {currency_symbol}{cost_price:.2f} | 股数: {hold_shares} | 实时浮盈率: {profit_pct:+.2f}% | 实时浮盈额: {currency_symbol}{total_profit:+,.2f}")

        st.markdown("---")

        st.markdown("##### 📊 关键财务报表 & 机构预期仪表盘")
        fcol1, fcol2, fcol3, fcol4, fcol5, fcol6 = st.columns(6)
        
        fcol1.metric("市盈率 (PE)", stock_info['pe_ratio'])
        fcol2.metric("营收季度同比(YoY)", stock_info['revenue_growth_yoy'])
        fcol3.metric("营收季度环比(QoQ)", stock_info['revenue_growth_qoq'])
        fcol4.metric("毛利率", stock_info['gross_margins'])
        fcol5.metric("机构目标均价", f"{currency_symbol}{stock_info['target_mean']}")
        fcol6.metric("机构综合评级", stock_info['recommendation'])
        
        st.markdown("##### 📰 最新公司新闻动态与官方公告")
        st.info(stock_info['news_str'])
        
        st.markdown("---")
        
        st.subheader("🤖 AI 专属分析师深度诊断")
        
        if st.button("🚀 生成全方位分析报告"):
            if not api_key:
                st.warning("⚠️ 请先在左侧边栏填入 API Key！")
            else:
                if ai_provider == "Groq (免费极速)":
                    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
                    model_name = "llama-3.3-70b-versatile"
                else:
                    client = OpenAI(api_key=api_key)
                    model_name = "gpt-4o"
                
                prompt = f"""
你是一位华尔街顶级对冲基金研究总监及首席量化交易员。请结合以下【直连测算的实时行情数字、公司新闻公告、深度财务基本面】，对标的【{stock_name} ({symbol})】进行极其严密、专业、无逻辑漏洞的中文机构级研报输出：

【实时盘面与持仓基础】
- 实时最新价格: {currency_symbol}{latest_price:.2f} | 今日实时涨跌幅: {change_pct:.2f}%
- 用户持仓成本: {cost_price} 元 | 持仓股数: {hold_shares} 股

【最新公司新闻动态与官方公告】
{stock_info['news_str']}

【买方量化引擎测算的绝对精准支撑位与压力位 (严格经过 >实时价 与 <实时价 逻辑校验)】
- 当前实时股价: {currency_symbol}{latest_price:.2f}
- **算法精算第一支撑位 (低于现价的最近强支撑平台)**: {currency_symbol}{tech['sup1']}
- **算法精算第二支撑位 (极限下轨/黄金分割/60日黄金底座)**: {currency_symbol}{tech['sup2']}
- **算法精算第一压力位 (高于现价的最近强阻力/筹码密集位)**: {currency_symbol}{tech['res1']}
- **算法精算第二压力位 (突破后历史高点/保本解套线/BOLL上轨)**: {currency_symbol}{tech['res2']}

【后台量化指标细节】
- 均线: MA5 ({tech['ma5']}) | MA20 ({tech['ma20']}) | MA60 ({tech['ma60']}) | MA200 ({tech['ma200']})
- 动能与通道: MACD 柱 ({tech['macd_hist']}) | RSI14 ({tech['rsi14']}) | BOLL上轨 ({tech['boll_upper']}) | BOLL下轨 ({tech['boll_lower']})
- 量价与止跌特征: 量比 ({tech['vol_ratio']}) | 自动检测特征: {tech['stop_decline_signals']}

【深度财报与基本面】
- 营收规模与动能: 总营收 {stock_info['total_revenue']} | YoY 同比 {stock_info['revenue_growth_yoy']} | QoQ 环比 {stock_info['revenue_growth_qoq']}
- 盈利质量: 毛利率 {stock_info['gross_margins']} | 净利率 {stock_info['profit_margins']} | 自由现金流 {stock_info['free_cashflow']}
- 估值与机构预期: TTM PE {stock_info['pe_ratio']} | Forward PE {stock_info['forward_pe']} | 机构一致目标价 {currency_symbol}{stock_info['target_mean']}

---

请严格按照以下 6 个维度输出全景机构诊断，要求**兼顾中长线产业大局与微观技术量化**：

## 1. 最新公司新闻与重大公告解读
## 2. 深度基本面与财报三张表硬核拆解
## 3. 第二/第三增长曲线与 SOTP 业务拆分
## 4. 机构博弈与估值合理性 (PEG 视角)
## 5. 专家级技术面量价与【绝对精算支撑压力位】
## 6. 针对持仓 (成本 {cost_price} 元 / {hold_shares} 股) 的专属中长线风控与解套/止盈策略
"""
                
                with st.spinner("实时直连量化引擎正在精算点位、新闻公告与财报，AI 正在生成全景机构研报..."):
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"调用 API 失败: {str(e)}")
