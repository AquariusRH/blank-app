import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dateutil import relativedelta as datere
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import os
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# ==================== 0. 頁面與字型設定 ====================
st.set_page_config(page_title="Jockey Race", layout="wide")

# --- 自動處理中文字型 (專為 Streamlit Cloud 設計) ---
FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
FONT_FILE = "NotoSansCJKtc-Regular.otf"

@st.cache_resource
def get_chinese_font():
    # 如果字型檔不存在，則下載
    if not os.path.exists(FONT_FILE):
        with st.spinner("正在下載中文字型 (首次運行需要)..."):
            try:
                r = requests.get(FONT_URL)
                with open(FONT_FILE, "wb") as f:
                    f.write(r.content)
            except:
                st.warning("無法下載中文字型，圖表文字可能顯示為方框。")
                return None
    
    # 加入字型管理器
    if os.path.exists(FONT_FILE):
        fm.fontManager.addfont(FONT_FILE)
        # 設定 Matplotlib 全局字型
        plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_FILE).get_name()
    return FONT_FILE

# 初始化字型
get_chinese_font()

st.title("🏇 Jockey Race 賽馬預測 (Streamlit 版)")

# ==================== 1. Session State 初始化 ====================
def init_session_state():
    defaults = {
        'monitoring': False, # 控制是否正在監控
        'reset': False,
        'odds_dict': {},
        'investment_dict': {},
        'overall_investment_dict': {},
        'weird_dict': {},
        'diff_dict': {},
        'race_dict': {},
        'post_time_dict': {},
        'numbered_dict': {},
        'race_dataframes': {},
        'ucb_dict': {},
        'api_called': False,
        'last_update': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================== 2. 數據下載與處理函數 ====================

def get_investment_data(Date, place, race_no, methodlist):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {'Content-Type': 'application/json'}
    payload_investment = {
        "operationName": "racing",
        "variables": {
            "date": str(Date),
            "venueCode": place,
            "raceNo": int(race_no),
            "oddsTypes": methodlist
        },
        "query": """
        query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
            raceMeetings(date: $date, venueCode: $venueCode) {
            poolInvs: pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
                id
                oddsType
                investment
            }
            }
        }
        """
    }
    try:
        response = requests.post(url, headers=headers, json=payload_investment, timeout=5)
        if response.status_code == 200:
            investment_data = response.json()
            investments = {k: [] for k in ["WIN", "PLA", "QIN", "QPL", "FCT", "TRI", "FF"]}
            
            race_meetings = investment_data.get('data', {}).get('raceMeetings', [])
            if race_meetings:
                for meeting in race_meetings:
                    pool_invs = meeting.get('poolInvs', [])
                    for pool in pool_invs:
                        if place not in ['ST','HV']:
                            id_val = pool.get('id', '')
                            if len(id_val) >= 10 and id_val[8:10] != place:
                                continue                
                        odds_type = pool.get('oddsType')
                        if odds_type in investments:
                            investments[odds_type].append(float(pool.get('investment', 0)))
            return investments
    except Exception as e:
        print(f"Error fetching investment: {e}")
    return None

def get_odds_data(Date, place, race_no, methodlist):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {'Content-Type': 'application/json'}
    payload_odds = {
        "operationName": "racing",
        "variables": {
            "date": str(Date),
            "venueCode": place,
            "raceNo": int(race_no),
            "oddsTypes": methodlist
        },
        "query": """
        query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
            raceMeetings(date: $date, venueCode: $venueCode) {
            pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
                id
                oddsType
                oddsNodes {
                combString
                oddsValue
                }
            }
            }
        }
        """
    }
    try:
        response = requests.post(url, headers=headers, json=payload_odds, timeout=5)
        if response.status_code == 200:
            odds_data = response.json()
            odds_values = {k: [] for k in ["WIN", "PLA", "QIN", "QPL", "FCT", "TRI", "FF"]}
            
            race_meetings = odds_data.get('data', {}).get('raceMeetings', [])
            for meeting in race_meetings:
                pm_pools = meeting.get('pmPools', [])
                for pool in pm_pools:
                    if place not in ['ST', 'HV']:
                        id_val = pool.get('id', '')
                        if len(id_val) >= 10 and id_val[8:10] != place:
                            continue
                    
                    odds_type = pool.get('oddsType')
                    if not odds_type or odds_type not in odds_values:
                        continue
                        
                    odds_nodes = pool.get('oddsNodes', [])
                    for node in odds_nodes:
                        oddsValue = node.get('oddsValue')
                        if oddsValue == 'SCR':
                            oddsValue = np.inf
                        else:
                            try:
                                oddsValue = float(oddsValue)
                            except (ValueError, TypeError):
                                continue
                        
                        if odds_type in ["QIN", "QPL", "FCT", "TRI", "FF"]:
                            comb_string = node.get('combString')
                            if comb_string:
                                odds_values[odds_type].append((comb_string, oddsValue))
                        else:
                            odds_values[odds_type].append(oddsValue)
            
            # Sort complex pools
            for ot in ["QIN", "QPL", "FCT", "TRI", "FF"]:
                odds_values[ot].sort(key=lambda x: x[0], reverse=False)
            return odds_values
    except Exception as e:
        print(f"Error fetching odds: {e}")
    return None

def save_odds_data(time_now, odds, methodlist):
    for method in methodlist:
        if method in ['WIN', 'PLA']:
            if not odds[method]: continue
            if st.session_state.odds_dict[method].empty:
                st.session_state.odds_dict[method] = pd.DataFrame(columns=np.arange(1, len(odds[method]) + 1))
            st.session_state.odds_dict[method].loc[time_now] = odds[method]
        elif method in ['QIN','QPL',"FCT","TRI","FF"]:
            if odds[method]:
                combination, odds_array = zip(*odds[method])
                if st.session_state.odds_dict[method].empty:
                    st.session_state.odds_dict[method] = pd.DataFrame(columns=combination)
                st.session_state.odds_dict[method].loc[time_now] = odds_array

def save_investment_data(time_now, investment, odds, methodlist):
    for method in methodlist:
        if method not in investment or not investment[method]: continue
        
        # Calculate investment per combination (Total Pool / 1000 / Odds) approx
        # Note: This formula assumes even distribution which is an estimation
        
        if method in ['WIN', 'PLA']:
            if not odds[method]: continue
            if st.session_state.investment_dict[method].empty:
                st.session_state.investment_dict[method] = pd.DataFrame(columns=np.arange(1, len(odds[method]) + 1))
            
            # Simple estimation
            inv_val = investment[method][0]
            investment_df = [round(inv_val / 1000 / (odd if odd != np.inf and odd > 0 else 9999), 2) for odd in odds[method]]
            st.session_state.investment_dict[method].loc[time_now] = investment_df
            
        elif method in ['QIN','QPL',"FCT","TRI","FF"]:
            if odds[method]:
                combination, odds_array = zip(*odds[method])
                if st.session_state.investment_dict[method].empty:
                    st.session_state.investment_dict[method] = pd.DataFrame(columns=combination)
                
                inv_val = investment[method][0]
                investment_df = [round(inv_val / 1000 / (odd if odd != np.inf and odd > 0 else 9999), 2) for odd in odds_array]
                st.session_state.investment_dict[method].loc[time_now] = investment_df

def investment_combined(time_now, method, df):
    sums = {}
    for col in df.columns:
        try:
            num1, num2 = str(col).split(',')
            num1, num2 = int(num1), int(num2)
            col_sum = df[col].sum()
            sums[num1] = sums.get(num1, 0) + col_sum
            sums[num2] = sums.get(num2, 0) + col_sum
        except:
            continue
    return pd.DataFrame([sums], index=[time_now]) / 2

def get_overall_investment(time_now, methodlist):
    investment_df = st.session_state.investment_dict
    if investment_df['WIN'].empty: return

    no_of_horse = len(investment_df['WIN'].columns)
    total_investment_df = pd.DataFrame(index=[time_now], columns=np.arange(1, no_of_horse + 1))
    
    # Update individual method tracking
    for method in methodlist:
        if st.session_state.investment_dict[method].empty: continue
        
        last_row = st.session_state.investment_dict[method].tail(1)
        
        if method in ['WIN','PLA']:
            st.session_state.overall_investment_dict[method] = pd.concat([
                st.session_state.overall_investment_dict.get(method, pd.DataFrame()), 
                last_row
            ])
        elif method in ['QIN','QPL']:
            combined_row = investment_combined(time_now, method, last_row)
            st.session_state.overall_investment_dict[method] = pd.concat([
                st.session_state.overall_investment_dict.get(method, pd.DataFrame()), 
                combined_row
            ])

    # Sum all methods for 'overall'
    for horse in range(1, no_of_horse + 1):
        total_inv = 0
        for method in ['WIN', 'PLA', 'QIN', 'QPL']:
            if method in st.session_state.overall_investment_dict and \
               not st.session_state.overall_investment_dict[method].empty:
                try:
                    total_inv += st.session_state.overall_investment_dict[method][horse].values[-1]
                except:
                    pass
        total_investment_df[horse] = total_inv
        
    st.session_state.overall_investment_dict['overall'] = pd.concat([
        st.session_state.overall_investment_dict.get('overall', pd.DataFrame()), 
        total_investment_df
    ])

def weird_data(time_now, investments, odds, methodlist):
    for method in methodlist:
        if st.session_state.investment_dict[method].empty or len(st.session_state.investment_dict[method]) < 2:
            continue
            
        latest_investment = st.session_state.investment_dict[method].tail(1).values
        # Using previous odds for expectation calculation might be safer, but logic follows user code
        last_time_odds_df = st.session_state.odds_dict[method].tail(2).head(1)
        
        if last_time_odds_df.empty: continue
        last_time_odds = last_time_odds_df.values
        
        try:
            pool_total = investments[method][0]
            expected = pool_total / 1000 / last_time_odds
            # Handling infinity/zero division
            expected = np.where(last_time_odds == np.inf, 0, expected)
            
            diff = np.round(latest_investment - expected, 0)
            diff_df = pd.DataFrame(diff, columns=st.session_state.investment_dict[method].columns, index=[time_now])

            if method in ['WIN','PLA']:
                st.session_state.diff_dict[method] = pd.concat([st.session_state.diff_dict.get(method, pd.DataFrame()), diff_df])
            elif method in ['QIN','QPL']:
                combined_diff = investment_combined(time_now, method, diff_df)
                st.session_state.diff_dict[method] = pd.concat([st.session_state.diff_dict.get(method, pd.DataFrame()), combined_diff])
        except Exception as e:
            # st.error(f"Error in weird_data: {e}")
            pass

def change_overall(time_now, methodlist):
    total_investment = 0
    valid_calc = False
    for method in methodlist:
        if method in st.session_state.diff_dict and not st.session_state.diff_dict[method].empty:
            total_investment += st.session_state.diff_dict[method].tail(1).fillna(0).values
            valid_calc = True
            
    if valid_calc:
        # Assuming columns match overall
        cols = st.session_state.overall_investment_dict['overall'].columns
        total_df = pd.DataFrame(total_investment, index=[time_now], columns=cols)
        st.session_state.diff_dict['overall'] = pd.concat([st.session_state.diff_dict.get('overall', pd.DataFrame()), total_df])

# ==================== 3. 繪圖函數 (簡化版) ====================

def print_bubble(race_no, print_list):
    # 確保有數據
    if 'WIN' not in st.session_state.overall_investment_dict or st.session_state.overall_investment_dict['WIN'].empty:
        return

    for method in print_list:
        if method not in ['WIN&QIN', 'PLA&QPL']: continue
        
        try:
            if method == 'WIN&QIN':
                vol_win = st.session_state.overall_investment_dict.get('WIN', pd.DataFrame())
                vol_qin = st.session_state.overall_investment_dict.get('QIN', pd.DataFrame())
                diff_win = st.session_state.diff_dict.get('WIN', pd.DataFrame())
                diff_qin = st.session_state.diff_dict.get('QIN', pd.DataFrame())
                method_name = ['WIN','QIN']
            else:
                vol_win = st.session_state.overall_investment_dict.get('PLA', pd.DataFrame())
                vol_qin = st.session_state.overall_investment_dict.get('QPL', pd.DataFrame())
                diff_win = st.session_state.diff_dict.get('PLA', pd.DataFrame())
                diff_qin = st.session_state.diff_dict.get('QPL', pd.DataFrame())
                method_name = ['PLA','QPL']

            if vol_win.empty or vol_qin.empty or diff_win.empty or diff_qin.empty:
                continue

            total_volume = vol_win.tail(1) + vol_qin.tail(1)
            # Sum last 10 periods for delta
            delta_I = diff_win.tail(10).sum(axis=0) * 10
            delta_Q = diff_qin.tail(10).sum(axis=0) * 10
            
            df = pd.DataFrame({
                'horse': total_volume.columns.astype(str),
                'ΔI': delta_I.values,
                'ΔQ': delta_Q.values,
                '總投注量': total_volume.iloc[0].fillna(0).round(0).astype(int).values
            })
            
            df = df[df['總投注量'] > 0] # Filter out scratched
            if df.empty: continue

            # Normalization for bubble size
            raw_size = df['總投注量']
            bubble_size = 20 + (raw_size - raw_size.min()) / (raw_size.max() - raw_size.min() + 1e-6) * 80
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['ΔI'], y=df['ΔQ'],
                mode='markers+text',
                text=df['horse'],
                textposition="middle center",
                textfont=dict(color="white", size=14, weight="bold"),
                marker=dict(
                    size=bubble_size,
                    sizemode='area',
                    sizeref=2.*bubble_size.max()/(60**2),
                    color=df['ΔI'],
                    colorscale='Bluered_r',
                    reversescale=True,
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                hovertemplate="<b>馬號：%{text}</b><br>總量：%{customdata:,}K<br>Δ%{yaxis.title.text}: %{y:.1f}K<br>Δ%{xaxis.title.text}: %{x:.1f}K",
                customdata=df['總投注量']
            ))

            fig.add_hline(y=0, line_color="lightgrey")
            fig.add_vline(x=0, line_color="lightgrey")
            fig.update_layout(
                title=f"{method} 氣泡圖 (第{race_no}場)",
                xaxis_title=method_name[0],
                yaxis_title=method_name[1],
                height=500,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Bubble Chart Error: {e}")

# ==================== 4. 主介面邏輯 ====================

# --- 輸入區 ---
with st.sidebar:
    st.header("設定")
    Date = st.date_input('日期:', value=datetime.now(timezone(timedelta(hours=8))).date())
    place = st.selectbox('場地:', ['ST', 'HV', 'S1', 'S2'])
    race_no = st.selectbox('場次:', np.arange(1, 12))
    
    st.markdown("---")
    st.subheader("監控選項")
    
    # 監控開關
    monitoring_on = st.toggle("啟動即時監控", value=False)
    
    if st.button("重置所有數據"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- 賽事資料加載 ---
@st.cache_data(ttl=3600)
def fetch_race_card(date_str, venue):
    # 這是一個簡化的 RaceCard 抓取，只抓基本資料以顯示
    # 完整邏輯較長，這裡保留核心概念：抓取馬名與基本資料
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "operationName": "raceMeetings",
        "variables": {"date": date_str, "venueCode": venue},
        "query": """
      fragment raceFragment on Race {
        id
        no
        status
        raceName_en
        raceName_ch
        postTime
        country_en
        country_ch
        distance
        wageringFieldSize
        go_en
        go_ch
        ratingType
        raceTrack {
          description_en
          description_ch
        }
        raceCourse {
          description_en
          description_ch
          displayCode
        }
        claCode
        raceClass_en
        raceClass_ch
        judgeSigns {
          value_en
        }
      }
  
      fragment racingBlockFragment on RaceMeeting {
        jpEsts: pmPools(
          oddsTypes: [TCE, TRI, FF, QTT, DT, TT, SixUP]
          filters: ["jackpot", "estimatedDividend"]
        ) {
          leg {
            number
            races
          }
          oddsType
          jackpot
          estimatedDividend
          mergedPoolId
        }
        poolInvs: pmPools(
          oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
        ) {
          id
          leg {
            races
          }
        }
        penetrometerReadings(filters: ["first"]) {
          reading
          readingTime
        }
        hammerReadings(filters: ["first"]) {
          reading
          readingTime
        }
        changeHistories(filters: ["top3"]) {
          type
          time
          raceNo
          runnerNo
          horseName_ch
          horseName_en
          jockeyName_ch
          jockeyName_en
          scratchHorseName_ch
          scratchHorseName_en
          handicapWeight
          scrResvIndicator
        }
      }
  
      query raceMeetings($date: String, $venueCode: String) {
        timeOffset {
          rc
        }
        activeMeetings: raceMeetings {
          id
          venueCode
          date
          status
          races {
            no
            postTime
            status
            wageringFieldSize
          }
        }
        raceMeetings(date: $date, venueCode: $venueCode) {
          id
          status
          venueCode
          date
          totalNumberOfRace
          currentNumberOfRace
          dateOfWeek
          meetingType
          totalInvestment
          country {
            code
            namech
            nameen
            seq
          }
          races {
            ...raceFragment
            runners {
              id
              no
              standbyNo
              status
              name_ch
              name_en
              horse {
                id
                code
              }
              color
              barrierDrawNumber
              handicapWeight
              currentWeight
              currentRating
              internationalRating
              gearInfo
              racingColorFileName
              allowance
              trainerPreference
              last6run
              saddleClothNo
              trumpCard
              priority
              finalPosition
              deadHeat
              winOdds
              jockey {
                code
                name_en
                name_ch
              }
              trainer {
                code
                name_en
                name_ch
              }
            }
          }
          obSt: pmPools(oddsTypes: [WIN, PLA]) {
            leg {
              races
            }
            oddsType
            comingleStatus
          }
          poolInvs: pmPools(
            oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
          ) {
            id
            leg {
              number
              races
            }
            status
            sellStatus
            oddsType
            investment
            mergedPoolId
            lastUpdateTime
          }
          ...racingBlockFragment
          pmPools(oddsTypes: []) {
            id
          }
          jkcInstNo: foPools(oddsTypes: [JKC], filters: ["top"]) {
            instNo
          }
          tncInstNo: foPools(oddsTypes: [TNC], filters: ["top"]) {
            instNo
          }
        }
      }
      """
  }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            races = data.get('data', {}).get('raceMeetings', [])
            race_info = {}
            for meeting in races:
                for race in meeting.get('races', []):
                    r_no = race['no']
                    runners = race.get('runners', [])
                    
                    # 關鍵修改：過濾後備馬匹 (standbyNo 為空字串或 None)
                    filtered_runners = [r for r in runners if not r.get('standbyNo')]

                    df = pd.DataFrame([{
                        "馬號": r['no'],
                        "馬名": r['name_ch'],
                        "騎師": r['jockey']['name_ch'] if r['jockey'] else '',
                        "練馬師": r['trainer']['name_ch'] if r['trainer'] else '',
                        "近績": r.get('last6run', ''),
                        "評分": int(r.get('currentRating', '0')),
                        #"排位": int(r.get('barrierDrawNumber', '0')),
                        #"負磅": int(r.get('handicapWeight', '0'))
                    } for r in filtered_runners])
                    
                    if not df.empty:
                        # 將馬號轉換為數字並排序，確保順序正確
                        df['馬號_int'] = pd.to_numeric(df['馬號'], errors='coerce')
                        df = df.sort_values("馬號_int").drop(columns=['馬號_int']).set_index("馬號")
                    
                    # Post Time
                    pt_str = race.get("postTime")
                    pt = datetime.fromisoformat(pt_str) if pt_str else None
                    
                    race_info[r_no] = {"df": df, "post_time": pt}
            return race_info
    except Exception as e:
        print(e)
    return {}

# 嘗試加載 Race Card
date_str = str(Date)
if not st.session_state.api_called:
    with st.spinner("載入賽事資料中..."):
        race_card_data = fetch_race_card(date_str, place)
        if race_card_data:
            st.session_state.race_dataframes = {k: v['df'] for k,v in race_card_data.items()}
            st.session_state.post_time_dict = {k: v['post_time'] for k,v in race_card_data.items()}
            st.session_state.api_called = True

# --- 顯示賽事資訊 ---
if race_no in st.session_state.race_dataframes:
    pt = st.session_state.post_time_dict.get(race_no)
    pt_str = pt.strftime("%H:%M") if pt else "--:--"
    st.info(f"📍 {place} 第 {race_no} 場 | 🕒 開跑: {pt_str}")
    with st.expander("查看排位表", expanded=False):
        st.dataframe(st.session_state.race_dataframes[race_no], use_container_width=True)
else:
    st.warning("找不到此場次資料，請確認日期與場地。")

# ==================== 5. 監控循環邏輯 ====================

methodlist = ['WIN', 'PLA', 'QIN', 'QPL'] # 簡化預設
print_list = ['WIN&QIN', 'PLA&QPL']

if monitoring_on:
    # 這裡就是 "Loop" 的替代方案
    # 我們執行一次更新，然後 sleep，然後 rerun
    
    st.markdown("### 🟢 即時監控中...")
    placeholder = st.empty()
    
    # 獲取時間
    time_now = datetime.now(timezone(timedelta(hours=8)))
    time_str = time_now.strftime('%H:%M:%S')
    
    # 1. 抓取數據
    with st.spinner(f"更新數據中 ({time_str})..."):
        odds = get_odds_data(Date, place, race_no, methodlist)
        investments = get_investment_data(Date, place, race_no, methodlist)
        
        if odds and investments:
            # 2. 處理數據
            save_odds_data(time_now, odds, methodlist)
            save_investment_data(time_now, investments, odds, methodlist)
            get_overall_investment(time_now, methodlist)
            weird_data(time_now, investments, odds, methodlist)
            change_overall(time_now, methodlist)
            
            st.session_state.last_update = time_now

    # 3. 顯示圖表 (在 placeholder 中顯示最新狀態)
    with placeholder.container():
        col1, col2 = st.columns(2)
        with col1:
             st.metric("最後更新", time_str)
        
        # 顯示氣泡圖
        print_bubble(race_no, print_list)
        
        # 顯示最新賠率/投注簡表 (可選)
        if 'WIN' in st.session_state.odds_dict and not st.session_state.odds_dict['WIN'].empty:
            st.write("最新獨贏賠率變化")
            df_win = st.session_state.odds_dict['WIN'].tail(5)
            # 簡單格式化時間索引
            df_win.index = [t.strftime('%H:%M:%S') for t in df_win.index]
            st.dataframe(df_win)

    # 4. 自動刷新機制
    time.sleep(15) # 等待 15 秒
    st.rerun()     # 重新執行腳本，達到 Loop 的效果

elif st.session_state.last_update:
    st.info(f"監控已暫停。最後數據時間: {st.session_state.last_update.strftime('%H:%M:%S')}")
    # 即使暫停，也顯示最後一次的圖表
    print_bubble(race_no, print_list)
    st.markdown("---")
    st.subheader("🤖 AI 綜合預測排名")
    
    # 計算預測
    prediction_df = calculate_smart_score(race_no)
    
    if not prediction_df.empty:
        # 整理顯示格式
        display_df = prediction_df.copy()
        display_df = display_df[['Odds', 'MoneyFlow', 'FormScore', 'TotalScore']]
        display_df.columns = ['當前賠率', '近期資金流(K)', '近績評分', '🔥綜合推薦分']
        
        # 格式化
        display_df['近期資金流(K)'] = display_df['近期資金流(K)'].apply(lambda x: f"{x:.1f}")
        display_df['近績評分'] = display_df['近績評分'].astype(int)
        display_df['🔥綜合推薦分'] = display_df['🔥綜合推薦分'].apply(lambda x: f"{x:.1f}")
        
        # 用不同顏色標註前三名
        def highlight_top(row):
            if float(row['🔥綜合推薦分']) >= float(display_df['🔥綜合推薦分'].iloc[0]):
                return ['background-color: #ffcccc'] * len(row) # 第一名 紅色
            elif float(row['🔥綜合推薦分']) >= float(display_df['🔥綜合推薦分'].iloc[2]):
                return ['background-color: #ffffcc'] * len(row) # 前三名 黃色
            else:
                return [''] * len(row)

        st.dataframe(display_df.style.apply(highlight_top, axis=1), use_container_width=True)
        
        # 簡單的文字解讀
        top_horse = display_df.index[0]
        st.info(f"💡 AI 分析建議：目前綜合數據最強的是 **{top_horse}號馬**。它的實力與資金流向總和評分最高。")
