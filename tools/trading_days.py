import pandas as pd
import pytz
import pandas_market_calendars as mcal

# 加载上海证券交易所的交易日历
sse = mcal.get_calendar('SSE')

# 获取指定日期范围内的有效交易日
start_date = '2023-01-01'
end_date = '2024-12-31'
trading_days = sse.valid_days(start_date,end_date).tz_convert('Asia/Shanghai')

# 找到每个预测日期的下一个有效交易日
def find_next_trading_day(current_date):
    # current_date 的下一天
    next_day = current_date + pd.DateOffset(days=1)
    # 查找下一个交易日
    while next_day not in trading_days.index:
        next_day += pd.DateOffset(days=1)
    return next_day


# 找到从当前日期起的第二个有效交易日
def find_t_plus_2(current_date):
    # current_date 的下一天
    next_day = current_date + pd.DateOffset(days=1)
    # 计数器，需要找到两个有效交易日
    count = 0
    while count < 2:
        if next_day.date().strftime('%Y-%m-%d') in trading_days.strftime('%Y-%m-%d'):
            count += 1
            if count == 2:
                break
        # Check if next_day is within pandas timestamp limit
        next_day += pd.DateOffset(days=1)
    return next_day
