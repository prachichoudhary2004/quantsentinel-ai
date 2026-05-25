with fills as (
    select * from {{ ref('stg_hyperliquid_fills') }}
),

sentiment as (
    select * from {{ ref('stg_fear_greed') }}
)

select
    f.trade_id,
    f.account_address,
    f.token_symbol,
    f.execution_price,
    f.size_usd,
    f.closed_pnl,
    f.raw_direction,
    f.timestamp_ist,
    s.sentiment_score,
    s.sentiment_class,
    -- Construct derived metrics
    case when f.closed_pnl > 0 then 1 else 0 end as is_win,
    case when f.closed_pnl != 0 then 1 else 0 end as is_closing_trade
from fills f
left join sentiment s
    on cast(to_timestamp(f.timestamp_ist, 'dd-MM-yyyy HH:mm') as date) = s.sentiment_date
