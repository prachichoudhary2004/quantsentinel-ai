with trades as (
    select * from {{ ref('fact_trades') }}
)

select
    account_address,
    count(trade_id) as total_trades,
    sum(case when is_closing_trade = 1 then 1 else 0 end) as realized_trades,
    sum(closed_pnl) as total_realized_pnl,
    avg(case when is_closing_trade = 1 then closed_pnl else null end) as avg_pnl_per_trade,
    avg(size_usd) as avg_position_size,
    -- Win rate calculation
    sum(is_win) * 1.0 / nullif(sum(is_closing_trade), 0) as win_rate,
    -- Profit Factor calculation
    sum(case when closed_pnl > 0 then closed_pnl else 0 end) / 
        nullif(abs(sum(case when closed_pnl < 0 then closed_pnl else 0 end)), 0) as profit_factor
from trades
group by account_address
