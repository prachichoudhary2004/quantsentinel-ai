with trades as (
    select * from {{ ref('fact_trades') }}
)

select
    token_symbol,
    count(trade_id) as total_trades,
    sum(closed_pnl) as total_realized_pnl,
    avg(execution_price) as avg_execution_price,
    avg(size_usd) as avg_trade_size,
    sum(is_win) * 1.0 / nullif(sum(is_closing_trade), 0) as win_rate
from trades
group by token_symbol
