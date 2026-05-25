with source_data as (
    select * from {{ source('raw_sources', 'hyperliquid_trader_data') }}
)

select
    cast(`Trade ID` as integer) as trade_id,
    cast(`Account` as string) as account_address,
    cast(`Coin` as string) as token_symbol,
    cast(`Execution Price` as double) as execution_price,
    cast(`Size USD` as double) as size_usd,
    cast(`Closed PnL` as double) as closed_pnl,
    cast(`Direction` as string) as raw_direction,
    cast(`Timestamp IST` as string) as timestamp_ist
from source_data
