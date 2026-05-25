with source_data as (
    select * from {{ source('raw_sources', 'bitcoin_fear_greed_index') }}
)

select
    cast(`date` as date) as sentiment_date,
    cast(`value` as integer) as sentiment_score,
    cast(`classification` as string) as sentiment_class
from source_data
