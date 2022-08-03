select date, round(cnt*100.0/cnt_sum, 2)
from(
    select date, cnt, sum(cnt) over () as cnt_sum
    from
        (select date_year || '.' ||
                case when date_month < 10 then '0' else '' end ||
                date_month as date, count(*) as cnt
        from release
            inner join release_info info on release.id = info.release
        where (date_year = 2019 and date_month >= 7) or 
            (date_year = 2020 and date_month <= 7)
        group by date_year, date_month
        order by date asc));