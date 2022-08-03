with recursive t1 (num, name) as (
	select row_number() over (order by alias.id asc), alias.name
	from artist
		inner join artist_alias as alias on artist.id = alias.artist
	where artist.name = "The Beatles"
	order by alias.id
),
t2 (num, name) as (
	select num, name from t1 where num = 1 -- initialize
	union all
	select t1.num, t2.name || ', ' || t1.name from t1, t2 where t1.num = t2.num + 1
)
select name from t2 order by num desc limit 1;