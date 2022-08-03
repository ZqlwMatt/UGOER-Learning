select (cast((date_year / 10) as int) * 10) || 's' as decade, count(*) as cnt
from release
	inner join release_info info on release.id = info.release
where date_year >= 1900 and status = 1
group by decade
order by cnt desc;