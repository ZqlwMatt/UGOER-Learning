SELECT area.name, count(*) as cnt
FROM artist
	INNER JOIN area ON artist.area = area.id -- 去重
where artist.begin_date_year < 1850
GROUP BY area.name
ORDER BY cnt DESC
LIMIT 10;