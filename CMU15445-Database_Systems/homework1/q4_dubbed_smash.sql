SELECT artist.name, count(distinct alias.name) as cnt
FROM artist
	INNER JOIN area ON area.id = artist.area
	INNER JOIN artist_alias AS alias ON alias.artist = artist.id
WHERE area.name = 'United Kingdom' AND artist.begin_date_year > 1950
GROUP BY artist.name
ORDER BY cnt DESC
LIMIT 10;