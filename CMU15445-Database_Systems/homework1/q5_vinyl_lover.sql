SELECT distinct release.name
FROM release
	INNER JOIN medium ON medium.release = release.id
	INNER JOIN medium_format AS format ON medium.format = format.id
	INNER JOIN artist_credit AS credit ON credit.id = release.artist_credit
	INNER JOIN artist_credit_name AS credit_name ON credit_name.artist_credit = credit.id
	INNER JOIN artist ON credit_name.artist = artist.id
	INNER JOIN release_info AS info ON info.release = release.id
WHERE format.name like '%Vinyl' AND artist.name = 'Coldplay'
ORDER BY info.date_year, info.date_month, info.date_day;