SELECT work.name, work_type.name
FROM work
	INNER JOIN (
        SELECT work.id as id, work.type as type, max(length(work.name)) as max_len
        FROM work
        GROUP BY type
    ) AS temp ON work.id = temp.id
    INNER JOIN work_type ON work.type = work_type.id
ORDER BY work.type ASC, work.name ASC;