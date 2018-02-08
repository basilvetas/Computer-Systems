SELECT SUM(is_delete) FROM tweets;

-- 1554

SELECT COUNT(reply_to) FROM tweets WHERE reply_to > 0;

-- 2531

SELECT uid, count(*) AS num FROM tweets WHERE is_delete=0 GROUP BY uid ORDER BY num DESC LIMIT 5;

-- 1269521828|5
-- 392695315|4
-- 424808364|3
-- 1706901902|3
-- 23991910|2