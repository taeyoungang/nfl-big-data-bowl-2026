# Data validation report

Games checked: 122. Run time 60s.

**Totals:** 94 PASS, 28 WARN, 6 FAIL, 14 INFO

PASS = no violations. WARN = violations at or below the allowed rate. FAIL = violations above the allowed rate. Rows that fail or warn are saved in `validation/exceptions/`.

## 1. Keys, joins and formats

| Status | Check | Result | Allowed | Note |
|---|---|---|---|---|
| PASS | games.gameId is unique | 0 of 122 (0.00%) violate | 0 |  |
| PASS | plays (gameId, playId) is unique | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | players.nflId is unique | 0 of 1,679 (0.00%) violate | 0 |  |
| PASS | pff (gameId, playId, nflId) is unique | 0 of 188,254 (0.00%) violate | 0 |  |
| PASS | every play belongs to a game in games.csv | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | every game has a tracking file | 0 of 122 (0.00%) violate | 0 |  |
| PASS | every tracking file belongs to a game in games.csv | 0 of 122 (0.00%) violate | 0 |  |
| PASS | every PFF row belongs to a play in plays.csv | 0 of 188,254 (0.00%) violate | 0 |  |
| PASS | every PFF nflId is in players.csv | 0 of 188,254 (0.00%) violate | 0 |  |
| PASS | offense and defense are the two teams of that game | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | games.week matches the game date (NFL weeks start on Thursday) | 0 of 122 (0.00%) violate | 0 |  |
| PASS | gameId starts with the game date (YYYYMMDD) | 0 of 122 (0.00%) violate | 0 |  |
| PASS | all games are season 2021 | 0 of 122 (0.00%) violate | 0 |  |

## 2. plays.csv: internal logic

| Status | Check | Result | Allowed | Note |
|---|---|---|---|---|
| PASS | down is 1-4 (0 only for two-point tries) | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | down = 0 exactly on two-point tries | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | quarter is 1-5 | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | gameClock is MM:SS between 0:00 and 15:00 | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | yardsToGo is between 0 and 50 | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | defendersInBox is between 1 and 11 when present | 0 of 8,550 (0.00%) violate | 0 |  |
| INFO | playId order vs game clock |  |  | 14 of 8,557 plays have a playId that is not in clock order, so playId is not a clock. The score checks below sort by quarter and game clock instead. |
| PASS | home score never decreases (plays sorted by quarter and clock) | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | visitor score never decreases (plays sorted by quarter and clock) | 0 of 8,557 (0.00%) violate | 0 |  |
| INFO | largest score change between consecutive rows |  |  | 14 points. Not a defect: plays.csv holds dropbacks only, so consecutive rows can span several scoring drives. Section 6 checks every row's pre-snap score directly against nflverse. |
| PASS | absoluteYardlineNumber agrees with yardlineSide/Number and tracked play direction | 0 of 8,556 (0.00%) violate | 0 |  |
| PASS | yardlineSide is blank only at midfield (yard line 50) | 0 of 125 (0.00%) violate | 0 |  |
| PASS | yardsToGo never exceeds the distance to the goal line | 0 of 8,556 (0.00%) violate | 0 |  |
| PASS | no penalty -> playResult equals prePenaltyPlayResult (fumble plays excluded) | 0 of 7,663 (0.00%) violate | 0 |  |
| INFO | how playResult is set on plays with penaltyYards |  |  | penalty yards replace the play: 543; play yards kept (penalty declined or offsetting): 121; play yards + penalty yards: 83; other: 9. When a penalty cancels the play (No Play), the penalty yards become the result. |
| FAIL | with one penalty, playResult is the penalty yards, the play yards, or their sum | 8 of 756 (1.06%) violate | 1.0% |  |
| PASS | a named foul implies penaltyYards is present | 0 of 736 (0.00%) violate | 0 |  |
| INFO | penaltyYards without a foul name |  |  | 20 plays (team penalties such as Illegal Formation have no player to name). |
| PASS | passResult S <-> description says 'sacked' | 0 of 8,557 (0.00%) violate | 1.0% |  |
| WARN | passResult IN <-> description says 'intercepted' | 6 of 8,557 (0.07%) violate | 1.0% |  |
| PASS | passResult I -> description says 'incomplete' | 0 of 2,755 (0.00%) violate | 5.0% |  |
| WARN | passResult C -> description is not incomplete/intercepted/sacked | 5 of 4,620 (0.11%) violate | 1.0% |  |
| PASS | passResult R -> description says 'scramble' | 0 of 449 (0.00%) violate | 5.0% |  |
| WARN | yards written in the description equal prePenaltyPlayResult (C/S/R plays without penalty/fumble) | 12 of 5,085 (0.24%) violate | 1.0% |  |
| PASS | coverage type matches coverage name (Cover-0/1/2-Man = Man; Cover-2/3/4/6 = Zone) | 0 of 8,069 (0.00%) violate | 1.0% |  |
| WARN | offensive personnel adds up to 11 (QB and 5 OL implied) | 1 of 8,557 (0.01%) violate | 1.0% |  |
| WARN | defensive personnel adds up to 11 | 1 of 8,557 (0.01%) violate | 1.0% |  |
| WARN | personnelO (RB/TE/WR counts) matches the positions of the 11 offensive players | 65 of 8,557 (0.76%) violate | 10.0% | Compared with officialPosition in players.csv. Small gaps are expected because personnel labels follow the play, not the roster. |
| WARN | personnelD (DL/LB/DB counts) matches the positions of the 11 defensive players | 393 of 8,557 (4.59%) violate | 10.0% |  |
| INFO | pff_playAction vs 'play_action' tracking event |  |  | agree on 98.6% of plays. PFF flags 2,038 play-action plays, tracking tags 1,971. Two different labelers, so some disagreement is expected. |

## 3. players.csv

| Status | Check | Result | Allowed | Note |
|---|---|---|---|---|
| PASS | height is formatted F-I | 0 of 1,679 (0.00%) violate | 0 |  |
| PASS | height is between 5'0" and 7'0" | 0 of 1,679 (0.00%) violate | 0 |  |
| PASS | weight is between 150 and 400 lb | 0 of 1,679 (0.00%) violate | 0 |  |
| PASS | age on 2021-09-09 is between 20 and 45 | 0 of 1,433 (0.00%) violate | 0 |  |
| INFO | missing values |  |  | birthDate missing for 246 players (14.7%), collegeName missing for 224. |
| PASS | officialPosition is a known position | 0 of 1,679 (0.00%) violate | 0 |  |
| WARN | weights are plausible for the position (QB, OL, DB, WR, TE, RB) | 1 of 1,143 (0.09%) violate | 1.0% |  |
| INFO | duplicate display names |  |  | 8 names are shared by more than one nflId (different people with the same name are legitimate). |
| PASS | every player in players.csv appears in some tracking file | 0 of 1,679 (0.00%) violate | 0 |  |
| PASS | every tracked nflId is in players.csv | 0 of 188,254 (0.00%) violate | 0 |  |

## 4. PFF scouting

| Status | Check | Result | Allowed | Note |
|---|---|---|---|---|
| PASS | exactly one 'Pass' (QB) row per play | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | 11 offensive roles per play | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | 11 defensive roles per play | 0 of 8,557 (0.00%) violate | 0 |  |
| WARN | the 'Pass' player is listed as a QB in players.csv | 1 of 8,557 (0.01%) violate | 0.5% |  |
| PASS | pff_hit contains only 0 or 1 | 0 of 94,127 (0.00%) violate | 0 |  |
| PASS | pff_hurry contains only 0 or 1 | 0 of 94,127 (0.00%) violate | 0 |  |
| PASS | pff_sack contains only 0 or 1 | 0 of 94,127 (0.00%) violate | 0 |  |
| PASS | pff_beatenByDefender contains only 0 or 1 | 0 of 48,087 (0.00%) violate | 0 |  |
| PASS | pff_hitAllowed contains only 0 or 1 | 0 of 48,087 (0.00%) violate | 0 |  |
| PASS | pff_hurryAllowed contains only 0 or 1 | 0 of 48,087 (0.00%) violate | 0 |  |
| PASS | pff_sackAllowed contains only 0 or 1 | 0 of 48,087 (0.00%) violate | 0 |  |
| PASS | pff_backFieldBlock contains only 0 or 1 | 0 of 47,903 (0.00%) violate | 0 |  |
| PASS | hit/hurry/sack credited only to defensive roles | 0 of 94,127 (0.00%) violate | 0 |  |
| PASS | 'allowed' flags set only for offensive roles | 0 of 48,087 (0.00%) violate | 0 |  |
| WARN | plays ending in a sack have a defender credited with a sack | 7 of 543 (1.29%) violate | 5.0% |  |
| PASS | a PFF-credited sack means the dropback ended in a sack | 0 of 536 (0.00%) violate | 5.0% |  |
| WARN | defender sack credit and blocker 'sack allowed' appear together | 188 of 8,557 (2.20%) violate | 5.0% |  |
| WARN | the 'blocked player' is a defender on that play | 2 of 46,526 (0.00%) violate | 1.0% |  |
| PASS | every PFF row has a matching tracked player | 0 of 188,254 (0.00%) violate | 0 |  |
| PASS | every tracked player has a PFF row | 0 of 188,254 (0.00%) violate | 0 |  |
| PASS | PFF offense/defense role agrees with the tracked player's team | 0 of 188,254 (0.00%) violate | 0 |  |
| WARN | just before the snap, LT-LG-C-RG-RT are lined up in the right left-to-right order for the tracked direction | 1 of 8,502 (0.01%) violate | 3.0% | Strictly in expected order on 100.0% of plays; in the mirror-image order on 0.0%. This confirms three things at once: PFF alignment labels, tracking coordinates, and playDirection. |
| WARN | ball is within 1.5 yd of the center just before the snap | 84 of 8,504 (0.99%) violate | 3.0% | median distance 0.50 yd. |

## 5. Tracking vs plays.csv

| Status | Check | Result | Allowed | Note |
|---|---|---|---|---|
| PASS | every play in plays.csv has tracking | 0 of 8,557 (0.00%) violate | 0.1% |  |
| PASS | every tracked play is in plays.csv | 0 of 8,557 (0.00%) violate | 0 |  |
| WARN | every play has a snap event | 24 of 8,557 (0.28%) violate | 1.0% |  |
| INFO | ball_snap vs autoevent_ballsnap |  |  | both tags exist on 3,756 plays; frame difference median 1, 190 differ by more than 2 frames. |
| PASS | each play has 11 offense, 11 defense, 1 ball, and no unknown team code | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | no two players on one team share a jersey number in a play | 0 of 188,254 (0.00%) violate | 0.1% |  |
| WARN | completed/incomplete/intercepted passes have a pass_forward event | 3 of 7,565 (0.04%) violate | 1.0% |  |
| FAIL | sacks have no pass_forward event | 30 of 543 (5.52%) violate | 1.0% |  |
| WARN | sacks have a qb_sack or qb_strip_sack event | 35 of 543 (6.45%) violate | 20.0% |  |
| PASS | qb_sack events occur only on plays whose passResult is S | 0 of 451 (0.00%) violate | 5.0% |  |
| WARN | ball is within 1.5 yd of the line of scrimmage just before the snap | 81 of 8,503 (0.95%) violate | 2.0% | median offset -0.16 yd, 95th percentile |offset| 0.72 yd. |
| PASS | offense is on the backfield side of the ball and defense on the other side just before the snap (confirms playDirection) | 0 of 8,503 (0.00%) violate | 0 |  |
| PASS | no offensive player is more than 2 yd past the line of scrimmage just before the snap | 0 of 8,503 (0.00%) violate | 3.0% | Whole-yard rounding of the line of scrimmage, motion men and tracking noise can cause a few. |
| WARN | no defender is more than 2 yd behind the line of scrimmage just before the snap | 5 of 8,503 (0.06%) violate | 3.0% |  |

## 6. Cross-check against nflverse play-by-play (independent source)

| Status | Check | Result | Allowed | Note |
|---|---|---|---|---|
| PASS | every game exists in nflverse | 0 of 122 (0.00%) violate | 0 |  |
| PASS | week matches nflverse | 0 of 122 (0.00%) violate | 0 |  |
| PASS | home team matches nflverse | 0 of 122 (0.00%) violate | 0 |  |
| PASS | visitor team matches nflverse | 0 of 122 (0.00%) violate | 0 |  |
| PASS | game date matches nflverse | 0 of 122 (0.00%) violate | 0 |  |
| PASS | every play in plays.csv exists in nflverse (same gameId and playId) | 0 of 8,557 (0.00%) violate | 0 |  |
| WARN | description text matches nflverse (first 60 characters, jersey numbers removed) | 3 of 8,557 (0.04%) violate | 5.0% |  |
| PASS | quarter matches | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | down matches | 0 of 8,522 (0.00%) violate | 0 |  |
| PASS | yards to go matches | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | possession team matches | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | defensive team matches | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | distance to goal line (from tracking direction + absoluteYardlineNumber) matches nflverse yardline_100 | 0 of 8,556 (0.00%) violate | 1.0% | An independent check on tracking coordinates, direction, and the line of scrimmage. |
| PASS | game clock matches (seconds) | 0 of 8,557 (0.00%) violate | 1.0% |  |
| PASS | pre-snap home and visitor scores match nflverse | 0 of 8,557 (0.00%) violate | 1.0% |  |
| INFO | plays with an accepted penalty |  |  | 623 plays. nflverse leaves complete/incomplete unset when a penalty cancels the play, while plays.csv still records what the pass did, so those plays are left out of the pass-outcome comparison below. |
| WARN | passResult C <-> nflverse complete_pass (plays without a penalty) | 20 of 7,934 (0.25%) violate | 2.0% |  |
| WARN | passResult I <-> nflverse incomplete_pass (plays without a penalty) | 17 of 7,934 (0.21%) violate | 2.0% |  |
| PASS | passResult IN <-> nflverse interception (plays without a penalty) | 0 of 7,934 (0.00%) violate | 2.0% |  |
| WARN | passResult S <-> nflverse sack (plays without a penalty) | 1 of 7,934 (0.01%) violate | 2.0% |  |
| PASS | passResult R <-> nflverse qb_scramble (plays without a penalty) | 0 of 7,934 (0.00%) violate | 2.0% |  |
| WARN | yards gained (plays without a penalty) match nflverse | 24 of 7,928 (0.30%) violate | 1.0% |  |
| INFO | completeness |  |  | nflverse has 9,608 QB dropbacks in these 122 games; plays.csv has 8,557. 1,540 nflverse dropbacks are absent from plays.csv (16.0%); 26 of them involve a penalty. |
| INFO | selection: dropbacks IN plays.csv |  |  | n=7,928; completion rate 62.9%; sack rate 6.5%; median air yards 6; thrown at or behind the line (air yards <= 0): 12% |
| INFO | selection: dropbacks NOT in plays.csv |  |  | n=1,514; completion rate 79.6%; sack rate 1.9%; median air yards -2; thrown at or behind the line (air yards <= 0): 70%. The omitted plays are mostly passes at or behind the line of scrimmage, so plays.csv is not a random sample of dropbacks. |
| INFO | plays.csv rows that nflverse does not mark as QB dropbacks |  |  | 489 (for example two-point tries or plays nullified by penalty are labelled differently). |

## 7. Tracking: structure and physics (every row of every file)

| Status | Check | Result | Allowed | Note |
|---|---|---|---|---|
| PASS | file name gameId equals gameId column | 0 of 8,314,178 (0.00%) violate | 0 |  |
| PASS | every frame has exactly 23 rows (22 players + ball) | 0 of 361,486 (0.00%) violate | 0 |  |
| PASS | no duplicate (play, object, frame) rows | 0 of 8,314,178 (0.00%) violate | 0 |  |
| PASS | frames per object run 1..N with no gaps | 0 of 196,811 (0.00%) violate | 0 |  |
| PASS | all objects in a play have the same frame count | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | one playDirection per play | 0 of 8,557 (0.00%) violate | 0 |  |
| PASS | each object has one team and one jersey number per play | 0 of 196,811 (0.00%) violate | 0 |  |
| PASS | an event tag is written on all 23 rows of its frame | 0 of 27,862 (0.00%) violate | 0 |  |
| PASS | at most one event type per frame | 0 of 27,862 (0.00%) violate | 0 |  |
| PASS | snap happens before pass_forward | 0 of 7,583 (0.00%) violate | 0 |  |
| FAIL | play duration from timestamps matches frames / 10 Hz (+/-1 s) | 6 of 8,557 (0.07%) violate | 0 |  |
| PASS | timestamps never go backwards | 0 of 361,486 (0.00%) violate | 0 |  |
| PASS | no more than 10 frames share one timestamp second | 0 of 43,827 (0.00%) violate | 0 |  |
| WARN | x within 0..120 yd | 112 of 7,952,692 (0.00%) violate | 0.1% |  |
| WARN | y within 0..53.3 yd | 857 of 7,952,692 (0.01%) violate | 0.1% |  |
| FAIL | x, y no more than 1 yd outside the field | 311 of 7,952,692 (0.00%) violate | 0 |  |
| PASS | dis equals the real position change between frames (+/-0.05 yd) | 0 of 7,764,438 (0.00%) violate | 0 |  |
| WARN | speed s is close to 10 x dis (+/-1.5 yd/s) | 2,293 of 7,764,438 (0.03%) violate | 1.0% |  |
| FAIL | no teleporting: position change <= 1.5 yd per frame (15 yd/s) | 27 of 7,764,438 (0.00%) violate | 0 |  |
| PASS | player speed <= 12 yd/s (about 27 mph) | 0 of 7,952,692 (0.00%) violate | 0 |  |
| FAIL | player acceleration <= 15 yd/s^2 | 10 of 7,952,692 (0.00%) violate | 0 |  |
| PASS | speed and acceleration are never negative | 0 of 7,952,692 (0.00%) violate | 0 |  |
| PASS | orientation o and direction dir within 0..360 | 0 of 7,952,692 (0.00%) violate | 0 |  |
| PASS | player rows all have o and dir | 0 of 7,952,692 (0.00%) violate | 0 |  |
| WARN | dir points along the direction of travel (+/-20 deg when moving > 3 yd/s) | 198 of 2,823,792 (0.01%) violate | 5.0% |  |
| INFO | observed maxima |  |  | max player speed 11.18 yd/s (22.9 mph), max player acceleration 21.92 yd/s^2, max one-frame move 3.45 yd, max ball speed 29.3 yd/s. |
