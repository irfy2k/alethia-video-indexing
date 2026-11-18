from common import aggregate, video_lines

# half rule: 2 of 4 frames -> present; 1 of 3 -> absent
assert aggregate([[2], [2], [], []])[0] == ["car"]
assert aggregate([[2], [], []])[0] == []
# order follows CLASS_IDS, not detection order
assert aggregate([[67, 0, 63]])[0] == ["person", "laptop", "cell phone"]
# people: median rounded down at .5
assert aggregate([[0], [0, 0]])[1] == 1
assert aggregate([[0, 0], [0, 0, 0]])[1] == 2
assert aggregate([[0], [], [0, 0]])[1] == 1
assert aggregate([[], [], [0]])[1] == 0
assert aggregate([[0], [0], [0, 0], [0, 0]])[1] == 1  # median 1.5 -> 1
assert aggregate([]) == ([], 0)
# empty seconds + last second included
L = video_lines("v:0", [(0.0, [0]), (0.5, [0]), (3.2, [16])])
assert [l["second"] for l in L] == [0, 1, 2, 3]
assert L[0] == {"video": "v:0", "second": 0, "objects": ["person"], "people": 1}
assert L[1]["objects"] == [] and L[1]["people"] == 0 and L[2]["people"] == 0
assert L[3]["objects"] == ["dog"]
assert video_lines("v:0", []) == []
print("ok")
