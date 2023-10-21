import math

layouts = []

layouts.append([160, 182, 288, 350, 312, 460, 160, 418])
layouts.append([161, 181, 243, 319, 243, 260, 158, 384])
layouts.append([130, 210, 280, 310, 280, 300, 180, 210])
layouts.append([110, 200, 270, 280, 225, 400, 180, 400])
layouts.append([130, 160, 275, 280, 280, 340, 160, 275])
layouts.append([180, 180, 295, 355, 168, 350, 150, 350])
layouts.append([132, 258, 248, 258, 248, 316, 132, 310])
layouts.append([133, 246, 314, 363, 302, 512, 135, 300])
layouts.append([ 90, 210, 240, 283, 200, 240, 100, 270])
layouts.append([126, 240, 240, 378, 230, 420, 132, 420])
layouts.append([140, 226, 260, 350, 266, 304, 200, 266])

# print(layouts)
b_idx = 0
q_idx = 1
s_idx = 2
c_idx = 3
total_rooms = 4

b1_idx = 0
b2_idx = 1
q1_idx = 2
q2_idx = 3
s1_idx = 4
s2_idx = 5
c1_idx = 6
c2_idx = 7
total_pts = 8

min_lenghts = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
max_lenghts = [0, 0, 0, 0, 0, 0, 0, 0]

min_area = [1000000, 1000000, 1000000, 1000000]
max_area = [0, 0, 0, 0]
# min_area_b = 100000
# min_area_q = 100000
# min_area_s = 100000
# min_area_c = 100000
# max_area_b = 0
# max_area_q = 0
# max_area_s = 0
# max_area_c = 0

for layout in layouts:
    b1 = math.floor(layout[b1_idx] / 10)
    b2 = math.ceil(layout[b2_idx] / 10)
    q1 = math.floor(layout[q1_idx] / 10)
    q2 = math.ceil(layout[q2_idx] / 10)
    s1 = math.floor(layout[s1_idx] / 10)
    s2 = math.ceil(layout[s2_idx] / 10)
    c1 = math.floor(layout[c1_idx] / 10)
    c2 = math.ceil(layout[c2_idx] / 10)

    area_b = b1 * b2 
    area_q = q1 * q2
    area_s = s1 * s2
    area_c = c1 * c2

    if b1 < min_lenghts[b1_idx]:
        min_lenghts[b1_idx] = b1
    if q1 < min_lenghts[q1_idx]:
        min_lenghts[q1_idx] = q1
    if s1 < min_lenghts[s1_idx]:
        min_lenghts[s1_idx] = s1
    if c1 < min_lenghts[c1_idx]:
        min_lenghts[c1_idx] = c1

    if b1 > max_lenghts[b1_idx]:
        max_lenghts[b1_idx] = b1
    if q1 > max_lenghts[q1_idx]:
        max_lenghts[q1_idx] = q1
    if s1 > max_lenghts[s1_idx]:
        max_lenghts[s1_idx] = s1
    if c1 > max_lenghts[c1_idx]:
        max_lenghts[c1_idx] = c1

    if b2 < min_lenghts[b2_idx]:
        min_lenghts[b2_idx] = b2
    if q2 < min_lenghts[q2_idx]:
        min_lenghts[q2_idx] = q2
    if s2 < min_lenghts[s2_idx]:
        min_lenghts[s2_idx] = s2
    if c2 < min_lenghts[c2_idx]:
        min_lenghts[c2_idx] = c2

    if b2 > max_lenghts[b2_idx]:
        max_lenghts[b2_idx] = b2
    if q2 > max_lenghts[q2_idx]:
        max_lenghts[q2_idx] = q2
    if s2 > max_lenghts[s2_idx]:
        max_lenghts[s2_idx] = s2
    if c2 > max_lenghts[c2_idx]:
        max_lenghts[c2_idx] = c2

    if area_b < min_area[b_idx]:
        min_area[b_idx] = area_b
    if area_q < min_area[q_idx]:
        min_area[q_idx] = area_q
    if area_s < min_area[s_idx]:
        min_area[s_idx] = area_s
    if area_c < min_area[c_idx]:
        min_area[c_idx] = area_c

    if area_b > max_area[b_idx]:
        max_area[b_idx] = area_b
    if area_q > max_area[q_idx]:
        max_area[q_idx] = area_q
    if area_s > max_area[s_idx]:
        max_area[s_idx] = area_s
    if area_c > max_area[c_idx]:
        max_area[c_idx] = area_c

    # print(layout)


print("b layout constraints")
print("(", min_lenghts[b1_idx], "-", max_lenghts[b1_idx], "),\t(", min_lenghts[b2_idx], "-", max_lenghts[b2_idx], ")")
print("min area:", min_area[b_idx], "\tmax area:", max_area[b_idx])
print("\n")

print("q layout constraints")
print("(", min_lenghts[q1_idx], "-", max_lenghts[q1_idx], "),\t(", min_lenghts[q2_idx], "-", max_lenghts[q2_idx], ")")
print("min area:", min_area[q_idx], "\tmax area:", max_area[q_idx])
print("\n")

print("s layout constraints")
print("(", min_lenghts[s1_idx], "-", max_lenghts[s1_idx], "),\t(", min_lenghts[s2_idx], "-", max_lenghts[s2_idx], ")")
print("min area:", min_area[s_idx], "\tmax area:", max_area[s_idx])
print("\n")

print("c layout constraints")
print("(", min_lenghts[c1_idx], "-", max_lenghts[c1_idx], "),\t(", min_lenghts[c2_idx], "-", max_lenghts[c2_idx], ")")
print("min area:", min_area[c_idx], "\tmax area:", max_area[c_idx])
print("\n")


step = 1
sizes = [0, 0, 0, 0]
nSizes = 0

for i in range(total_rooms):
    # print("\n\n\n\ni", i)
    room_min_1 = min_lenghts[i*2]
    room_min_2 = min_lenghts[(i*2) + 1]

    room_max_1 = max_lenghts[i*2]
    room_max_2 = max_lenghts[(i*2) + 1]

    for j in range(room_min_2 - room_min_1):
        # print("j", j)
        size_1 = j + room_min_1

        for k in range(room_max_2 - room_max_1):
            # print("k", k)
            size_2 = k + room_max_1
            area = size_1 * size_2

            # if(area >= min_area[i] and area <= max_area[i]):
            #     sizes[i] = sizes[i] + 1

            sizes[i] = sizes[i] + 1

nSizes = 1
for i in range(total_rooms):
    nSizes = nSizes * sizes[i]

print("sizes", sizes)
print("nSizes", nSizes)


# print("\n------------------------------------------\n\n")

# for i in range(len(min_lenghts)):
#     min_value = min_lenghts[i]
#     min_value = math.floor(min_value / 10)
#     min_lenghts[i] = min_value

# for i in range(len(max_lenghts)):
#     min_value = max_lenghts[i]
#     min_value = math.ceil(min_value / 10)
#     max_lenghts[i] = min_value


# print("b layout constraints")
# print("(", min_lenghts[b1_idx], "-", max_lenghts[b1_idx], "),\t(", min_lenghts[b2_idx], "-", max_lenghts[b2_idx], ")")
# print("min area:", min_area[b_idx], "\tmax area:", max_area[b_idx])
# print("\n")

# print("q layout constraints")
# print("(", min_lenghts[q1_idx], "-", max_lenghts[q1_idx], "),\t(", min_lenghts[q2_idx], "-", max_lenghts[q2_idx], ")")
# print("min area:", min_area[q_idx], "\tmax area:", max_area[q_idx])
# print("\n")

# print("s layout constraints")
# print("(", min_lenghts[s1_idx], "-", max_lenghts[s1_idx], "),\t(", min_lenghts[s2_idx], "-", max_lenghts[s2_idx], ")")
# print("min area:", min_area[s_idx], "\tmax area:", max_area[s_idx])
# print("\n")

# print("c layout constraints")
# print("(", min_lenghts[c1_idx], "-", max_lenghts[c1_idx], "),\t(", min_lenghts[c2_idx], "-", max_lenghts[c2_idx], ")")
# print("min area:", min_area[c_idx], "\tmax area:", max_area[c_idx])
# print("\n")

# print("b. min l:", min_b, "\tmax l:", max_b, "\tmin area:", min_area_b, "\tmax area:", max_area_b)
# print("q. min l:", min_q, "\tmax l:", max_q, "\tmin area:", min_area_q, "\tmax area:", max_area_q)
# print("s. min l:", min_s, "\tmax l:", max_s, "\tmin area:", min_area_s, "\tmax area:", max_area_s)
# print("c. min l:", min_c, "\tmax l:", max_c, "\tmin area:", min_area_c, "\tmax area:", max_area_c)

# sizes_b
# sizes_b_no_area_limit

