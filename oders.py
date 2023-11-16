import tkinter as tk
from itertools import permutations
import matplotlib.pyplot as plt
import heapq
from numpy import floor
# Creating the main window
winodw_size = 800
n = 5
block_size = 120
gap = 25
canva_size = n*block_size+2*gap
root = tk.Tk()
root.title("Orders manager")
root.geometry(f"{winodw_size}x{winodw_size}")

# Points of interest describing the order
order_info = {'delivery': []}
point_id = []  # id of different points drawn
current_orientation = 'forward'
current_ang = 90
commands = []

# Creating a canvas
canvas = tk.Canvas(root, width=canva_size, height=canva_size, bg="white")
canvas.pack()

# Drawing coordinates
for i in range(gap, canva_size-gap+block_size, block_size):
    canvas.create_line(i, gap, i, canva_size-gap,
                       fill="black")  # Vertical lines
    canvas.create_text(i, canva_size-gap/2, text=str((i-gap)//block_size),
                       fill="black")  # X-axis numbers
for j in range(gap, canva_size+block_size-gap, block_size):
    canvas.create_line(gap, j, canva_size-gap, j,
                       fill="black")  # Horizontal lines
    canvas.create_text(gap/2, j, text=str((canva_size-j-gap)//block_size),
                       fill="black")  # Y-axis numbers

# Stupid approximations


def true_val(x):
    return round(x)


# Defining mechanisms


def putting_order(event):
    x, y = event.x, event.y
    order_info['delivery'].append(
        (true_val((x-gap)/block_size), true_val((canva_size-y-gap)/block_size)))
    point_id.append(canvas.create_oval(x-5, y-5, x + 5, y + 5, fill="red"))


def putting_robot(event):
    if "start" not in order_info:
        x, y = event.x, event.y
        order_info["start"] = (
            (true_val((x-gap)/block_size), true_val((canva_size-y-gap)/block_size)))
        canvas.create_oval(x-5, y-5, x + 5, y + 5, fill="green")
    else:
        pass


def place_button():
    canvas.unbind('<1>')
    canvas.bind('<1>', putting_order)


def initial_position():
    canvas.unbind('<1>')
    canvas.bind('<1>', putting_robot)


def undo():
    if len(order_info["delivery"]) != 0:
        canvas.unbind('<1>')
        order_info["delivery"].pop()
        c = point_id.pop()
        canvas.delete(c)


def up():
    global current_orientation, current_ang
    current_orientation = 'forward'
    current_ang = 90
    if 'start' in order_info:
        x0 = order_info['start'][0]*block_size+gap
        y0 = canva_size-order_info['start'][1]*block_size-gap
        canvas.create_line(x0, y0, x0, y0-block_size/5,
                           arrow=tk.LAST, tags="arrow")


def down():
    global current_orientation, current_ang
    current_orientation = 'back'
    current_ang = 270
    if 'start' in order_info:
        x0 = order_info['start'][0]*block_size+gap
        y0 = canva_size-order_info['start'][1]*block_size-gap
        canvas.create_line(x0, y0, x0, y0+block_size/5,
                           arrow=tk.LAST, tags="arrow")


def left():
    global current_orientation, current_ang
    current_orientation = 'left'
    current_ang = 180
    if 'start' in order_info:
        x0 = order_info['start'][0]*block_size+gap
        y0 = canva_size-order_info['start'][1]*block_size-gap
        canvas.create_line(x0, y0, x0-block_size/5, y0,
                           arrow=tk.LAST, tags="arrow")


def right():
    global current_orientation, current_ang
    current_orientation = 'right'
    current_ang = 0
    if 'start' in order_info:
        x0 = order_info['start'][0]*block_size+gap
        y0 = canva_size-order_info['start'][1]*block_size-gap
        canvas.create_line(x0, y0, x0+block_size/5, y0,
                           arrow=tk.LAST, tags="arrow")


def finished():
    root.quit()


up_button = tk.Button(root, text="Up", command=up)
up_button.place(x=winodw_size-220, y=-25+winodw_size-100)
down_button = tk.Button(root, text="Down", command=down)
down_button.place(x=winodw_size-220, y=-25+winodw_size-60)

left_button = tk.Button(root, text="Left", command=left)
left_button.place(x=winodw_size-270, y=-25+winodw_size-90)

right_button = tk.Button(root, text="Right", command=right)
right_button.place(x=winodw_size-170, y=-25+winodw_size-90)
place_order = tk.Button(root, text="Place order", command=place_button)
place_order.pack()
initial_position_button = tk.Button(
    root, text="Place starting point", command=initial_position)
initial_position_button.pack()
undo_button = tk.Button(root, text="Undo", command=undo)
undo_button.pack()
finished_button = tk.Button(root, text="Order", command=finished)
finished_button.pack()
root.mainloop()


# Calculating the optimal path

def is_cell_available(cell, unavailable_cells):
    """Check if a cell is available."""
    return cell not in unavailable_cells


def manhattan_distance(p1, p2):
    """Calculate the Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def orientation_unit(degree):
    if degree == 0:
        return 'right'
    elif degree == 90:
        return 'forward'
    elif degree == 180:
        return 'left'
    elif degree == 270:
        return 'back'


def direction_diff(next):
    global current_orientation, current_ang
    if current_orientation != next:
        if current_orientation == 'forward' and next == 'right' or \
                current_orientation == 'back' and next == 'left' or \
                current_orientation == 'left' and next == 'forward' or \
                current_orientation == 'right' and next == 'back':
            current_ang = (current_ang-90) % 360
            current_orientation = orientation_unit(current_ang)
            return 'right'
        elif current_orientation == 'forward' and next == 'left' or \
                current_orientation == 'back' and next == 'right' or \
                current_orientation == 'left' and next == 'back' or \
                current_orientation == 'right' and next == 'forward':
            current_ang = (current_ang+90) % 360
            current_orientation = orientation_unit(current_ang)
            return 'left'
        else:
            current_ang = (current_ang+180) % 360
            current_orientation = orientation_unit(current_ang)
            return 'u_turn'
    else:
        return 'forward'


def get_direction(from_point, to_point):
    """Determine the direction from one point to another."""
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    if dx == 1:
        return direction_diff('right')
    elif dx == -1:
        return direction_diff('left')
    elif dy == 1:
        return direction_diff('forward')
    elif dy == -1:
        return direction_diff('back')


def a_star_search(start, goal, unavailable_cells):
    """Perform A* search to find the shortest path avoiding unavailable cells."""
    open_set = set()
    open_set.add(start)
    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}

    while open_heap:
        current = heapq.heappop(open_heap)[1]
        open_set.remove(current)

        if current == goal:
            # Reconstruct path
            total_path = [current]
            while current in came_from:
                current = came_from[current]
                total_path.append(current)
            return total_path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Neighbors
            neighbor = (current[0] + dx, current[1] + dy)
            tentative_g_score = g_score[current] + 1
            if 0 <= neighbor[0] and 0 <= neighbor[1] and is_cell_available(neighbor, unavailable_cells) and (neighbor not in g_score or tentative_g_score < g_score[neighbor]):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + \
                    manhattan_distance(neighbor, goal)
                if neighbor not in open_set:
                    open_set.add(neighbor)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    return None


def brute_force_tsp_manhattan_dynamic(start, points, unavailable_cells):
    """Find the shortest path in a dynamic environment using brute force with A* search."""
    global commands
    shortest_path = []
    current_position = start
    points = points.copy()

    for point in points:
        while True:
            path_segment = a_star_search(
                current_position, point, unavailable_cells)
            if path_segment is None:
                print("No path found to point", point, "due to obstacles.")
                break

            segment_directions = [get_direction(
                path_segment[i], path_segment[i+1]) for i in range(len(path_segment)-1)]
            commands += segment_directions
            # print("Directions to", point, ":", segment_directions)

            if any(cell in unavailable_cells for cell in path_segment):
                print("Path to", point, "affected by new obstacle. Recalculating...")
                continue

            shortest_path.extend(path_segment[1:])
            current_position = point
            break

    return_to_start_path = a_star_search(
        current_position, start, unavailable_cells)
    if return_to_start_path is not None:
        shortest_path.extend(return_to_start_path[1:])
    else:
        print("No path found to return to the start point", start)

    return shortest_path


shortest_path = brute_force_tsp_manhattan_dynamic(
    order_info["start"], order_info["delivery"], [])
print(commands)

# Defining the commands
# if current_orientation != next:
# if current_orientation == 'forward' and next == 'right' or \
# current_orientation == 'back' and next == 'left' or \
# current_orientation == 'left' and next == 'forward' or \
# current_orientation == 'right' and next == 'back':
# orientation_commands.append(90)
# elif current_orientation == 'forward' and next == 'left' or \
# current_orientation == 'back' and next == 'right' or \
# current_orientation == 'left' and next == 'back' or \
# current_orientation == 'right' and next == 'forward':
# orientation_commands.append(-90)
# else:
# orientation_commands.append(180)
