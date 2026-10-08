# Shortest Path Town

An interactive Python visualization for exploring how different graph-search
algorithms find routes through a randomly generated town. Select an algorithm,
choose a start and destination intersection, and watch the search animate on a
map of roads with different lengths.

## Requirements

- Python 3.8 or newer
- `matplotlib`
- `numpy`

## Setup

Open a terminal in the folder containing `shortest_path_town.py`:

```bash
cd /path/to/shortestPath
```

Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install matplotlib numpy
```

## Run

From the project folder, run:

```bash
python shortest_path_town.py
```

If the virtual environment is not activated, you can run it directly with:

```bash
.venv/bin/python shortest_path_town.py
```

On Windows, use `.venv\Scripts\python.exe shortest_path_town.py` instead.

## How to use

The window is arranged as a compact header, a left sidebar, and a large map:

- **Header:** Shows the application name, selected algorithm, and current
  instruction or search result.
- **Choose Algorithm card:** Select Dijkstra, A*, Bidirectional Dijkstra,
  Bidirectional A*, Greedy Best-First, or BFS.
- **How to use card:** Shows the endpoint-selection instructions and keyboard
  shortcuts.
- **Map:** Displays intersections, weighted roads, search progress, and a
  legend for the final route, visited and unvisited intersections, and road
  weights.

To find a route:

1. Select an algorithm in the **Choose Algorithm** card.
2. Click an intersection on the map to set the start (green marker).
3. Click a different intersection to set the destination (red marker).
4. Watch the algorithm explore. The live header status reports progress; when
   the search ends, it shows the route length, number of intersections, and
   elapsed search time.
5. Click an intersection after the search finishes to reset the map and begin
   another route.

The result is shortest by total road length for Dijkstra, A*, and both
bidirectional variants. BFS finds a route with the fewest roads, while Greedy
Best-First finds a route without guaranteeing that it is shortest by length.

Keyboard controls:

| Key | Action |
| --- | --- |
| `Space` | Pause or resume the animation |
| `+` or `=` | Speed up the animation |
| `-` or `_` | Slow down the animation |
| `r` | Clear the current selection and reset the search |
| `n` | Generate a new town |
| `q` | Close the application window |

### Map colours and status

| Colour | Meaning |
| --- | --- |
| Pale blue node | Unvisited intersection |
| Bright blue node | Visited intersection |
| Teal node | Discovered intersection in the search frontier |
| Slate-blue road | Road in the town map; its label shows its weight |
| Blue road | Current best-known predecessor tree |
| Cyan road | Road currently being checked |
| Green road | Final route |
| Green marker and label | Start |
| Red marker and label | Destination |

Road labels show lengths in kilometres. The header status changes as the
selection and search progress, then displays the result. The map legend is a
quick key to the final path, visited and unvisited nodes, and road weights.

## Algorithms

The map is a weighted graph: intersections are vertices, roads are edges, and
each road's length is its positive weight. The town is generated with a
7-by-11 grid, small random offsets to intersection positions, occasional
diagonal roads, and some randomly removed roads. Roads are removed only when
the resulting town remains connected.

### Dijkstra

Maintains the best known distance from the start to every discovered
intersection. At each step, it visits the unvisited intersection with the
smallest known distance and updates the distances to its neighbours. Once the
destination is visited, its route is shortest by total road length.

**Guarantee:** Shortest road-length route.

```text
distance[start] = 0
push (0, start) into priority queue

while queue is not empty:
    (cost, current) = pop the smallest-cost entry
    if current is already visited:
        continue
    mark current visited
    if current is destination:
        stop

    for each road current -> neighbor with length:
        new_cost = distance[current] + length
        if new_cost < distance[neighbor]:
            distance[neighbor] = new_cost
            previous[neighbor] = current
            push (new_cost, neighbor) into priority queue

follow previous links from destination to start to build the route
```

### A*

Like Dijkstra, A* tracks the best known distance so far, but prioritizes the
sum of that distance and an estimate of the remaining distance to the
destination. This project uses straight-line distance scaled to the map's road
lengths as its heuristic. That estimate does not overstate the remaining road
distance, so A* retains the shortest-route guarantee while often exploring
fewer intersections.

**Guarantee:** Shortest road-length route.

```text
distance[start] = 0
push (heuristic(start, destination), 0, start) into priority queue

while queue is not empty:
    (priority, cost_so_far, current) = pop the smallest-priority entry
    if current is already visited:
        continue
    mark current visited
    if current is destination:
        stop

    for each road current -> neighbor with length:
        new_cost = cost_so_far + length
        if new_cost < distance[neighbor]:
            distance[neighbor] = new_cost
            previous[neighbor] = current
            priority = new_cost + heuristic(neighbor, destination)
            push (priority, new_cost, neighbor) into priority queue

follow previous links from destination to start to build the route
```

Here, `heuristic(node, destination)` is the straight-line distance estimate
scaled to the map's road lengths.

### Bidirectional Dijkstra

Runs Dijkstra searches from both the start and destination. It joins the two
searches when they meet and stops when no shorter connection can be found.
Searching from both ends can reduce the explored area on suitable maps.

**Guarantee:** Shortest road-length route.

```text
forward_distance[start] = 0
backward_distance[destination] = 0
push start into forward priority queue
push destination into backward priority queue
best_route = infinity

while both queues are not empty:
    if smallest unsettled forward distance
       + smallest unsettled backward distance >= best_route:
        stop

    expand the queue with the smaller minimum distance
    for each road current -> neighbor with length:
        update this direction's distance and predecessor if improved
        if neighbor is known from both directions:
            best_route = min(best_route,
                             forward_distance[neighbor]
                             + backward_distance[neighbor])

join the forward and backward predecessor chains at their best meeting point
```

### Bidirectional A*

Searches from both ends like Bidirectional Dijkstra, while using a
straight-line distance estimate to prioritize each side's frontier.

**Guarantee:** Shortest road-length route.

```text
initialize forward search at start and backward search at destination
best_route = infinity

while both priority queues are not empty:
    if smallest unsettled forward distance
       + smallest unsettled backward distance >= best_route:
        stop

    compute each queue priority as:
        distance_so_far + straight_line_estimate_to_its_search_target
    expand the queue with the smaller priority
    relax its neighbouring roads and update predecessors
    when both searches know a node:
        update best_route with the combined route through that node

join the forward and backward predecessor chains at their best meeting point
```

### Greedy Best-First

Prioritizes intersections that appear closest to the destination, using
straight-line distance. It does not use the full route-so-far distance to
choose which intersection to explore, so it can be drawn toward a promising
but inefficient route.

**Guarantee:** Finds a route in this connected town, but not necessarily the
shortest one.

```text
push start into priority queue with priority
    straight_line_distance(start, destination)

while queue is not empty:
    current = pop the entry with the smallest straight-line estimate
    if current is already visited:
        continue
    mark current visited
    if current is destination:
        stop

    for each unvisited neighbor of current:
        if the route through current improves its recorded route length:
            record the route length and previous[neighbor] = current
            push neighbor with priority
                straight_line_distance(neighbor, destination)

follow previous links from destination to start to build the route
```

### Breadth-First Search (BFS)

Explores the graph level by level, first visiting intersections one road away,
then two roads away, and so on. It ignores the road lengths when choosing its
route.

**Guarantee:** Route with the fewest roads. It is shortest by road length only
when every road has the same length.

```text
enqueue start
mark start discovered

while queue is not empty:
    current = dequeue the oldest entry
    if current is destination:
        stop

    for each undiscovered neighbor of current:
        mark neighbor discovered
        previous[neighbor] = current
        enqueue neighbor

follow previous links from destination to start to build the route
```

## Comparison

| Algorithm | What it optimizes | Uses road lengths to choose route? | Shortest road-length guarantee? | Typical search behaviour |
| --- | --- | --- | --- | --- |
| Dijkstra | Total road length | Yes | Yes | Explores outward in order of distance from the start |
| A* | Total road length, guided by estimated remaining distance | Yes | Yes | Often explores more directly toward the destination |
| Bidirectional Dijkstra | Total road length | Yes | Yes | Searches out from both endpoints |
| Bidirectional A* | Total road length, guided from both endpoints | Yes | Yes | Directs two frontiers toward one another |
| Greedy Best-First | Apparent closeness to destination | No; length is only accumulated for the reported route | No | Often heads toward the goal, but may take a detour |
| BFS | Number of roads | No | No, unless road lengths are equal | Explores evenly by number of roads from the start |

The displayed **elapsed search time** measures the time spent advancing the
algorithm's search generator. It excludes animation delays, pauses, and drawing
time, so it compares search computation rather than how long the visualization
takes to play.
