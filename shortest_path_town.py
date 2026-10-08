"""Interactive visualization of graph-search algorithms on a generated town.

The town is represented as a weighted, undirected graph. Intersections are
nodes, roads are edges, and each edge weight is that road's length in
kilometres. The application animates six searches: Dijkstra, A*, bidirectional
Dijkstra, bidirectional A*, Greedy Best-First, and Breadth-First Search (BFS).

Requirements:
    Install the third-party packages with
    ``python -m pip install matplotlib numpy``.

Run:
    ``python shortest_path_town.py``

Use:
    1. Choose an algorithm from the radio-button selector.
    2. Click an intersection to select the green start point.
    3. Click another intersection to select the red destination.
    4. Watch the algorithm inspect roads and display its resulting route.
       Dijkstra and A* variants minimize total road length. BFS minimizes the
       number of roads, and Greedy Best-First is not guaranteed to be optimal.

Keyboard controls:
    Space       Pause or resume the animation.
    + or =      Make animation steps faster.
    - or _      Make animation steps slower.
    r           Clear the current selection and search.
    n           Generate a different random town.
    q           Close the application window.

Map colours:
    Pale blue node Not yet visited.
    Bright blue node Visited.
    Dark slate road Unexplored road.
    Blue road Current predecessor tree.
    Green road Final route; green marker is start, red marker is destination.
"""
import heapq
import math
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.widgets import RadioButtons


# Navy and blue colors based on the application's dark route-lab interface.
THEME = {
    "background": "#071426",
    "header": "#0D1D34",
    "panel": "#0D1B30",
    "map": "#08172A",
    "border": "#193557",
    "text": "#E6F0FF",
    "muted": "#9BB5D8",
    "road": "#718CB6",
    "road_label": "#18345A",
    "node": "#DCEBFF",
    "node_edge": "#65A5FF",
    "frontier": "#10D6A2",
    "visited": "#168BFF",
    "tree": "#138BFF",
    "current": "#16C7F4",
    "path": "#00D6A0",
    "start": "#00D6A0",
    "end": "#F0445E",
    "accent": "#2788FF",
}


# --------------------------------------------------------------------------
# Town generation
# --------------------------------------------------------------------------
def is_connected(adj):
    """Return whether every graph node can be reached from the first node.

    Args:
        adj: Adjacency mapping where ``adj[node]`` contains that node's
            neighbors. Edge values (road lengths) are not needed for checking
            connectivity.

    The depth-first traversal uses ``set`` for fast membership checks and a
    Python ``list`` as a stack. ``list(adj)`` obtains the node keys so the
    first node can be used as the traversal seed. The graph generated here is
    non-empty; for an empty mapping, ``nodes[0]`` would raise IndexError.
    """
    # dict iteration yields keys. Keep them in a list so index zero selects
    # one valid node as the start of the reachability traversal.
    nodes = list(adj)
    # A set records which nodes have already been discovered, preventing
    # repeated visits when multiple roads lead to the same node.
    seen = {nodes[0]}
    # The list is used as a last-in/first-out stack for depth-first search.
    stack = [nodes[0]]
    while stack:
        # list.pop() with no index removes and returns the last stack item.
        u = stack.pop()
        # Iterating over adj[u] visits the keys of this node's neighbor map.
        for v in adj[u]:
            if v not in seen:
                # Mark before adding to avoid scheduling the same node twice.
                seen.add(v)
                stack.append(v)
    # len() counts discovered nodes and graph nodes; equality means all nodes
    # were reached from the seed and therefore the graph is connected.
    return len(seen) == len(nodes)


def build_town(rows=7, cols=11, seed=None):
    """Generate a randomly perturbed grid town and its weighted road graph.

    Args:
        rows: Number of intersection rows in the grid.
        cols: Number of intersection columns in the grid.
        seed: Optional random seed. Supplying the same seed and dimensions
            makes the generated town reproducible.

    Returns:
        A pair ``(pos, adj)``. ``pos`` maps each integer node ID to an (x, y)
        coordinate. ``adj`` maps each node ID to a dictionary of neighboring
        node IDs and road lengths in kilometres. Every road is stored in both
        directions because the generated roads are undirected.

    ``random.Random(seed)`` creates a local pseudo-random generator, so this
    function does not alter the application's or caller's global random state.
    """
    # Use a local random-number generator so callers can control repeatability
    # without changing random values used elsewhere in the process.
    rng = random.Random(seed)
    pos = {}
    # Assign each grid location a unique integer ID and perturb its
    # coordinates slightly so the plotted town does not look perfectly rigid.
    for r in range(rows):
        for c in range(cols):
            pos[r * cols + c] = (c + rng.uniform(-0.3, 0.3),
                                 r + rng.uniform(-0.3, 0.3))

    edges = []
    # Connect each grid point to its right and lower neighbor when those
    # neighbors exist. Add occasional diagonals to create alternative routes.
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                edges.append((i, i + 1))
            if r + 1 < rows:
                edges.append((i, i + cols))
            if c + 1 < cols and r + 1 < rows and rng.random() < 0.15:
                edges.append((i, i + cols + 1))
            if c > 0 and r + 1 < rows and rng.random() < 0.15:
                edges.append((i, i + cols - 1))

    # A dictionary-of-dictionaries is an adjacency list: looking up adj[a][b]
    # gives the road length from node a to neighboring node b.
    adj = {i: {} for i in pos}
    for a, b in edges:
        # math.dist() calculates Euclidean distance between the two (x, y)
        # coordinates. Scaling by 0.5 defines the road-length units used here.
        w = math.dist(pos[a], pos[b]) * 0.5
        # Store both directions because each road can be traversed either way.
        adj[a][b] = w
        adj[b][a] = w

    # Shuffle the candidate roads so the blocked streets vary with the seed.
    rng.shuffle(edges)
    removed = 0
    for a, b in edges:
        # Stop after removing approximately 22% of all candidate roads.
        if removed >= len(edges) * 0.22:
            break
        # dict.pop(key) removes a road and returns its weight so it can be
        # restored if removing it would disconnect the town.
        w = adj[a].pop(b)
        adj[b].pop(a)
        # Keep the removal only if every intersection remains reachable.
        if is_connected(adj):
            removed += 1
        else:
            # The removal would create an isolated region, so restore both
            # directional entries with their original shared weight.
            adj[a][b] = w
            adj[b][a] = w
    return pos, adj


# --------------------------------------------------------------------------
# Search algorithms yield events so each can be animated step by step.
# --------------------------------------------------------------------------
# Event protocol used by every search generator:
#   ("start", start_node)
#   ("visit", node, accumulated_distance)
#   ("relax", from_node, to_node, candidate_distance, improved[, backward])
#   ("done", ordered_path, total_distance)
# The UI consumes this common format, so each search can share one animator.
def dijkstra_steps(adj, start, end):
    """Yield animation events while finding a shortest weighted route.

    Args:
        adj: Weighted adjacency mapping of the town.
        start: Node ID at which to begin the search.
        end: Destination node ID.

    Yields tuples consumed by :meth:`ShortestPathApp.tick`: ``start``, ``visit``,
    ``relax``, and finally ``done`` with the path and its total length.

    Dijkstra's algorithm is correct for non-negative road lengths. The
    priority queue may contain outdated entries; ``visited`` ensures each
    node is finalized only once.
    """
    # Initially, no node has a known route; math.inf is larger than every
    # finite route length. The start node is already reachable at cost zero.
    dist = {n: math.inf for n in adj}
    dist[start] = 0.0
    # prev records the node immediately before each reached node on the
    # currently best-known path, allowing the final route to be reconstructed.
    prev = {}
    visited = set()
    # heapq implements a min-priority queue using a normal Python list.
    pq = [(0.0, start)]
    yield ("start", start)

    while pq:
        # heappop() removes the entry with the smallest tentative distance.
        d, u = heapq.heappop(pq)
        if u in visited:
            # Ignore duplicate queue entries for nodes already finalized.
            continue
        visited.add(u)
        yield ("visit", u, d)
        if u == end:
            # With non-negative weights, the first finalized destination has
            # the minimum possible path length.
            break
        # dict.items() supplies each neighboring node together with its edge
        # weight, so the candidate route can be calculated.
        for v, w in adj[u].items():
            if v in visited:
                continue
            nd = d + w
            # Relaxation asks whether the route through u improves the
            # previously recorded route to v.
            improved = nd < dist[v]
            yield ("relax", u, v, nd, improved)
            if improved:
                dist[v] = nd
                prev[v] = u
                # heappush() adds a new tentative route to the min-heap.
                heapq.heappush(pq, (nd, v))

    # Follow predecessor links backward from the destination. The search is
    # expected to operate on a connected generated town, so every link exists.
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    # reverse() changes the list in place from end-to-start into start-to-end.
    path.reverse()
    yield ("done", path, dist[end])


def a_star_steps(adj, pos, start, end):
    """Yield animation events while finding a shortest route with A*.

    Args:
        adj: Weighted adjacency mapping of the town.
        pos: Mapping from node IDs to their (x, y) coordinates.
        start: Node ID at which to begin the search.
        end: Destination node ID.

    Yields the same event shapes as :func:`dijkstra_steps`. A* orders the
    priority queue by ``distance_so_far + heuristic_estimate``. The heuristic
    is straight-line distance scaled by 0.5, matching the generated road
    weights and providing an admissible estimate for this graph.
    """
    # dist stores the best known actual road length from start, independent
    # of the heuristic priority used to decide which node to explore next.
    dist = {n: math.inf for n in adj}
    dist[start] = 0.0
    prev = {}
    visited = set()

    def heuristic(node):
        """Estimate remaining road length using scaled straight-line distance.

        math.dist() computes the Euclidean distance between the two positions.
        Since each generated road weight is half its geometric length, scaling
        the direct geometric distance by the same factor does not overestimate
        the route length.
        """
        return math.dist(pos[node], pos[end]) * 0.5

    # Each heap entry is (estimated total cost, actual cost so far, node ID).
    pq = [(heuristic(start), 0.0, start)]
    yield ("start", start)

    while pq:
        # Pop the node with the lowest estimated complete-route cost.
        _, d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        yield ("visit", u, d)
        if u == end:
            break
        for v, w in adj[u].items():
            if v in visited:
                continue
            nd = d + w
            improved = nd < dist[v]
            yield ("relax", u, v, nd, improved)
            if improved:
                dist[v] = nd
                prev[v] = u
                # Add estimated remaining distance to the known cost. The
                # actual cost is retained separately for accurate updates.
                heapq.heappush(pq, (nd + heuristic(v), nd, v))

    # Reconstruct and order the route using the predecessor map.
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    yield ("done", path, dist[end])


def bidirectional_dijkstra_steps(adj, start, end):
    """Yield events for a bidirectional Dijkstra shortest-path search.

    This public algorithm wrapper delegates to :func:`bidirectional_steps`
    without positions, which makes the internal heuristic zero. The shared
    implementation then expands the frontier with the lower priority.
    """
    yield from bidirectional_steps(adj, start, end)


def bidirectional_a_star_steps(adj, pos, start, end):
    """Yield events for bidirectional search prioritized with A* estimates.

    Args:
        adj: Weighted adjacency mapping of the town.
        pos: Mapping from node IDs to their (x, y) coordinates.
        start: Start node ID.
        end: Destination node ID.

    ``yield from`` forwards each event from the shared bidirectional search
    generator to the animation caller.
    """
    yield from bidirectional_steps(adj, start, end, pos=pos)


def bidirectional_steps(adj, start, end, pos=None):
    """Search outward from both endpoints and yield visualization events.

    Args:
        adj: Weighted adjacency mapping of the town.
        start: Start node ID.
        end: Destination node ID.
        pos: Optional node-coordinate mapping. If supplied, straight-line
            estimates prioritize each frontier (the bidirectional A* mode).
            If omitted, both heuristic values are zero (bidirectional
            Dijkstra mode).

    The two predecessor mappings point in opposite directions: ``forward_prev``
    traces from a meeting node back toward start; ``backward_next`` traces
    from the meeting node toward end. ``best_distance`` records the shortest
    complete connection found so far. A ``relax`` event includes an extra
    boolean identifying backward-search edges so the display can orient the
    predecessor-tree line correctly.
    """
    # Keep independent distances, predecessor links, visited nodes, and
    # priority queues for the searches from the start and destination.
    forward_dist = {start: 0.0}
    backward_dist = {end: 0.0}
    forward_prev = {}
    backward_next = {}
    forward_visited = set()
    backward_visited = set()
    forward_pq = [(0.0, 0.0, start)]
    backward_pq = [(0.0, 0.0, end)]
    best_distance = math.inf
    meeting = None

    def heuristic(node, target):
        """Return a non-overestimating estimate to target, or zero for Dijkstra.

        Because a straight segment is no longer than a road route, scaled
        Euclidean distance is a lower bound on remaining road length.
        """
        if pos is None:
            return 0.0
        return math.dist(pos[node], pos[target]) * 0.5

    def min_open_distance(distances, visited):
        """Return the smallest tentative path length not yet finalized.

        The generator expression filters out visited nodes. The ``default``
        argument to min() provides infinity when no unvisited entries remain,
        which lets the termination condition handle an exhausted frontier.
        """
        return min(
            (distance for node, distance in distances.items()
             if node not in visited),
            default=math.inf,
        )

    def update_best(node):
        """Update the best complete route if both searches reached node.

        ``nonlocal`` allows this nested function to update the surrounding
        search's best_distance and meeting variables.
        """
        nonlocal best_distance, meeting
        if node in forward_dist and node in backward_dist:
            candidate = forward_dist[node] + backward_dist[node]
            if candidate < best_distance:
                best_distance = candidate
                meeting = node

    yield ("start", start)

    while forward_pq and backward_pq:
        # Once the two lowest remaining actual distances cannot beat the
        # complete route already found, no shorter connection can remain.
        if (best_distance < math.inf
                and min_open_distance(forward_dist, forward_visited)
                + min_open_distance(backward_dist, backward_visited)
                >= best_distance):
            break

        # Heap index zero is the smallest-priority item. Expand the side whose
        # next item has the lower priority estimate.
        forward_priority = forward_pq[0][0]
        backward_priority = backward_pq[0][0]
        is_forward = forward_priority <= backward_priority
        pq = forward_pq if is_forward else backward_pq
        distances = forward_dist if is_forward else backward_dist
        visited = forward_visited if is_forward else backward_visited
        parents = forward_prev if is_forward else backward_next
        target = end if is_forward else start

        # Discard stale heap entries whose distance was superseded by a better
        # relaxation while this item waited in the priority queue.
        _, distance, node = heapq.heappop(pq)
        if node in visited or distance != distances[node]:
            continue
        visited.add(node)
        yield ("visit", node, distance)
        update_best(node)

        for neighbor, weight in adj[node].items():
            if neighbor not in visited:
                candidate_distance = distance + weight
                # get(neighbor, math.inf) handles nodes not yet discovered by
                # this side of the bidirectional search.
                improved = candidate_distance < distances.get(
                    neighbor, math.inf
                )
                yield (
                    "relax", node, neighbor, candidate_distance,
                    improved, not is_forward,
                )
                if improved:
                    distances[neighbor] = candidate_distance
                    parents[neighbor] = node
                    candidate_priority = (
                        candidate_distance + heuristic(neighbor, target)
                    )
                    # Queue both the priority used for ordering and the true
                    # distance used for reporting and path-cost comparisons.
                    heapq.heappush(
                        pq, (candidate_priority, candidate_distance, neighbor)
                    )
            # A route may improve the best connection even if this particular
            # neighbor was already known to the other side.
            update_best(neighbor)

    if meeting is None:
        # Make an impossible route failure explicit rather than returning a
        # misleading empty or partially reconstructed path.
        raise RuntimeError("No route exists between the selected intersections.")

    # Reconstruct the start-to-meeting half by walking predecessors backward.
    path = [meeting]
    while path[-1] != start:
        path.append(forward_prev[path[-1]])
    path.reverse()
    # The backward map already points from the meeting node toward end, so
    # append nodes in that order without reversing the second half.
    node = meeting
    while node != end:
        node = backward_next[node]
        path.append(node)
    yield ("done", path, best_distance)


def breadth_first_steps(adj, start, end):
    """Yield events for a breadth-first search from start to end.

    Args:
        adj: Weighted adjacency mapping of the town. BFS uses neighbor
            connectivity, not road weights, to choose its route.
        start: Start node ID.
        end: Destination node ID.

    Yields animation events matching the other algorithms. BFS finds a route
    with the fewest edges (roads); because roads can have different weights,
    that route is not necessarily the shortest by total road length. The
    cumulative weight is tracked only so the visualizer can show a distance.
    """
    # deque is a double-ended queue. append() adds discovered nodes to the
    # back, and popleft() removes them from the front in FIFO order.
    queue = deque([start])
    prev = {}
    # This total is for displaying the discovered BFS route's length, not for
    # deciding which node should be visited next.
    distances = {start: 0.0}
    # Mark nodes when enqueued so each node is scheduled only once.
    discovered = {start}
    yield ("start", start)

    while queue:
        # FIFO order visits nodes by number of roads from the start.
        node = queue.popleft()
        yield ("visit", node, distances[node])
        if node == end:
            break
        for neighbor, weight in adj[node].items():
            # A neighbor is new only when BFS has not already enqueued it.
            improved = neighbor not in discovered
            candidate_distance = distances[node] + weight
            yield ("relax", node, neighbor, candidate_distance, improved)
            if improved:
                discovered.add(neighbor)
                distances[neighbor] = candidate_distance
                prev[neighbor] = node
                queue.append(neighbor)

    # Walk predecessor links back to start, then reverse the list so the
    # animation receives the route in travel order.
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    yield ("done", path, distances[end])


def greedy_best_first_steps(adj, pos, start, end):
    """Yield events for Greedy Best-First search guided by goal proximity.

    Args:
        adj: Weighted adjacency mapping of the town.
        pos: Mapping from node IDs to (x, y) coordinates.
        start: Start node ID.
        end: Destination node ID.

    The heap is ordered using only straight-line distance to the destination;
    accumulated road length is retained for reporting and relaxation but does
    not contribute to the priority. This greedy strategy can therefore find a
    route quickly, but does not guarantee a shortest route.
    """
    # Each queue item is (estimated distance to goal, distance from start,
    # node). heapq uses the tuple's first field as its primary sort key.
    queue = [(math.dist(pos[start], pos[end]), 0.0, start)]
    distances = {start: 0.0}
    prev = {}
    visited = set()
    yield ("start", start)

    while queue:
        # heappop() chooses the node whose geometric distance to end is least.
        _, distance, node = heapq.heappop(queue)
        # Ignore entries already finalized or made obsolete by a better path.
        if node in visited or distance != distances[node]:
            continue
        visited.add(node)
        yield ("visit", node, distance)
        if node == end:
            break
        for neighbor, weight in adj[node].items():
            if neighbor in visited:
                continue
            candidate_distance = distance + weight
            improved = candidate_distance < distances.get(
                neighbor, math.inf
            )
            yield ("relax", node, neighbor, candidate_distance, improved)
            if improved:
                distances[neighbor] = candidate_distance
                prev[neighbor] = node
                # The priority is solely the remaining straight-line distance;
                # candidate_distance is kept as a secondary tuple value.
                estimate = math.dist(pos[neighbor], pos[end])
                heapq.heappush(
                    queue, (estimate, candidate_distance, neighbor)
                )

    # Reconstruct the route from destination back to start, then restore the
    # forward travel order for the final animation event.
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    yield ("done", path, distances[end])


# --------------------------------------------------------------------------
# Interactive app
# --------------------------------------------------------------------------
class ShortestPathApp:
    """Own the Matplotlib window, generated graph, and animation state.

    The class connects mouse and keyboard callbacks to graph-search generators.
    Search functions yield one event at a time; a Matplotlib timer requests
    those events and updates the drawing without blocking the interactive UI.
    """

    def __init__(self):
        """Create the figure, controls, event handlers, timer, and first town.

        ``plt.subplots`` returns a figure and axes. ``add_axes`` creates the
        separate region for algorithm radio buttons. Matplotlib's callback
        registration methods arrange for instance methods to be called on
        relevant UI events.
        """
        # Create the main figure and map axes, reserving a header, sidebar,
        # and wide central area for the road network.
        self.algorithm = "Dijkstra"
        self.fig, self.ax = plt.subplots(figsize=(13, 7.75))
        self.fig.patch.set_facecolor(THEME["background"])
        self.fig.subplots_adjust(left=0.24, right=0.98, top=0.84, bottom=0.06)
        self.ax.set_position([0.24, 0.06, 0.74, 0.78])

        # Build a compact header with a location glyph, app name, subtitle,
        # and a live algorithm/status readout aligned on the right.
        header_ax = self.fig.add_axes([0.0, 0.84, 1.0, 0.16], zorder=0)
        header_ax.set_facecolor(THEME["header"])
        header_ax.set_xticks([])
        header_ax.set_yticks([])
        header_ax.scatter(
            [0.04], [0.52], s=520, marker="s", color=THEME["accent"], zorder=1,
        )
        header_ax.scatter(
            [0.04], [0.54], s=115, marker="o", color=THEME["text"], zorder=2,
        )
        header_ax.scatter(
            [0.04], [0.54], s=32, marker="o", color=THEME["accent"], zorder=3,
        )
        header_ax.set_xlim(0, 1)
        header_ax.set_ylim(0, 1)
        for spine in header_ax.spines.values():
            spine.set_visible(False)
        manager = plt.get_current_fig_manager()
        if hasattr(manager, "set_window_title"):
            manager.set_window_title("Shortest Path / Town")
        self.fig.text(
            0.075, 0.925, "Shortest Path / Town", fontsize=18,
            fontweight="bold", color=THEME["text"], va="center",
        )
        self.fig.text(
            0.076, 0.885, "Interactive Route-Finding Lab", fontsize=9,
            color=THEME["muted"], va="center",
        )
        self.header_algorithm = self.fig.text(
            0.72, 0.925, self.algorithm,
            fontsize=9, fontweight="bold", color=THEME["accent"],
            ha="left", va="center",
        )
        self.header_status = self.fig.text(
            0.72, 0.885, "Choose a start intersection",
            fontsize=8, color=THEME["muted"], ha="left", va="center",
        )
        self.fig.add_artist(plt.Line2D(
            [0.0, 1.0], [0.84, 0.84], transform=self.fig.transFigure,
            color=THEME["border"], linewidth=1.0,
        ))

        # Frame the map as its own panel, then use a dark interior for the
        # network so its roads and active route have clear contrast.
        self.ax.set_facecolor(THEME["map"])
        self.ax.patch.set_edgecolor(THEME["border"])
        self.ax.patch.set_linewidth(1.2)

        # The left column contains independent algorithm and help cards.
        algorithm_ax = self.fig.add_axes([0.02, 0.46, 0.21, 0.35])
        algorithm_ax.set_facecolor(THEME["panel"])
        algorithm_ax.patch.set_edgecolor(THEME["border"])
        algorithm_ax.patch.set_linewidth(1.0)
        self.selection_highlight = Rectangle(
            (0.02, 0.80), 0.96, 0.115,
            transform=algorithm_ax.transAxes,
            facecolor="#173B78",
            edgecolor="none",
            zorder=1.5,
        )
        algorithm_ax.add_patch(self.selection_highlight)
        algorithm_ax.text(
            0.25,
            0.95,
            "Choose Algorithm",
            transform=algorithm_ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color=THEME["text"],
            ha="left",
            va="center",
        )
        self.algorithm_radio = RadioButtons(
            algorithm_ax,
            (
                "Dijkstra",
                "A*",
                "Bidirectional Dijkstra",
                "Bidirectional A*",
                "Greedy Best-First",
                "BFS",
            ),
            active=0,
            activecolor=THEME["accent"],
        )
        # RadioButtons creates one Text label for each option; recolor each
        # label so it remains readable against the dark widget panel. Move
        # the full list down slightly to leave a clearer gap below its heading.
        radio_rows = np.linspace(0.78, 0.12, len(self.algorithm_radio.labels))
        for index, (label, row_y) in enumerate(
            zip(self.algorithm_radio.labels, radio_rows)
        ):
            label.set_position((0.25, row_y))
            label.set_color(
                THEME["accent"] if index == 0 else THEME["text"]
            )
            label.set_fontsize(9)
            if index == 0:
                label.set_fontweight("bold")
        radio_offsets = self.algorithm_radio.ax.collections[0].get_offsets()
        radio_offsets[:, 1] = radio_rows
        self.algorithm_radio.ax.collections[0].set_offsets(radio_offsets)
        self.selection_highlight.set_y(radio_rows[0] - 0.06)
        # on_clicked() registers the callback that updates the chosen mode.
        self.algorithm_radio.on_clicked(self.on_algorithm_change)
        # The timer advances one generator event per configured interval.
        self.interval = 150  # ms per step
        self.paused = False
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self.tick)
        # mpl_connect() registers handlers and returns connection IDs (not
        # needed here because handlers remain active for the app lifetime).
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        help_ax = self.fig.add_axes([0.02, 0.06, 0.21, 0.35])
        help_ax.set_facecolor(THEME["panel"])
        help_ax.patch.set_edgecolor(THEME["border"])
        help_ax.patch.set_linewidth(1.0)
        help_ax.set_xticks([])
        help_ax.set_yticks([])
        for spine in help_ax.spines.values():
            spine.set_visible(False)
        help_ax.text(
            0.08, 0.9, "How to use", transform=help_ax.transAxes,
            fontsize=10, fontweight="bold", color=THEME["text"], va="center",
        )
        help_ax.text(
            0.08, 0.73, "Click two intersections\nto find a route.",
            transform=help_ax.transAxes, fontsize=9, color=THEME["text"],
            linespacing=1.45, va="top",
        )
        help_ax.text(
            0.08, 0.49,
            "SPACE   Pause / resume\n"
            "+ / -      Adjust speed\n"
            "R            Reset search\n"
            "N            New town\n"
            "Q            Quit",
            transform=help_ax.transAxes, fontsize=8.5,
            color=THEME["muted"], linespacing=1.65, family="monospace",
            va="top",
        )
        self.new_town()

    # ---- setup / drawing --------------------------------------------------
    def new_town(self):
        """Generate a fresh random graph and reset the display around it."""
        # build_town() returns the coordinates and weighted adjacency mapping.
        self.pos, self.adj = build_town()
        self.reset()

    def reset(self):
        """Stop any search and redraw a clean map for the current town.

        Reset clears node selections, generator state, animation sets, and
        elapsed search timing. It leaves the selected algorithm and town graph
        intact; :meth:`new_town` replaces the graph before calling reset.
        """
        # stop() prevents a previously scheduled timer callback from
        # progressing an old or cleared search.
        self.timer.stop()
        self.header_algorithm.set_text(self.algorithm)
        # These states form the interaction flow: choose start, choose end,
        # run search, then allow another search.
        self.state = "pick_start"
        self.start = self.end = None
        self.gen = None
        self.search_elapsed = 0.0
        self.visited, self.frontier, self.prev = set(), set(), {}
        self.paused = False

        # Clear prior artists and configure a coordinate-proportional,
        # axis-free drawing area.
        ax = self.ax
        ax.clear()
        ax.set_facecolor(THEME["map"])
        ax.patch.set_edgecolor(THEME["border"])
        ax.patch.set_linewidth(1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        # values() returns all coordinate tuples; comprehensions select x and
        # y independently so axis bounds include every intersection.
        xs = [p[0] for p in self.pos.values()]
        ys = [p[1] for p in self.pos.values()]
        ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
        ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
        ax.add_patch(FancyBboxPatch(
            (0.002, 0.002), 0.996, 0.996,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=THEME["border"],
            linewidth=1.0,
            zorder=13,
            clip_on=False,
        ))

        # All roads
        # The adjacency is symmetric, so a < b draws each undirected road only
        # once and avoids duplicate overlapping line segments.
        roads = [[self.pos[a], self.pos[b]]
                 for a in self.adj for b in self.adj[a] if a < b]
        ax.add_collection(LineCollection(roads, colors=THEME["road"],
                                         linewidths=2.5, zorder=1))
        # Road length labels (small)
        # Label each undirected road with its weight. The midpoint is the
        # arithmetic mean of its endpoint coordinates.
        for a in self.adj:
            for b, w in self.adj[a].items():
                if a < b:
                    mx = (self.pos[a][0] + self.pos[b][0]) / 2
                    my = (self.pos[a][1] + self.pos[b][1]) / 2
                    ax.text(mx, my, f"{w:.1f}", fontsize=6, color=THEME["text"],
                            ha="center", va="center", zorder=2,
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc=THEME["road_label"],
                                      ec=THEME["border"], lw=0.35, alpha=0.95))

        # Animated layers
        # Separate line collections let the search tree, active road, and
        # final path be recolored or cleared independently during animation.
        self.tree_lc = LineCollection([], colors=THEME["tree"], linewidths=3.5, zorder=3)
        self.cur_lc = LineCollection([], colors=THEME["current"], linewidths=5, zorder=4)
        self.path_lc = LineCollection([], colors=THEME["path"], linewidths=7, zorder=5)
        for lc in (self.tree_lc, self.cur_lc, self.path_lc):
            ax.add_collection(lc)

        # sorted() provides a stable node order, and np.array() converts the
        # coordinate list to columns usable by Matplotlib's scatter().
        all_xy = np.array([self.pos[n] for n in sorted(self.pos)])
        ax.scatter(all_xy[:, 0], all_xy[:, 1], s=42, c=THEME["node"],
                   edgecolors=THEME["node_edge"], linewidths=1.6, zorder=6)
        self.frontier_sc = ax.scatter([], [], s=110, c=THEME["frontier"],
                                      edgecolors=THEME["background"], linewidths=1.5,
                                      zorder=7)
        self.visited_sc = ax.scatter([], [], s=110, c=THEME["visited"],
                                     edgecolors=THEME["background"], linewidths=1.5,
                                     zorder=7)
        self.start_sc = ax.scatter([], [], s=380, c=THEME["start"], marker="o",
                                   edgecolors=THEME["background"], linewidths=2, zorder=9)
        self.end_sc = ax.scatter([], [], s=380, c=THEME["end"], marker="s",
                                 edgecolors=THEME["background"], linewidths=2, zorder=9)

        # Add a compact, color-keyed legend along the lower-right map edge.
        ax.add_patch(FancyBboxPatch(
            (0.60, 0.012), 0.385, 0.058,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor=THEME["panel"],
            edgecolor=THEME["border"],
            linewidth=0.8,
            zorder=11,
        ))
        legend_items = (
            (0.625, THEME["path"], "Shortest path", 0.642),
            (0.765, THEME["visited"], "Visited", 0.782),
            (0.847, THEME["node"], "Unvisited", 0.864),
        )
        for x, color, label, text_x in legend_items:
            ax.scatter(
                [x], [0.041], transform=ax.transAxes, s=24,
                facecolor=color, edgecolor=THEME["accent"], linewidth=1,
                zorder=12,
            )
            ax.text(
                text_x, 0.041, label, transform=ax.transAxes,
                fontsize=6.5, color=THEME["muted"], va="center", zorder=12,
            )
        ax.plot(
            [0.94, 0.955], [0.041, 0.041], transform=ax.transAxes,
            color=THEME["road"], linewidth=1.5, zorder=12,
        )
        ax.text(
            0.96, 0.041, "Road", transform=ax.transAxes,
            fontsize=6.5, color=THEME["muted"], va="center", zorder=12,
        )
        self.set_title(
            "Choose a start intersection"
        )
        self.fig.canvas.draw_idle()

    def on_algorithm_change(self, algorithm):
        """Save the selected algorithm and refresh the instruction text.

        Changing algorithms during an active search is ignored by the running
        generator: its algorithm was captured when the destination was chosen.
        """
        self.algorithm = algorithm
        for label in self.algorithm_radio.labels:
            selected = label.get_text() == algorithm
            label.set_color(THEME["accent"] if selected else THEME["text"])
            label.set_fontweight("bold" if selected else "normal")
        selected_label = next(
            label for label in self.algorithm_radio.labels
            if label.get_text() == algorithm
        )
        self.selection_highlight.set_y(
            selected_label.get_position()[1] - 0.06
        )
        # Avoid changing displayed instructions mid-search; the currently
        # executing algorithm remains fixed for that search.
        if self.state == "running":
            self.header_algorithm.set_text(self.search_algorithm)
            return
        self.header_algorithm.set_text(algorithm)
        if self.state == "pick_end":
            self.set_title("Choose a destination")
        elif self.state == "done":
            self.set_title("Click the map to try another route")
        else:
            self.set_title("Choose a start intersection")
        self.fig.canvas.draw_idle()

    def set_title(self, text):
        """Update the current interaction status in the application header."""
        self.header_status.set_text(text)

    def offsets(self, nodes):
        """Convert an iterable of node IDs into an N-by-2 coordinate array.

        Matplotlib scatter collections expect point offsets in an array. A
        correctly shaped empty array clears a collection with no selected
        nodes. ``list()`` materializes iterables so they can be checked and
        traversed reliably.
        """
        nodes = list(nodes)
        if not nodes:
            # Shape (0, 2) represents zero x/y coordinate pairs.
            return np.empty((0, 2))
        return np.array([self.pos[n] for n in nodes])

    def seg(self, a, b):
        """Return the two endpoint coordinates for a road line segment."""
        return [self.pos[a], self.pos[b]]

    def refresh(self):
        """Copy current search state into Matplotlib artists and redraw.

        set_offsets() updates scatter-point locations, while set_segments()
        updates line collections. draw_idle() asks the GUI event loop to
        repaint at a convenient time rather than forcing an immediate draw.
        """
        self.visited_sc.set_offsets(self.offsets(self.visited))
        self.frontier_sc.set_offsets(self.offsets(self.frontier))
        self.tree_lc.set_segments([self.seg(v, u) for v, u in self.prev.items()])
        self.fig.canvas.draw_idle()

    # ---- events -----------------------------------------------------------
    def nearest_node(self, x, y, max_dist=0.6):
        """Return the closest town node within max_dist, or None.

        Args:
            x: Click x-coordinate in map data coordinates.
            y: Click y-coordinate in map data coordinates.
            max_dist: Maximum accepted Euclidean distance from the click.

        math.hypot(dx, dy) computes the Euclidean distance in a numerically
        stable way. The strict ``<`` comparison excludes a node exactly at the
        threshold; ``best`` remains None when no node is close enough.
        """
        best, best_d = None, max_dist
        for n, (px, py) in self.pos.items():
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best, best_d = n, d
        return best

    def on_click(self, event):
        """Handle map clicks for selecting endpoints and starting searches.

        Matplotlib provides the event's axes, mouse button, and data-space
        coordinates. Clicks outside the map, non-left clicks, and clicks that
        do not land near a node are ignored.
        """
        # Ignore events that cannot represent a valid left-click on the map.
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None:
            return
        # Map a pixel click to the nearest graph intersection within tolerance.
        node = self.nearest_node(event.xdata, event.ydata)
        if node is None:
            return

        if self.state == "done":
            self.reset()

        if self.state == "pick_start":
            self.start = node
            self.start_sc.set_offsets(self.offsets([node]))
            self.state = "pick_end"
            self.ax.annotate(
                "Start",
                self.pos[node],
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=THEME["text"],
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": THEME["start"],
                    "edgecolor": "none",
                },
                zorder=11,
            )
            self.set_title("Choose a destination")
            self.fig.canvas.draw_idle()
        elif self.state == "pick_end":
            if node == self.start:
                return
            self.end = node
            self.end_sc.set_offsets(self.offsets([node]))
            self.ax.annotate(
                "End",
                self.pos[node],
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=THEME["text"],
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": THEME["end"],
                    "edgecolor": "none",
                },
                zorder=11,
            )
            self.state = "running"
            self.search_algorithm = self.algorithm
            self.header_algorithm.set_text(self.search_algorithm)
            self.search_elapsed = 0.0
            # Store factories (lambdas) rather than already-created generators.
            # The selected factory is invoked only after both endpoints are set.
            search_functions = {
                "Dijkstra": lambda: dijkstra_steps(
                    self.adj, self.start, self.end
                ),
                "A*": lambda: a_star_steps(
                    self.adj, self.pos, self.start, self.end
                ),
                "Bidirectional Dijkstra": lambda: bidirectional_dijkstra_steps(
                    self.adj, self.start, self.end
                ),
                "Bidirectional A*": lambda: bidirectional_a_star_steps(
                    self.adj, self.pos, self.start, self.end
                ),
                "Greedy Best-First": lambda: greedy_best_first_steps(
                    self.adj, self.pos, self.start, self.end
                ),
                "BFS": lambda: breadth_first_steps(
                    self.adj, self.start, self.end
                ),
            }
            # Calling the selected lambda creates a lazy generator. Its
            # search work runs incrementally as tick() calls next().
            self.gen = search_functions[self.search_algorithm]()
            self.set_title("Searching")
            self.timer.start()

    def on_key(self, event):
        """Handle keyboard shortcuts for reset, town generation, speed, and quit."""
        if event.key == "r":
            # Reuse the same reset path as a new search, preserving this town.
            self.reset()
        elif event.key == "n":
            # Generate a new graph, then reset the map to show that graph.
            self.new_town()
        elif event.key == " ":
            if self.state == "running":
                # Toggle pause only when a search is active; timer.stop/start
                # controls whether the next generator event will be requested.
                self.paused = not self.paused
                if self.paused:
                    self.timer.stop()
                    self.set_title("Paused")
                    self.fig.canvas.draw_idle()
                else:
                    self.timer.start()
        elif event.key in ("+", "="):
            # Reduce the interval by 30% for faster updates, but never below
            # 15 ms so the timer remains within a practical lower bound.
            self.set_speed(max(15, int(self.interval * 0.7)))
        elif event.key in ("-", "_"):
            # Increase the interval by 40% for slower updates, capped at
            # 2000 ms (two seconds) between events.
            self.set_speed(min(2000, int(self.interval * 1.4)))
        elif event.key == "q":
            # Close this figure's window; pyplot handles the GUI shutdown.
            plt.close(self.fig)

    def set_speed(self, ms):
        """Set the timer interval in milliseconds and restart it if needed.

        A running, unpaused timer must be stopped and restarted for the GUI
        timer to use the updated interval. A paused or inactive search is left
        stopped.
        """
        self.interval = ms
        self.timer.interval = ms
        if self.state == "running" and not self.paused:
            self.timer.stop()
            self.timer.start()

    # ---- animation step ---------------------------------------------------
    def tick(self):
        """Consume one search event and apply its visual and textual updates.

        A timer calls this method repeatedly. Each yielded tuple describes a
        semantic step (start, visit, relax, or done), keeping algorithm logic
        separate from drawing logic.
        """
        try:
            # next_search_event() advances the generator by exactly one event.
            ev = self.next_search_event()
        except (StopIteration, TypeError):
            # StopIteration means the generator has no more events. TypeError
            # can occur if no valid generator is available to advance.
            self.timer.stop()
            return

        # Tuple element zero identifies which kind of update is needed.
        kind = ev[0]
        if kind == "start":
            self.frontier.add(ev[1])
            self.set_title("Search started")

        elif kind == "visit":
            # A visited node leaves the frontier, and any previous active-road
            # highlight is cleared before the new node's status is displayed.
            _, u, d = ev
            self.visited.add(u)
            self.frontier.discard(u)
            self.cur_lc.set_segments([])
            self.set_title(f"Visited {len(self.visited)}  ·  {d:.2f} km")
            if (u == self.end and self.search_algorithm not in (
                    "Bidirectional Dijkstra", "Bidirectional A*")):
                # Single-frontier generators stop after visiting the
                # destination, so request their final route event immediately.
                self.timer.stop()
                ev = self.next_search_event()
                if ev[0] != "done":
                    raise RuntimeError(
                        "Search continued after visiting the destination."
                    )
                kind = ev[0]

        elif kind == "relax":
            # The optional event detail marks backward-search events. The
            # predecessor line must be oriented to match the search direction.
            _, u, v, nd, improved, *event_details = ev
            self.cur_lc.set_segments([self.seg(u, v)])
            if improved:
                if event_details and event_details[0]:
                    self.prev[u] = v
                else:
                    self.prev[v] = u
                self.frontier.add(v)
                self.set_title(f"Better route found  ·  {nd:.2f} km")
            else:
                self.set_title("Checking alternate road")

        if kind == "done":
            # Draw the completed route, clear pending nodes, and retain the
            # finished state so the next map click starts another attempt.
            _, path, total = ev
            self.cur_lc.set_segments([])
            self.path_lc.set_segments([self.seg(a, b)
                                       for a, b in zip(path, path[1:])])
            self.frontier.clear()
            self.state = "done"
            self.timer.stop()
            # Keep the result label honest about BFS and greedy search, which
            # do not guarantee the shortest total road length.
            result_type = {
                "BFS": "Fewest roads",
                "Greedy Best-First": "Route found",
            }.get(self.search_algorithm, "Shortest path")
            self.set_title(
                f"{result_type}  ·  {total:.2f} km  ·  {len(path)} nodes  ·  "
                f"{self.search_elapsed:.4f} sec"
            )

        self.refresh()

    def next_search_event(self):
        """Advance the search generator and accumulate its execution time.

        ``time.perf_counter()`` is a monotonic high-resolution clock suitable
        for measuring elapsed intervals. The finally block updates the total
        even if the generator raises or finishes, so the timer counts the time
        spent inside each call to next(), rather than animation waiting time.
        """
        started_at = time.perf_counter()
        try:
            return next(self.gen)
        finally:
            self.search_elapsed += time.perf_counter() - started_at


if __name__ == "__main__":
    # Constructing the app prepares the window and first town. plt.show()
    # starts the GUI event loop and keeps the window responsive until closed.
    app = ShortestPathApp()
    plt.show()
