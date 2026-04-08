import streamlit as st
import math
import graphviz
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import copy

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class IndexNode:
    """Represents a node in the index tree"""
    node_id: str
    level: int
    keys: List[int] = field(default_factory=list)
    children: List['IndexNode'] = field(default_factory=list)
    parent: Optional['IndexNode'] = None
    data_bucket_ids: List[int] = field(default_factory=list)
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def last_key(self) -> int:
        """Returns the last (maximum) key indexed by this node"""
        if self.keys:
            return max(self.keys)
        return 0
    
    def __repr__(self):
        return f"IndexNode({self.node_id}, level={self.level}, keys={self.keys})"


@dataclass
class DataBucket:
    """Represents a data bucket in the broadcast"""
    bucket_id: int
    keys: List[int] = field(default_factory=list)
    
    def __repr__(self):
        return f"DataBucket({self.bucket_id}, keys={self.keys})"


@dataclass 
class ControlIndexEntry:
    """An entry in the control index"""
    threshold_key: int
    target_pointer: str
    description: str


@dataclass
class BroadcastSegment:
    """Represents a segment in the broadcast schedule"""
    segment_type: str  # 'rep', 'ind', 'data', 'control'
    content: any
    position: int
    description: str


@dataclass
class AccessStep:
    """Represents a step in the access protocol"""
    step_num: int
    action: str
    position: int
    bucket_accessed: str
    mode: str  # 'active' or 'doze'
    details: str


# ============================================================================
# INDEX TREE CONSTRUCTION
# ============================================================================

class IndexTree:
    """Class to build and manage the index tree"""
    
    def __init__(self, num_data_buckets: int, bucket_capacity: int, key_values: List[int] = None):
        self.num_data_buckets = num_data_buckets
        self.bucket_capacity = bucket_capacity
        self.num_levels = self._calculate_levels()
        self.root = None
        self.data_buckets = []
        self.all_nodes = {}  # node_id -> IndexNode
        self.nodes_by_level = {}  # level -> List[IndexNode]
        
        # Generate key values if not provided
        if key_values is None:
            self.key_values = list(range(1, num_data_buckets + 1))
        else:
            self.key_values = sorted(key_values)[:num_data_buckets]
        
        self._build_tree()
    
    def _calculate_levels(self) -> int:
        """Calculate the number of levels in the index tree"""
        if self.num_data_buckets <= 1:
            return 1
        return math.ceil(math.log(self.num_data_buckets, self.bucket_capacity))
    
    def _build_tree(self):
        """Build the index tree bottom-up"""
        n = self.bucket_capacity
        
        # Create data buckets
        self.data_buckets = []
        for i in range(self.num_data_buckets):
            db = DataBucket(bucket_id=i, keys=[self.key_values[i]])
            self.data_buckets.append(db)
        
        # Initialize nodes_by_level
        for level in range(self.num_levels + 1):
            self.nodes_by_level[level] = []
        
        # Create leaf level index nodes (level k-1, just above data)
        leaf_nodes = []
        node_counter = 0
        
        for i in range(0, self.num_data_buckets, n):
            chunk = self.data_buckets[i:i+n]
            node_id = f"L{self.num_levels-1}_N{node_counter}"
            keys = [db.keys[0] for db in chunk]
            data_ids = [db.bucket_id for db in chunk]
            
            node = IndexNode(
                node_id=node_id,
                level=self.num_levels - 1,
                keys=keys,
                data_bucket_ids=data_ids
            )
            leaf_nodes.append(node)
            self.all_nodes[node_id] = node
            self.nodes_by_level[self.num_levels - 1].append(node)
            node_counter += 1
        
        # Build upper levels
        current_level_nodes = leaf_nodes
        current_level = self.num_levels - 2
        
        while len(current_level_nodes) > 1 or current_level >= 0:
            next_level_nodes = []
            node_counter = 0
            
            for i in range(0, len(current_level_nodes), n):
                chunk = current_level_nodes[i:i+n]
                node_id = f"L{current_level}_N{node_counter}"
                
                # Keys are the maximum keys from children
                keys = [child.last_key() for child in chunk]
                
                node = IndexNode(
                    node_id=node_id,
                    level=current_level,
                    keys=keys,
                    children=chunk
                )
                
                # Set parent references
                for child in chunk:
                    child.parent = node
                
                next_level_nodes.append(node)
                self.all_nodes[node_id] = node
                self.nodes_by_level[current_level].append(node)
                node_counter += 1
            
            current_level_nodes = next_level_nodes
            current_level -= 1
            
            if len(current_level_nodes) == 1 and current_level < 0:
                break
        
        # Set root
        if current_level_nodes:
            self.root = current_level_nodes[0]
            self.root.level = 0
            # Fix the root's node_id and level tracking
            if self.root.node_id in self.all_nodes:
                old_level = int(self.root.node_id.split('_')[0][1:])
                if old_level in self.nodes_by_level:
                    if self.root in self.nodes_by_level[old_level]:
                        self.nodes_by_level[old_level].remove(self.root)
            self.root.node_id = "ROOT"
            self.all_nodes["ROOT"] = self.root
            if 0 not in self.nodes_by_level:
                self.nodes_by_level[0] = []
            self.nodes_by_level[0].append(self.root)
    
    def get_nodes_at_level(self, level: int) -> List[IndexNode]:
        """Get all nodes at a specific level"""
        return self.nodes_by_level.get(level, [])
    
    def get_path_to_root(self, node: IndexNode) -> List[IndexNode]:
        """Get the path from a node to the root (excluding the node itself)"""
        path = []
        current = node.parent
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def find_lca(self, node1: IndexNode, node2: IndexNode) -> Optional[IndexNode]:
        """Find the Least Common Ancestor of two nodes"""
        path1 = self.get_path_to_root(node1) + [node1]
        path2 = self.get_path_to_root(node2) + [node2]
        
        lca = None
        for n1, n2 in zip(path1, path2):
            if n1.node_id == n2.node_id:
                lca = n1
            else:
                break
        return lca
    
    def get_path_from_lca(self, lca: IndexNode, target: IndexNode) -> List[IndexNode]:
        """Get path from LCA to target (excluding target)"""
        full_path = self.get_path_to_root(target) + [target]
        result = []
        found_lca = False
        for node in full_path:
            if found_lca and node.node_id != target.node_id:
                result.append(node)
            if node.node_id == lca.node_id:
                found_lca = True
        return result


# ============================================================================
# DISTRIBUTED INDEXING ALGORITHM
# ============================================================================

class DistributedIndex:
    """Implements the distributed indexing algorithm"""
    
    def __init__(self, index_tree: IndexTree, replication_level: int):
        self.tree = index_tree
        self.r = replication_level  # Number of replicated levels
        self.k = index_tree.num_levels
        self.n = index_tree.bucket_capacity
        self.nrr = []  # Non-Replicated Roots
        self.broadcast_schedule = []
        self.control_indices = {}
        
        self._identify_nrr()
        self._generate_schedule()
    
    def _identify_nrr(self):
        """Identify Non-Replicated Roots at level (r+1)"""
        if self.r >= self.k:
            # Everything is replicated, NRR is empty
            self.nrr = []
        elif self.r == 0:
            # Only root is replicated (effectively)
            self.nrr = [self.tree.root] if self.tree.root else []
        else:
            # NRR is at level r (0-indexed from root)
            nrr_level = self.r
            self.nrr = self.tree.get_nodes_at_level(nrr_level)
        
        # If NRR is empty or we're at root level, use leaf nodes
        if not self.nrr and self.k > 0:
            # Find the deepest level with nodes
            for level in range(self.k - 1, -1, -1):
                nodes = self.tree.get_nodes_at_level(level)
                if nodes:
                    self.nrr = nodes
                    break
    
    def _generate_rep(self, b_index: int) -> List[Dict]:
        """Generate Rep(B) - the replicated part for B"""
        if b_index >= len(self.nrr):
            return []
        
        B = self.nrr[b_index]
        segments = []
        
        if b_index == 0:
            # Rep(B1) = Path(I, B1) - path from root to first NRR
            path = self.tree.get_path_to_root(B)
            for node in path:
                control_index = self._generate_control_index(node, b_index)
                segments.append({
                    'type': 'rep',
                    'node': node,
                    'control_index': control_index,
                    'description': f"Replicated: {node.node_id}"
                })
        else:
            # Rep(Bi) = Path(LCA(B_{i-1}, Bi), Bi)
            B_prev = self.nrr[b_index - 1]
            lca = self.tree.find_lca(B_prev, B)
            if lca:
                path = self.tree.get_path_from_lca(lca, B)
                for node in path:
                    control_index = self._generate_control_index(node, b_index)
                    segments.append({
                        'type': 'rep',
                        'node': node,
                        'control_index': control_index,
                        'description': f"Replicated: {node.node_id}"
                    })
        
        return segments
    
    def _generate_ind(self, B: IndexNode) -> List[Dict]:
        """Generate Ind(B) - the non-replicated index below B"""
        segments = []
        
        def traverse(node: IndexNode):
            if node.node_id != B.node_id:
                segments.append({
                    'type': 'ind',
                    'node': node,
                    'description': f"Index: {node.node_id}"
                })
            for child in node.children:
                traverse(child)
        
        # Add B itself
        segments.append({
            'type': 'ind',
            'node': B,
            'description': f"Index (NRR): {B.node_id}"
        })
        
        # Add children
        for child in B.children:
            traverse(child)
        
        return segments
    
    def _generate_data(self, B: IndexNode) -> List[Dict]:
        """Generate Data(B) - the data buckets indexed by B"""
        segments = []
        
        def get_data_buckets(node: IndexNode) -> List[int]:
            if node.data_bucket_ids:
                return node.data_bucket_ids
            result = []
            for child in node.children:
                result.extend(get_data_buckets(child))
            return result
        
        data_ids = get_data_buckets(B)
        for data_id in sorted(data_ids):
            if data_id < len(self.tree.data_buckets):
                db = self.tree.data_buckets[data_id]
                segments.append({
                    'type': 'data',
                    'bucket': db,
                    'description': f"Data Bucket {data_id}: keys={db.keys}"
                })
        
        return segments
    
    def _generate_control_index(self, node: IndexNode, nrr_index: int) -> List[ControlIndexEntry]:
        """Generate control index for a replicated node"""
        control_index = []
        
        # Get the path from root to this node
        path = self.tree.get_path_to_root(node) + [node]
        
        # Entry for "begin" - go to next bcast if key is less than last broadcasted
        if nrr_index > 0:
            prev_nrr = self.nrr[nrr_index - 1]
            last_key = self._get_last_key_before(prev_nrr)
            control_index.append(ControlIndexEntry(
                threshold_key=last_key,
                target_pointer="NEXT_BCAST",
                description=f"If key ≤ {last_key}, go to next broadcast"
            ))
        
        # Entries for each level in the path
        for i, path_node in enumerate(path[:-1]):
            last_key = path_node.last_key()
            next_occurrence = self._find_next_occurrence(path_node, nrr_index)
            control_index.append(ControlIndexEntry(
                threshold_key=last_key,
                target_pointer=next_occurrence,
                description=f"If key > {last_key}, go to {next_occurrence}"
            ))
        
        return control_index
    
    def _get_last_key_before(self, nrr_node: IndexNode) -> int:
        """Get the last key broadcasted before the given NRR node"""
        def get_max_key(node: IndexNode) -> int:
            if node.data_bucket_ids:
                max_key = 0
                for did in node.data_bucket_ids:
                    if did < len(self.tree.data_buckets):
                        db = self.tree.data_buckets[did]
                        if db.keys:
                            max_key = max(max_key, max(db.keys))
                return max_key
            max_key = 0
            for child in node.children:
                max_key = max(max_key, get_max_key(child))
            return max_key
        
        return get_max_key(nrr_node)
    
    def _find_next_occurrence(self, node: IndexNode, current_nrr_index: int) -> str:
        """Find the next occurrence of a node in the schedule"""
        # Look for the node in subsequent NRR sections
        for i in range(current_nrr_index + 1, len(self.nrr)):
            B = self.nrr[i]
            if i == 0:
                path = self.tree.get_path_to_root(B)
            else:
                B_prev = self.nrr[i - 1]
                lca = self.tree.find_lca(B_prev, B)
                if lca:
                    path = self.tree.get_path_from_lca(lca, B)
                else:
                    path = []
            
            for p_node in path:
                if p_node.node_id == node.node_id:
                    return f"Section_{i}"
        
        return "NEXT_BCAST"
    
    def _generate_schedule(self):
        """Generate the complete broadcast schedule"""
        self.broadcast_schedule = []
        position = 0
        
        if not self.nrr: # Handle case where tree is small and has no NRR
            return

        for i, B in enumerate(self.nrr):
            # Add Rep(B)
            rep_segments = self._generate_rep(i)
            for seg in rep_segments:
                seg['position'] = position
                seg['section'] = i
                self.broadcast_schedule.append(seg)
                position += 1
            
            # Add Ind(B)
            ind_segments = self._generate_ind(B)
            for seg in ind_segments:
                seg['position'] = position
                seg['section'] = i
                self.broadcast_schedule.append(seg)
                position += 1
            
            # Add Data(B)
            data_segments = self._generate_data(B)
            for seg in data_segments:
                seg['position'] = position
                seg['section'] = i
                self.broadcast_schedule.append(seg)
                position += 1
    
    def get_schedule_length(self) -> int:
        """Get the total length of the broadcast schedule"""
        return len(self.broadcast_schedule)
    
    @staticmethod
    def calculate_optimal_r(num_data_buckets: int, bucket_capacity: int, k: int) -> int:
        """Calculate the optimal replication level"""
        if k <= 1:
            return 0
        
        n = bucket_capacity
        Data = num_data_buckets
        
        try:
            numerator = Data * (n - 1) + n ** (k + 1)
            denominator = n - 1
            
            if numerator <= 0 or denominator <= 0:
                return 0
            
            r_star = 0.5 * (math.log(numerator / denominator, n) - 1)
            r_star = max(0, min(k - 1, int(r_star) + 1))
            return r_star
        except (ValueError, ZeroDivisionError):
            return 0


# ============================================================================
# ACCESS PROTOCOL SIMULATION
# ============================================================================

class AccessSimulator:
    """Simulates the client access protocol"""
    
    def __init__(self, distributed_index: DistributedIndex):
        self.di = distributed_index
        self.schedule = distributed_index.broadcast_schedule
        self.tree = distributed_index.tree
    
    def simulate_access(self, tune_in_position: int, target_key: int) -> List[AccessStep]:
        """Simulate the access protocol for a given target key"""
        steps = []
        step_num = 1

        if not self.schedule:
             steps.append(AccessStep(step_num=1, action="Error", position=-1, bucket_accessed="None", mode="active", details="Broadcast schedule is empty. Cannot simulate."))
             return steps

        current_pos = tune_in_position % len(self.schedule)
        
        # Step 1: Initial probe - tune to current bucket
        steps.append(AccessStep(
            step_num=step_num,
            action="Initial Probe",
            position=current_pos,
            bucket_accessed=self._get_bucket_name(current_pos),
            mode="active",
            details=f"Tune into broadcast at position {current_pos}"
        ))
        step_num += 1
        
        # Find the next index/control segment
        next_index_pos = self._find_next_index(current_pos)
        
        if next_index_pos is None:
            # Wrap around to beginning
            next_index_pos = self._find_next_index(0)
            if next_index_pos is None:
                steps.append(AccessStep(
                    step_num=step_num,
                    action="Error",
                    position=-1,
                    bucket_accessed="None",
                    mode="active",
                    details="No index found in schedule"
                ))
                return steps
        
        # Step 2: Go to doze mode and wait for index
        if next_index_pos > current_pos:
            steps.append(AccessStep(
                step_num=step_num,
                action="Doze Mode",
                position=current_pos,
                bucket_accessed="-",
                mode="doze",
                details=f"Sleep until position {next_index_pos} (skip {next_index_pos - current_pos - 1} buckets)"
            ))
            step_num += 1
        
        # Step 3: Access the index/control segment
        current_pos = next_index_pos
        segment = self.schedule[current_pos]
        
        steps.append(AccessStep(
            step_num=step_num,
            action="Read Index",
            position=current_pos,
            bucket_accessed=self._get_bucket_name(current_pos),
            mode="active",
            details=f"Read index bucket: {segment.get('description', 'Unknown')}"
        ))
        step_num += 1
        
        # Check control index if present
        if 'control_index' in segment and segment['control_index']:
            ci_result = self._check_control_index(segment['control_index'], target_key, current_pos)
            steps.append(AccessStep(
                step_num=step_num,
                action="Check Control Index",
                position=current_pos,
                bucket_accessed=self._get_bucket_name(current_pos),
                mode="active",
                details=ci_result['details']
            ))
            step_num += 1
            
            if ci_result['redirect']:
                # Need to redirect based on control index
                redirect_pos = ci_result['redirect_pos']
                if redirect_pos > current_pos:
                    steps.append(AccessStep(
                        step_num=step_num,
                        action="Doze Mode",
                        position=current_pos,
                        bucket_accessed="-",
                        mode="doze",
                        details=f"Sleep until redirect position {redirect_pos}"
                    ))
                    step_num += 1
                current_pos = redirect_pos
        
        # Follow index pointers to find data
        found = False
        max_iterations = len(self.schedule) + 5
        iterations = 0
        
        while not found and iterations < max_iterations:
            iterations += 1
            
            # Find next relevant index or data bucket
            for pos in range(current_pos, len(self.schedule)):
                seg = self.schedule[pos]
                
                if seg['type'] == 'data':
                    bucket = seg['bucket']
                    if target_key in bucket.keys:
                        # Found the data!
                        if pos > current_pos:
                            steps.append(AccessStep(
                                step_num=step_num,
                                action="Doze Mode",
                                position=current_pos,
                                bucket_accessed="-",
                                mode="doze",
                                details=f"Sleep until data bucket at position {pos}"
                            ))
                            step_num += 1
                        
                        steps.append(AccessStep(
                            step_num=step_num,
                            action="Download Data",
                            position=pos,
                            bucket_accessed=f"Data Bucket {bucket.bucket_id}",
                            mode="active",
                            details=f"Found target key {target_key} in Data Bucket {bucket.bucket_id}!"
                        ))
                        found = True
                        break
                
                elif seg['type'] in ['ind', 'rep']:
                    node = seg['node']
                    if self._node_contains_key(node, target_key):
                        if pos > current_pos:
                            steps.append(AccessStep(
                                step_num=step_num,
                                action="Doze Mode",
                                position=current_pos,
                                bucket_accessed="-",
                                mode="doze",
                                details=f"Sleep until index at position {pos}"
                            ))
                            step_num += 1
                        
                        steps.append(AccessStep(
                            step_num=step_num,
                            action="Read Index",
                            position=pos,
                            bucket_accessed=node.node_id,
                            mode="active",
                            details=f"Index {node.node_id} covers key {target_key}, following pointer..."
                        ))
                        step_num += 1
                        current_pos = pos + 1
                        break
            else:
                # Reached end of schedule without finding
                if not found:
                    steps.append(AccessStep(
                        step_num=step_num,
                        action="Wait for Next Bcast",
                        position=len(self.schedule),
                        bucket_accessed="-",
                        mode="doze",
                        details="Key not found in current broadcast, waiting for next..."
                    ))
                break
        
        if not found and iterations >= max_iterations:
            steps.append(AccessStep(
                step_num=step_num,
                action="Search Terminated",
                position=-1,
                bucket_accessed="-",
                mode="active",
                details=f"Key {target_key} not found (max iterations reached)"
            ))
        
        return steps
    
    def _find_next_index(self, start_pos: int) -> Optional[int]:
        """Find the next index segment starting from position"""
        for pos in range(start_pos, len(self.schedule)):
            seg = self.schedule[pos]
            if seg['type'] in ['rep', 'ind']:
                return pos
        return None
    
    def _get_bucket_name(self, pos: int) -> str:
        """Get the name of the bucket at position"""
        if pos < 0 or pos >= len(self.schedule):
            return "Unknown"
        seg = self.schedule[pos]
        if seg['type'] == 'data':
            return f"Data Bucket {seg['bucket'].bucket_id}"
        elif seg['type'] in ['rep', 'ind']:
            return seg['node'].node_id
        return "Unknown"
    
    def _check_control_index(self, control_index: List[ControlIndexEntry], 
                            target_key: int, current_pos: int) -> Dict:
        """Check control index and determine redirection"""
        for entry in control_index:
            if target_key <= entry.threshold_key:
                if entry.target_pointer == "NEXT_BCAST":
                    return {
                        'redirect': True,
                        'redirect_pos': 0,  # Beginning of next bcast
                        'details': f"Key {target_key} ≤ {entry.threshold_key}: redirect to next broadcast"
                    }
                else:
                    # Find the section
                    section_num = int(entry.target_pointer.split('_')[1]) if '_' in entry.target_pointer else 0
                    redirect_pos = self._find_section_start(section_num)
                    return {
                        'redirect': True,
                        'redirect_pos': redirect_pos,
                        'details': f"Key {target_key} > {entry.threshold_key}: redirect to {entry.target_pointer}"
                    }
        
        return {
            'redirect': False,
            'redirect_pos': current_pos,
            'details': f"Key {target_key} is in range, continue with current index"
        }
    
    def _find_section_start(self, section_num: int) -> int:
        """Find the starting position of a section"""
        for pos, seg in enumerate(self.schedule):
            if seg.get('section', -1) == section_num:
                return pos
        return 0
    
    def _node_contains_key(self, node: IndexNode, key: int) -> bool:
        """Check if a node's subtree contains the given key"""
        if key in node.keys:
            return True
        
        # Check data buckets
        for did in node.data_bucket_ids:
            if did < len(self.tree.data_buckets):
                if key in self.tree.data_buckets[did].keys:
                    return True
        
        # Check children
        for child in node.children:
            if self._node_contains_key(child, key):
                return True
        
        return False


# ============================================================================
# VISUALIZATION (UPDATED)
# ============================================================================

def create_tree_visualization(tree: IndexTree, replication_level: int) -> graphviz.Digraph:
    """Create a Graphviz visualization of the index tree"""
    dot = graphviz.Digraph(comment='Index Tree')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='rounded')

    # <<< THIS IS THE FIX >>>
    # Adjust graph attributes to make the tree taller and narrower.
    # ranksep: Increases vertical distance between levels.
    # nodesep: Decreases horizontal distance between nodes on the same level.
    # ratio='compress': Tells the engine to make the drawing compact.
    dot.attr('graph', ranksep='1.2', nodesep='0.3', ratio='compress')
    
    def add_node(node: IndexNode, is_replicated: bool):
        color = 'lightblue' if is_replicated else 'lightgreen'
        border = 'red' if node.level == replication_level else 'black'
        label = f"{node.node_id}\nKeys: {node.keys}"
        
        dot.node(node.node_id, label, fillcolor=color, style='filled,rounded', 
                color=border, penwidth='2' if node.level == replication_level else '1')
        
        for child in node.children:
            child_replicated = child.level < replication_level
            add_node(child, child_replicated)
            dot.edge(node.node_id, child.node_id)
    
    if tree.root:
        add_node(tree.root, tree.root.level < replication_level)
    
    # Add data buckets
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightyellow')
    for db in tree.data_buckets:
        dot.node(f"D{db.bucket_id}", f"Data {db.bucket_id}\n{db.keys}")
    
    # Connect leaf nodes to data buckets
    for level in range(tree.num_levels):
        for node in tree.get_nodes_at_level(level):
            for did in node.data_bucket_ids:
                dot.edge(node.node_id, f"D{did}", style='dashed')
    
    return dot


def create_schedule_visualization(schedule: List[Dict]) -> pd.DataFrame:
    """Create a DataFrame visualization of the broadcast schedule"""
    data = []
    for i, seg in enumerate(schedule):
        if seg['type'] == 'data':
            bucket = seg['bucket']
            name = f"Data_{bucket.bucket_id}"
            keys = str(bucket.keys)
        else:
            name = seg['node'].node_id
            keys = str(seg['node'].keys)
        
        data.append({
            'Position': i,
            'Type': seg['type'].upper(),
            'Name': name,
            'Keys': keys,
            'Section': seg.get('section', '-'),
            'Description': seg.get('description', '')
        })
    
    return pd.DataFrame(data)


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(tree: IndexTree, di: DistributedIndex) -> Dict:
    """Calculate access time and tuning time metrics"""
    n = tree.bucket_capacity
    k = tree.num_levels
    r = di.r
    Data = tree.num_data_buckets
    
    # Calculate index size
    index_size = sum(n**i for i in range(k))
    
    # Probe wait (approximate)
    if n > 1 and r < k and n**r > 0:
        probe_wait = 0.5 * ((n**(k-r) - 1) / (n - 1) + Data / (n**r))
    else:
        probe_wait = 0.5 * Data
    
    # Beast wait (approximate)
    index_overhead = (n**r - 1) if r > 0 else 0
    beast_wait = 0.5 * (index_overhead + index_size + Data)
    
    # Access time
    access_time = probe_wait + beast_wait
    
    # Tuning time
    tuning_time = k + 3  # log_n(Data) + 3 probes
    
    # Compare with optimal methods
    access_opt_time = Data / 2
    access_opt_tune = Data / 2
    
    tune_opt_access = Data + index_size
    tune_opt_tune = k + 1
    
    return {
        'access_time': access_time,
        'tuning_time': tuning_time,
        'probe_wait': probe_wait,
        'beast_wait': beast_wait,
        'schedule_length': len(di.broadcast_schedule),
        'index_size': index_size,
        'replication_overhead': index_overhead,
        'access_opt_time': access_opt_time,
        'access_opt_tune': access_opt_tune,
        'tune_opt_access': tune_opt_access,
        'tune_opt_tune': tune_opt_tune
    }


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="Distributed Indexing for Data Broadcasting",
        page_icon="📡",
        layout="wide"
    )
    
    st.title("📡 Distributed Indexing for Data Broadcasting")
    st.markdown("""
    This application implements the **Partial Replication Scheme (Distributed Indexing)** 
    for organizing and accessing data broadcast over wireless channels.
    """)
    
    # ========================================================================
    # SIDEBAR - ALL USER INPUTS
    # ========================================================================
    st.sidebar.header("⚙️ User Inputs")
    
    # Input 1: Number of Data Buckets
    st.sidebar.subheader("1. Number of Data Buckets")
    num_data_buckets = st.sidebar.number_input(
        "Total data buckets in the file",
        min_value=1,
        max_value=1000,
        value=27,
        help="Defines file size and index tree structure"
    )
    
    # Input 2: Bucket Capacity (n)
    st.sidebar.subheader("2. Bucket Capacity (n)")
    bucket_capacity = st.sidebar.number_input(
        "Pointers per index bucket (fanout)",
        min_value=2,
        max_value=50,
        value=3,
        help="Determines tree fanout and index tree depth"
    )
    
    # Input 3: Data Bucket Key Values
    st.sidebar.subheader("3. Data Bucket Key Values")
    key_option = st.sidebar.radio(
        "Key Generation Method",
        ["Auto-generate (1 to N)", "Custom keys"],
        help="Populates the index tree with searchable keys"
    )
    
    if key_option == "Custom keys":
        key_input = st.sidebar.text_area(
            "Enter keys (comma-separated)",
            value=",".join(str(i) for i in range(1, min(num_data_buckets + 1, 101))),
            help="Enter exactly N keys for N data buckets"
        )
        try:
            key_values = [int(k.strip()) for k in key_input.split(",") if k.strip()]
            if len(key_values) < num_data_buckets:
                st.sidebar.warning(f"Only {len(key_values)} keys provided. Auto-filling remaining.")
                key_values.extend(range(max(key_values)+1, max(key_values)+1+num_data_buckets-len(key_values)))
        except:
            st.sidebar.error("Invalid key format. Using auto-generated keys.")
            key_values = list(range(1, num_data_buckets + 1))
    else:
        key_values = list(range(1, num_data_buckets + 1))
    
    # Build the index tree first to get num_levels
    tree = IndexTree(num_data_buckets, bucket_capacity, key_values)
    
    # Input 4: Replication Level (r)
    st.sidebar.subheader("4. Replication Level (r)")
    optimal_r = DistributedIndex.calculate_optimal_r(
        num_data_buckets, bucket_capacity, tree.num_levels
    )
    
    st.sidebar.info(f"📊 Computed Optimal r* = {optimal_r}")
    st.sidebar.caption(f"Index tree has {tree.num_levels} levels")
    
    replication_option = st.sidebar.radio(
        "Replication Level Selection",
        ["Use optimal (auto-computed)", "Specify manually"],
        help="Controls how much of the upper index tree is replicated"
    )
    
    if replication_option == "Use optimal (auto-computed)":
        r = optimal_r
    else:
        r = st.sidebar.slider(
            "Manual Replication Level (r)",
            min_value=0,
            max_value=max(0, tree.num_levels - 1),
            value=min(optimal_r, max(0, tree.num_levels - 1)),
            help="Number of top levels to replicate"
        )
    
    st.sidebar.markdown("---")
    
    # Input 5: Client Tune-in Position
    st.sidebar.subheader("5. Client Tune-in Position")
    
    # Create distributed index to get schedule length
    di = DistributedIndex(tree, r)
    schedule_length = len(di.broadcast_schedule)
    
    tune_in_position = st.sidebar.number_input(
        "Position where client tunes in",
        min_value=0,
        max_value=max(0, schedule_length - 1) if schedule_length > 0 else 0,
        value=0,
        help="Starting point for demonstrating the access protocol"
    )
    
    st.sidebar.caption(f"Broadcast schedule length: {schedule_length} buckets")
    
    # Input 6: Target Search Key (K)
    st.sidebar.subheader("6. Target Search Key (K)")
    available_keys = sorted(set(k for db in tree.data_buckets for k in db.keys))
    
    if available_keys:
        default_index = min(len(available_keys)//2, len(available_keys)-1) if available_keys else 0
        target_key = st.sidebar.selectbox(
            "Key to search for",
            options=available_keys,
            index=default_index,
            help="The record the client wants to find"
        )
    else:
        target_key = st.sidebar.number_input(
            "Key to search for",
            value=1,
            help="The record the client wants to find"
        )
    
    st.sidebar.markdown("---")
    
    # Summary of inputs
    st.sidebar.subheader("📋 Input Summary")
    st.sidebar.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | Data Buckets | {num_data_buckets} |
    | Bucket Capacity (n) | {bucket_capacity} |
    | Tree Levels (k) | {tree.num_levels} |
    | Replication Level (r) | {r} |
    | Tune-in Position | {tune_in_position} |
    | Target Key (K) | {target_key} |
    """)
    
    # ========================================================================
    # MAIN CONTENT AREA WITH TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌳 Index Structure",
        "📋 Broadcast Schedule", 
        "🔍 Access Simulation",
        "📊 Metrics"
    ])
    
    # Tab 1: Index Structure (UPDATED LAYOUT)
    with tab1:
        st.header("Index Tree Structure")
        
        # Tree Visualization - Full Width
        st.subheader("Tree Visualization")
        try:
            dot = create_tree_visualization(tree, r)
            st.graphviz_chart(dot, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating visualization: {e}")
            st.info("Showing text representation instead:")
            
            def print_tree(node, indent=0):
                prefix = "  " * indent
                is_rep = "🔵 (Replicated)" if node.level < r else "🟢"
                st.text(f"{prefix}{is_rep} {node.node_id}: keys={node.keys}")
                for child in node.children:
                    print_tree(child, indent + 1)
            
            if tree.root:
                print_tree(tree.root)
        
        st.markdown("---")
        
        # Tree Properties, Legend, and NRR - Below the tree in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔢 Tree Properties")
            st.markdown(f"""
            | Property | Value |
            |----------|-------|
            | Number of Levels (k) | **{tree.num_levels}** |
            | Bucket Capacity (n) | **{bucket_capacity}** |
            | Replication Level (r) | **{r}** |
            | Data Buckets | **{num_data_buckets}** |
            | Total Index Nodes | **{len(tree.all_nodes)}** |
            """)
        
        with col2:
            st.subheader("🎨 Legend")
            st.markdown("""
            | Color | Meaning |
            |-------|---------|
            | 🔵 **Light Blue** | Replicated index buckets |
            | 🟢 **Light Green** | Non-replicated index buckets |
            | 🟡 **Light Yellow** | Data buckets |
            | 🔴 **Red Border** | NRR level (Non-Replicated Roots) |
            
            **Edges:**
            - **Solid lines**: Index to index connections
            - **Dashed lines**: Index to data connections
            """)
        
        with col3:
            st.subheader("📍 Non-Replicated Roots (NRR)")
            if di.nrr:
                st.markdown(f"NRR nodes are at **Level {r}** of the tree:")
                nrr_data = []
                for i, nrr_node in enumerate(di.nrr):
                    nrr_data.append({
                        "Index": i,
                        "Node ID": nrr_node.node_id,
                        "Keys": str(nrr_node.keys)
                    })
                nrr_df = pd.DataFrame(nrr_data)
                st.dataframe(nrr_df, use_container_width=True, hide_index=True)
            else:
                st.info("No NRR nodes (single level tree or r=0)")
        
        # Additional explanation
        st.markdown("---")
        with st.expander("ℹ️ Understanding the Index Tree Structure"):
            st.markdown(f"""
            ### How the Index Tree is Organized
            
            1. **Index Levels**: The tree has **{tree.num_levels}** levels (Level 0 = ROOT).
            
            2. **Replication**: The top **{r}** levels (Levels 0 to {r-1 if r > 0 else 'none'}) are **replicated** 
               in the broadcast schedule. These appear in light blue.
            
            3. **Non-Replicated Roots (NRR)**: Nodes at Level **{r}** are the NRR nodes. 
               Each NRR node is the root of a subtree that appears only once in the broadcast.
            
            4. **Data Buckets**: The {num_data_buckets} data buckets (shown in yellow ellipses) 
               contain the actual data keys.
            
            5. **Fanout**: Each index node has up to **{bucket_capacity}** children/pointers.
            """)
    
    # Tab 2: Broadcast Schedule
    with tab2:
        st.header("Broadcast Schedule")
        
        st.subheader("Schedule Overview")
        schedule_df = create_schedule_visualization(di.broadcast_schedule)
        
        # Color-code the dataframe
        def highlight_type(row):
            if row['Type'] == 'REP':
                return ['background-color: #ADD8E6'] * len(row)  # Light blue
            elif row['Type'] == 'IND':
                return ['background-color: #90EE90'] * len(row)  # Light green
            elif row['Type'] == 'DATA':
                return ['background-color: #FFFFE0'] * len(row)  # Light yellow
            return [''] * len(row)
        
        styled_df = schedule_df.style.apply(highlight_type, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.subheader("Schedule Structure")
        if di.nrr:
            st.markdown(f"""
            The broadcast consists of **{len(di.nrr)}** sections, each containing:
            - **Rep(B)**: Replicated path from LCA to NRR node (🔵 Light Blue)
            - **Ind(B)**: Non-replicated index subtree below B (🟢 Light Green)
            - **Data(B)**: Data buckets indexed by B (🟡 Light Yellow)
            """)
        
        # Show sections
        sections = {}
        for seg in di.broadcast_schedule:
            sec = seg.get('section', 0)
            if sec not in sections:
                sections[sec] = {'rep': [], 'ind': [], 'data': []}
            sections[sec][seg['type']].append(seg)
        
        for sec_num, sec_content in sections.items():
            nrr_name = di.nrr[sec_num].node_id if sec_num < len(di.nrr) else 'Unknown'
            with st.expander(f"📦 Section {sec_num}: NRR = {nrr_name}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**🔵 Rep(B)**")
                    if sec_content['rep']:
                        for s in sec_content['rep']:
                            st.text(f"• {s['node'].node_id}")
                    else:
                        st.text("(none)")
                with col2:
                    st.markdown("**🟢 Ind(B)**")
                    for s in sec_content['ind']:
                        st.text(f"• {s['node'].node_id}")
                with col3:
                    st.markdown("**🟡 Data(B)**")
                    for s in sec_content['data']:
                        st.text(f"• Bucket {s['bucket'].bucket_id}: {s['bucket'].keys}")
    
    # Tab 3: Access Simulation
    with tab3:
        st.header("Access Protocol Simulation")
        
        # Display current parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Tune-in Position:** {tune_in_position}")
        with col2:
            st.info(f"**Target Key:** {target_key}")
        with col3:
            st.info(f"**Schedule Length:** {schedule_length}")
        
        # Run simulation button
        if st.button("🔍 Run Access Simulation", type="primary", use_container_width=True):
            simulator = AccessSimulator(di)
            steps = simulator.simulate_access(tune_in_position, target_key)
            
            st.subheader("Access Sequence Results")
            
            # Summary metrics
            active_steps = sum(1 for s in steps if s.mode == 'active')
            doze_steps = sum(1 for s in steps if s.mode == 'doze')
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Steps", len(steps))
            with col2:
                st.metric("Active Mode Steps", active_steps, help="🔋 Energy consuming")
            with col3:
                st.metric("Doze Mode Steps", doze_steps, help="😴 Energy saving")
            with col4:
                efficiency = (doze_steps / len(steps) * 100) if len(steps) > 0 else 0
                st.metric("Doze Efficiency", f"{efficiency:.1f}%")
            
            # Detailed steps
            st.subheader("Step-by-Step Access Protocol")
            
            for step in steps:
                if step.mode == 'active':
                    icon = "🔋"
                    color = "green"
                    bg_color = "#e6ffe6"
                else:
                    icon = "😴"
                    color = "blue"
                    bg_color = "#e6e6ff"
                
                with st.expander(f"Step {step.step_num}: {step.action} {icon}", expanded=True):
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">
                    <b>Position:</b> {step.position}<br>
                    <b>Bucket:</b> {step.bucket_accessed}<br>
                    <b>Mode:</b> <span style="color: {color}; font-weight: bold;">{step.mode.upper()}</span><br>
                    <b>Details:</b> {step.details}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visual timeline table
            st.subheader("Access Timeline Summary")
            timeline_data = []
            for step in steps:
                timeline_data.append({
                    'Step': step.step_num,
                    'Action': step.action,
                    'Position': step.position,
                    'Bucket': step.bucket_accessed,
                    'Mode': f"{'🔋' if step.mode == 'active' else '😴'} {step.mode.upper()}"
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Click the button above to run the access simulation with the current parameters.")
    
    # Tab 4: Metrics
    with tab4:
        st.header("Performance Metrics")
        
        metrics = calculate_metrics(tree, di)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Distributed Indexing Metrics")
            st.metric("Access Time (buckets)", f"{metrics['access_time']:.2f}")
            st.metric("Tuning Time (buckets)", f"{metrics['tuning_time']}")
            st.metric("Probe Wait", f"{metrics['probe_wait']:.2f}")
            st.metric("Beast Wait", f"{metrics['beast_wait']:.2f}")
            st.metric("Schedule Length", metrics['schedule_length'])
            st.metric("Index Size", metrics['index_size'])
            st.metric("Replication Overhead", metrics['replication_overhead'])
        
        with col2:
            st.subheader("⚖️ Comparison with Optimal Methods")
            
            st.markdown("**Access-Optimal (No Index)**")
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Access Time", f"{metrics['access_opt_time']:.2f}")
            with col2b:
                st.metric("Tuning Time", f"{metrics['access_opt_tune']:.2f}")
            
            st.markdown("**Tune-Optimal (Index at Start)**")
            col2c, col2d = st.columns(2)
            with col2c:
                st.metric("Access Time", f"{metrics['tune_opt_access']:.2f}")
            with col2d:
                st.metric("Tuning Time", f"{metrics['tune_opt_tune']}")
        
        st.subheader("📊 Analysis")
        
        # Calculate improvements
        tune_improvement = ((metrics['access_opt_tune'] - metrics['tuning_time']) / 
                          metrics['access_opt_tune'] * 100) if metrics['access_opt_tune'] > 0 else 0
        access_vs_tune_opt = ((metrics['tune_opt_access'] - metrics['access_time']) / 
                            metrics['tune_opt_access'] * 100) if metrics['tune_opt_access'] > 0 else 0
        
        col3a, col3b = st.columns(2)
        with col3a:
            st.success(f"**Tuning Time Reduction:** {tune_improvement:.1f}% vs Access-Optimal")
        with col3b:
            st.success(f"**Access Time Reduction:** {access_vs_tune_opt:.1f}% vs Tune-Optimal")
        
        st.markdown(f"""
        ### Key Findings:
        
        1. **Tuning Time Improvement**: Distributed indexing reduces tuning time by 
           **{tune_improvement:.1f}%** compared to access-optimal method (no indexing).
        
        2. **Access Time Improvement**: Distributed indexing improves access time by 
           **{access_vs_tune_opt:.1f}%** compared to tune-optimal method (index only at start).
        
        3. **Trade-off Balance**: The replication level r={r} provides an optimal balance between:
           - Minimizing probe wait (requires more replication)
           - Minimizing beast wait (requires less replication)
        
        4. **Energy Efficiency**: With tuning time of only **{metrics['tuning_time']}** buckets,
           the client spends most time in doze mode, conserving battery power significantly.
        """)
        
        # Formulas section
        with st.expander("📐 Formulas Used"):
            st.markdown("**Number of Index Tree Levels:**")
            st.latex(r"k = \lceil \log_n(Data) \rceil")
            
            st.markdown("**Optimal Replication Level:**")
            st.latex(r"r^* = \left\lfloor \frac{1}{2} \cdot \left( \log_n \left( \frac{Data \cdot (n-1) + n^{k+1}}{n-1} \right) - 1 \right) \right\rfloor + 1")
            
            st.markdown("**Tuning Time Upper Bound:**")
            st.latex(r"TuningTime \leq \lceil \log_n(Data) \rceil + 3")
            
            st.markdown("**Access Time:**")
            st.latex(r"AccessTime = ProbeWait + BeastWait")


if __name__ == "__main__":
    main()
