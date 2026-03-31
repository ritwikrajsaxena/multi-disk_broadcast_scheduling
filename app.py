import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import gcd
from functools import reduce
from collections import defaultdict, deque
import random
import time

# ==================== UTILITY FUNCTIONS ====================

def lcm(a, b):
    """Calculate Least Common Multiple of two numbers"""
    return abs(a * b) // gcd(a, b)

def lcm_multiple(numbers):
    """Calculate LCM of multiple numbers"""
    return reduce(lcm, numbers, 1)

# ==================== BROADCAST DISK SCHEDULER ====================

class BroadcastDiskScheduler:
    """
    Implementation of the Multi-Disk Broadcast Scheduling Algorithm
    Based on: "Broadcast Disks: Data Management for Asymmetric Communication"
    by Acharya, Alonso, Franklin, and Zdonik
    """
    
    def __init__(self):
        self.disks = {}  # {disk_id: [(item, probability), ...]}
        self.relative_frequencies = {}  # {disk_id: frequency}
        self.chunks = {}  # {disk_id: [[items in chunk], ...]}
        self.schedule = []
        self.cycle_length = 0
        
    def add_disk(self, disk_id, items_with_probs, relative_frequency):
        """
        Add a disk to the scheduler
        
        Args:
            disk_id: Unique identifier for the disk
            items_with_probs: List of tuples [(item_name, access_probability), ...]
            relative_frequency: How often this disk should be broadcast relative to others
        """
        self.disks[disk_id] = items_with_probs
        self.relative_frequencies[disk_id] = relative_frequency
        
    def calculate_chunks(self):
        """
        Divide each disk into chunks based on relative frequencies
        Uses LCM to determine number of chunks per disk
        """
        if not self.disks:
            return {}
        
        # Calculate LCM of all frequencies
        freqs = list(self.relative_frequencies.values())
        total_lcm = lcm_multiple(freqs)
        
        self.chunks = {}
        
        for disk_id, items_with_probs in self.disks.items():
            freq = self.relative_frequencies[disk_id]
            num_chunks = total_lcm // freq
            
            items = [item[0] for item in items_with_probs]
            
            # Distribute items across chunks
            disk_chunks = []
            
            if not items:
                disk_chunks = [[] for _ in range(num_chunks)]
            else:
                for i in range(num_chunks):
                    start_idx = (i * len(items)) // num_chunks
                    end_idx = ((i + 1) * len(items)) // num_chunks
                    
                    if start_idx < end_idx:
                        chunk_items = items[start_idx:end_idx]
                    else:
                        # Handle case where items don't divide evenly
                        chunk_items = [items[i % len(items)]]
                    
                    disk_chunks.append(chunk_items)
            
            self.chunks[disk_id] = disk_chunks
        
        return self.chunks
    
    def generate_schedule(self):
        """
        Generate the broadcast schedule by interleaving chunks from all disks
        Returns a list of broadcast entries
        """
        if not self.chunks:
            self.calculate_chunks()
        
        if not self.disks:
            return []
        
        freqs = list(self.relative_frequencies.values())
        total_lcm = lcm_multiple(freqs)
        
        self.schedule = []
        slot = 0
        
        # Track which chunk to use next for each disk
        chunk_counters = {disk_id: 0 for disk_id in self.disks}
        
        # Generate minor cycles
        for minor_cycle in range(total_lcm):
            for disk_id in sorted(self.disks.keys()):
                freq = self.relative_frequencies[disk_id]
                interval = total_lcm // freq
                
                # Check if this disk broadcasts in this minor cycle
                if minor_cycle % interval == 0:
                    num_chunks = len(self.chunks[disk_id])
                    chunk_idx = chunk_counters[disk_id] % num_chunks
                    chunk_items = self.chunks[disk_id][chunk_idx]
                    
                    for item in chunk_items:
                        self.schedule.append({
                            'slot': slot,
                            'minor_cycle': minor_cycle + 1,
                            'disk_id': disk_id,
                            'chunk_idx': chunk_idx + 1,
                            'item': item
                        })
                        slot += 1
                    
                    chunk_counters[disk_id] += 1
        
        self.cycle_length = slot
        return self.schedule
    
    def get_item_statistics(self):
        """Calculate statistics for each item in the schedule"""
        if not self.schedule:
            self.generate_schedule()
        
        item_stats = {}
        
        # Count occurrences and find positions
        item_positions = defaultdict(list)
        for entry in self.schedule:
            item_positions[entry['item']].append(entry['slot'])
        
        # Calculate statistics for each item
        for disk_id, items_with_probs in self.disks.items():
            for item, prob in items_with_probs:
                positions = item_positions[item]
                frequency = len(positions)
                
                if frequency > 0:
                    inter_arrival = self.cycle_length / frequency
                    expected_delay = inter_arrival / 2
                else:
                    inter_arrival = float('inf')
                    expected_delay = float('inf')
                
                item_stats[item] = {
                    'disk': disk_id,
                    'access_probability': prob,
                    'broadcast_frequency': frequency,
                    'inter_arrival_time': inter_arrival,
                    'expected_delay': expected_delay,
                    'positions': positions
                }
        
        return item_stats
    
    def calculate_average_expected_delay(self):
        """Calculate the weighted average expected delay"""
        stats = self.get_item_statistics()
        
        total_delay = 0
        total_prob = 0
        
        for item, item_stats in stats.items():
            prob = item_stats['access_probability']
            delay = item_stats['expected_delay']
            
            if delay != float('inf'):
                total_delay += prob * delay
                total_prob += prob
        
        if total_prob > 0:
            return total_delay / total_prob * total_prob  # Weighted average
        return 0
    
    def calculate_flat_broadcast_delay(self):
        """Calculate expected delay for a flat (uniform) broadcast"""
        all_items = []
        for disk_id, items_with_probs in self.disks.items():
            all_items.extend(items_with_probs)
        
        if not all_items:
            return 0
        
        n_items = len(all_items)
        # In flat broadcast, each item appears once per cycle
        # Expected delay = n_items / 2
        
        total_delay = 0
        for item, prob in all_items:
            total_delay += prob * (n_items / 2)
        
        return total_delay


# ==================== CACHE SIMULATOR ====================

class CacheSimulator:
    """Simulate different cache replacement policies"""
    
    def __init__(self, cache_size, schedule, item_stats):
        self.cache_size = cache_size
        self.schedule = schedule
        self.item_stats = item_stats
        self.cache = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_delay': 0
        }
    
    def generate_requests(self, num_requests):
        """Generate random requests based on access probabilities"""
        items = list(self.item_stats.keys())
        probs = [self.item_stats[item]['access_probability'] for item in items]
        
        # Normalize probabilities
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1 / len(items)] * len(items)
        
        requests = np.random.choice(items, size=num_requests, p=probs)
        return list(requests)
    
    def simulate_lru(self, requests):
        """Simulate LRU cache replacement"""
        cache = {}  # item -> last_access_time
        cache_order = deque()  # Order of items (for LRU)
        
        stats = {'hits': 0, 'misses': 0, 'total_delay': 0}
        
        for t, item in enumerate(requests):
            if item in cache:
                # Cache hit
                stats['hits'] += 1
                cache_order.remove(item)
                cache_order.append(item)
                cache[item] = t
            else:
                # Cache miss
                stats['misses'] += 1
                stats['total_delay'] += self.item_stats[item]['expected_delay']
                
                # Add to cache
                if len(cache) >= self.cache_size:
                    # Evict LRU item
                    lru_item = cache_order.popleft()
                    del cache[lru_item]
                
                cache[item] = t
                cache_order.append(item)
        
        return stats
    
    def simulate_pix(self, requests):
        """
        Simulate PIX (Probability / Inverse Broadcast Frequency) cache replacement
        Evicts item with lowest P/X value
        """
        cache = {}  # item -> PIX value
        
        stats = {'hits': 0, 'misses': 0, 'total_delay': 0}
        
        for item in requests:
            if item in cache:
                stats['hits'] += 1
            else:
                stats['misses'] += 1
                stats['total_delay'] += self.item_stats[item]['expected_delay']
                
                # Calculate PIX value for new item
                prob = self.item_stats[item]['access_probability']
                freq = self.item_stats[item]['broadcast_frequency']
                pix_value = prob / freq if freq > 0 else 0
                
                if len(cache) >= self.cache_size:
                    # Evict item with lowest PIX value
                    min_item = min(cache.keys(), key=lambda x: cache[x])
                    del cache[min_item]
                
                cache[item] = pix_value
        
        return stats
    
    def simulate_lix(self, requests, history_window=10):
        """
        Simulate LIX cache replacement
        Uses recent access history to estimate probability
        """
        cache = {}  # item -> estimated LIX value
        access_history = defaultdict(list)  # item -> list of access times
        
        stats = {'hits': 0, 'misses': 0, 'total_delay': 0}
        
        for t, item in enumerate(requests):
            # Update access history
            access_history[item].append(t)
            if len(access_history[item]) > history_window:
                access_history[item] = access_history[item][-history_window:]
            
            if item in cache:
                stats['hits'] += 1
                # Update LIX value
                estimated_prob = len(access_history[item]) / history_window
                freq = self.item_stats[item]['broadcast_frequency']
                cache[item] = estimated_prob / freq if freq > 0 else 0
            else:
                stats['misses'] += 1
                stats['total_delay'] += self.item_stats[item]['expected_delay']
                
                # Calculate LIX value
                estimated_prob = len(access_history[item]) / history_window
                freq = self.item_stats[item]['broadcast_frequency']
                lix_value = estimated_prob / freq if freq > 0 else 0
                
                if len(cache) >= self.cache_size:
                    # Evict item with lowest LIX value
                    min_item = min(cache.keys(), key=lambda x: cache[x])
                    del cache[min_item]
                
                cache[item] = lix_value
        
        return stats


# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="Broadcast Disk Scheduler",
        page_icon="📡",
        layout="wide"
    )
    
    st.title("📡 Broadcast Disk Scheduling System")
    st.markdown("""
    Implementation of the **Multi-Disk Broadcast Scheduling Algorithm** based on the paper:
    *"Broadcast Disks: Data Management for Asymmetric Communication"* by Acharya et al.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Number of disks
    num_disks = st.sidebar.number_input(
        "Number of Disks",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of virtual disks (different broadcast frequencies)"
    )
    
    # Mode selection
    input_mode = st.sidebar.radio(
        "Input Mode",
        ["Manual Entry", "Example Data", "Random Generation"],
        help="Choose how to input data items and probabilities"
    )
    
    # Initialize scheduler
    scheduler = BroadcastDiskScheduler()
    
    # ==================== INPUT SECTION ====================
    st.header("📥 Data Input")
    
    if input_mode == "Example Data":
        # Use example from the paper
        st.info("Using example data similar to the paper (7 items across 3 disks)")
        
        # Disk 1: Fastest (Hot data)
        scheduler.add_disk(
            disk_id=1,
            items_with_probs=[('a', 0.35)],
            relative_frequency=4
        )
        
        # Disk 2: Medium
        scheduler.add_disk(
            disk_id=2,
            items_with_probs=[('b', 0.20), ('c', 0.15)],
            relative_frequency=2
        )
        
        # Disk 3: Slowest (Cold data)
        scheduler.add_disk(
            disk_id=3,
            items_with_probs=[('d', 0.12), ('e', 0.10), ('f', 0.05), ('g', 0.03)],
            relative_frequency=1
        )
        
        # Display example data
        example_data = []
        for disk_id, items in scheduler.disks.items():
            for item, prob in items:
                example_data.append({
                    'Disk': disk_id,
                    'Item': item,
                    'Access Probability': prob,
                    'Relative Frequency': scheduler.relative_frequencies[disk_id]
                })
        
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)
    
    elif input_mode == "Random Generation":
        st.info("Generating random data items with Zipf-distributed access probabilities")
        
        col1, col2 = st.columns(2)
        with col1:
            total_items = st.number_input("Total Items", min_value=3, max_value=50, value=10)
        with col2:
            zipf_alpha = st.slider("Zipf Skewness (α)", min_value=0.5, max_value=2.0, value=1.0)
        
        # Generate Zipf distribution
        ranks = np.arange(1, total_items + 1)
        probs = 1 / (ranks ** zipf_alpha)
        probs = probs / probs.sum()
        
        # Distribute items across disks
        items_per_disk = total_items // num_disks
        remainder = total_items % num_disks
        
        item_idx = 0
        for disk_id in range(1, num_disks + 1):
            # Calculate items for this disk
            n_items = items_per_disk + (1 if disk_id <= remainder else 0)
            
            # Get items and probabilities
            disk_items = []
            for _ in range(n_items):
                if item_idx < total_items:
                    item_name = chr(ord('a') + item_idx) if item_idx < 26 else f"item_{item_idx}"
                    disk_items.append((item_name, float(probs[item_idx])))
                    item_idx += 1
            
            # Faster disks have higher frequency
            freq = num_disks - disk_id + 1
            
            scheduler.add_disk(
                disk_id=disk_id,
                items_with_probs=disk_items,
                relative_frequency=freq
            )
        
        # Display generated data
        gen_data = []
        for disk_id, items in scheduler.disks.items():
            for item, prob in items:
                gen_data.append({
                    'Disk': disk_id,
                    'Item': item,
                    'Access Probability': f"{prob:.4f}",
                    'Relative Frequency': scheduler.relative_frequencies[disk_id]
                })
        
        st.dataframe(pd.DataFrame(gen_data), use_container_width=True)
    
    else:  # Manual Entry
        st.info("Enter data items and access probabilities for each disk")
        
        tabs = st.tabs([f"Disk {i+1}" for i in range(num_disks)])
        
        for disk_idx, tab in enumerate(tabs):
            with tab:
                disk_id = disk_idx + 1
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Items input
                    items_input = st.text_area(
                        f"Items for Disk {disk_id}",
                        value="a, b, c" if disk_id == 1 else "",
                        help="Enter comma-separated item names",
                        key=f"items_{disk_id}"
                    )
                    
                    probs_input = st.text_area(
                        f"Access Probabilities",
                        value="0.3, 0.2, 0.1" if disk_id == 1 else "",
                        help="Enter comma-separated probabilities (should match number of items)",
                        key=f"probs_{disk_id}"
                    )
                
                with col2:
                    rel_freq = st.number_input(
                        "Relative Frequency",
                        min_value=1,
                        max_value=20,
                        value=num_disks - disk_idx,
                        help="Higher = more frequent broadcast",
                        key=f"freq_{disk_id}"
                    )
                
                # Parse inputs
                try:
                    items = [x.strip() for x in items_input.split(',') if x.strip()]
                    probs = [float(x.strip()) for x in probs_input.split(',') if x.strip()]
                    
                    if items and probs and len(items) == len(probs):
                        items_with_probs = list(zip(items, probs))
                        scheduler.add_disk(
                            disk_id=disk_id,
                            items_with_probs=items_with_probs,
                            relative_frequency=rel_freq
                        )
                    elif items or probs:
                        st.warning(f"⚠️ Disk {disk_id}: Number of items ({len(items)}) must match probabilities ({len(probs)})")
                except ValueError as e:
                    st.error(f"Error parsing input: {e}")
    
    # ==================== SCHEDULE GENERATION ====================
    if scheduler.disks:
        st.header("📊 Broadcast Schedule Generation")
        
        if st.button("🚀 Generate Broadcast Schedule", type="primary"):
            
            with st.spinner("Calculating chunks and generating schedule..."):
                # Calculate chunks
                chunks = scheduler.calculate_chunks()
                
                # Generate schedule
                schedule = scheduler.generate_schedule()
                
                # Get statistics
                item_stats = scheduler.get_item_statistics()
            
            # ==================== CHUNKS VISUALIZATION ====================
            st.subheader("📦 Chunks per Disk")
            
            chunks_data = []
            for disk_id, disk_chunks in chunks.items():
                for chunk_idx, chunk_items in enumerate(disk_chunks):
                    chunks_data.append({
                        'Disk': f"D{disk_id}",
                        'Chunk': f"C{disk_id},{chunk_idx+1}",
                        'Items': ', '.join(chunk_items) if chunk_items else '(empty)',
                        'Frequency': scheduler.relative_frequencies[disk_id]
                    })
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(pd.DataFrame(chunks_data), use_container_width=True)
            
            with col2:
                # Disk visualization
                fig_disks = go.Figure()
                
                colors = px.colors.qualitative.Set2
                
                for idx, (disk_id, disk_chunks) in enumerate(chunks.items()):
                    items_text = []
                    for chunk in disk_chunks:
                        items_text.append(', '.join(chunk) if chunk else '∅')
                    
                    fig_disks.add_trace(go.Bar(
                        name=f"Disk {disk_id} (freq={scheduler.relative_frequencies[disk_id]})",
                        x=[f"C{disk_id},{i+1}" for i in range(len(disk_chunks))],
                        y=[1] * len(disk_chunks),
                        text=items_text,
                        textposition='inside',
                        marker_color=colors[idx % len(colors)]
                    ))
                
                fig_disks.update_layout(
                    title="Disk Structure",
                    barmode='group',
                    showlegend=True,
                    height=300
                )
                
                st.plotly_chart(fig_disks, use_container_width=True)
            
            # ==================== BROADCAST SCHEDULE ====================
            st.subheader("📻 Complete Broadcast Cycle")
            
            # Convert schedule to DataFrame
            schedule_df = pd.DataFrame(schedule)
            schedule_df.columns = ['Slot', 'Minor Cycle', 'Disk', 'Chunk', 'Item']
            
            # Display options
            col1, col2 = st.columns(2)
            with col1:
                view_option = st.radio(
                    "View",
                    ["Table", "Timeline", "Sequence"],
                    horizontal=True
                )
            with col2:
                st.metric("Cycle Length", f"{scheduler.cycle_length} slots")
            
            if view_option == "Table":
                st.dataframe(schedule_df, use_container_width=True, height=400)
            
            elif view_option == "Timeline":
                # Create timeline visualization
                fig_timeline = go.Figure()
                
                disk_colors = {disk_id: colors[idx % len(colors)] 
                              for idx, disk_id in enumerate(scheduler.disks.keys())}
                
                for _, row in schedule_df.iterrows():
                    fig_timeline.add_trace(go.Bar(
                        x=[1],
                        y=[row['Item']],
                        orientation='h',
                        name=f"Disk {row['Disk']}",
                        marker_color=disk_colors[row['Disk']],
                        text=f"Slot {row['Slot']}",
                        hovertemplate=f"Item: {row['Item']}<br>Slot: {row['Slot']}<br>Disk: {row['Disk']}<br>Chunk: {row['Chunk']}<extra></extra>",
                        showlegend=False
                    ))
                
                fig_timeline.update_layout(
                    title="Broadcast Timeline",
                    barmode='stack',
                    height=max(400, len(schedule_df) * 25),
                    xaxis_title="Broadcast Slot",
                    yaxis_title="Item"
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            else:  # Sequence
                # Visual sequence representation
                st.write("**Broadcast Sequence:**")
                
                sequence_items = schedule_df['Item'].tolist()
                sequence_disks = schedule_df['Disk'].tolist()
                
                # Create colored sequence
                sequence_html = ""
                for item, disk in zip(sequence_items, sequence_disks):
                    color = colors[disk - 1]
                    sequence_html += f'<span style="background-color:{color}; padding:5px; margin:2px; border-radius:5px; display:inline-block;">{item}</span>'
                
                st.markdown(sequence_html, unsafe_allow_html=True)
                
                # Show pattern
                st.write("")
                st.write("**Legend:**")
                for disk_id in scheduler.disks.keys():
                    color = colors[disk_id - 1]
                    st.markdown(
                        f'<span style="background-color:{color}; padding:5px; border-radius:5px;">Disk {disk_id} (frequency: {scheduler.relative_frequencies[disk_id]})</span>',
                        unsafe_allow_html=True
                    )
            
            # ==================== ITEM STATISTICS ====================
            st.subheader("📈 Item Statistics")
            
            stats_data = []
            for item, stats in item_stats.items():
                stats_data.append({
                    'Item': item,
                    'Disk': stats['disk'],
                    'Access Prob.': f"{stats['access_probability']:.4f}",
                    'Broadcast Freq.': stats['broadcast_frequency'],
                    'Inter-arrival Time': f"{stats['inter_arrival_time']:.2f}",
                    'Expected Delay': f"{stats['expected_delay']:.2f}",
                    'Positions in Cycle': str(stats['positions'])
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            # ==================== PERFORMANCE COMPARISON ====================
            st.subheader("⚡ Performance Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            multi_disk_delay = scheduler.calculate_average_expected_delay()
            flat_delay = scheduler.calculate_flat_broadcast_delay()
            improvement = ((flat_delay - multi_disk_delay) / flat_delay * 100) if flat_delay > 0 else 0
            
            with col1:
                st.metric(
                    "Multi-Disk Expected Delay",
                    f"{multi_disk_delay:.3f} slots"
                )
            
            with col2:
                st.metric(
                    "Flat Broadcast Delay",
                    f"{flat_delay:.3f} slots"
                )
            
            with col3:
                st.metric(
                    "Improvement",
                    f"{improvement:.1f}%",
                    delta=f"{flat_delay - multi_disk_delay:.3f} slots saved"
                )
            
            # Delay comparison chart
            fig_delay = go.Figure()
            
            items = list(item_stats.keys())
            multi_disk_delays = [item_stats[item]['expected_delay'] for item in items]
            flat_delays = [scheduler.cycle_length / 2] * len(items)  # Same for all in flat
            
            fig_delay.add_trace(go.Bar(
                name='Multi-Disk',
                x=items,
                y=multi_disk_delays,
                marker_color='#2ecc71'
            ))
            
            fig_delay.add_trace(go.Bar(
                name='Flat Broadcast',
                x=items,
                y=flat_delays,
                marker_color='#e74c3c'
            ))
            
            fig_delay.update_layout(
                title="Expected Delay Comparison by Item",
                xaxis_title="Data Item",
                yaxis_title="Expected Delay (slots)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_delay, use_container_width=True)
            
            # ==================== CACHE SIMULATION ====================
            st.subheader("💾 Cache Policy Simulation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cache_size = st.number_input(
                    "Cache Size",
                    min_value=1,
                    max_value=len(item_stats),
                    value=min(3, len(item_stats)),
                    help="Number of items the cache can hold"
                )
            
            with col2:
                num_requests = st.number_input(
                    "Number of Requests",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100
                )
            
            with col3:
                if st.button("🔄 Run Simulation"):
                    simulator = CacheSimulator(cache_size, schedule, item_stats)
                    requests = simulator.generate_requests(num_requests)
                    
                    # Run all policies
                    lru_stats = simulator.simulate_lru(requests)
                    pix_stats = simulator.simulate_pix(requests)
                    lix_stats = simulator.simulate_lix(requests)
                    
                    # Store in session state
                    st.session_state['cache_results'] = {
                        'LRU': lru_stats,
                        'PIX': pix_stats,
                        'LIX': lix_stats
                    }
            
            if 'cache_results' in st.session_state:
                results = st.session_state['cache_results']
                
                # Results table
                cache_results_df = pd.DataFrame([
                    {
                        'Policy': policy,
                        'Hits': stats['hits'],
                        'Misses': stats['misses'],
                        'Hit Rate': f"{stats['hits'] / (stats['hits'] + stats['misses']) * 100:.2f}%",
                        'Total Delay': f"{stats['total_delay']:.2f}",
                        'Avg Delay/Miss': f"{stats['total_delay'] / stats['misses']:.2f}" if stats['misses'] > 0 else "N/A"
                    }
                    for policy, stats in results.items()
                ])
                
                st.dataframe(cache_results_df, use_container_width=True)
                
                # Visualization
                fig_cache = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Hit Rate Comparison', 'Total Delay Comparison')
                )
                
                policies = list(results.keys())
                hit_rates = [results[p]['hits'] / (results[p]['hits'] + results[p]['misses']) * 100 
                            for p in policies]
                total_delays = [results[p]['total_delay'] for p in policies]
                
                fig_cache.add_trace(
                    go.Bar(x=policies, y=hit_rates, marker_color=['#3498db', '#2ecc71', '#9b59b6']),
                    row=1, col=1
                )
                
                fig_cache.add_trace(
                    go.Bar(x=policies, y=total_delays, marker_color=['#3498db', '#2ecc71', '#9b59b6']),
                    row=1, col=2
                )
                
                fig_cache.update_layout(height=400, showlegend=False)
                fig_cache.update_yaxes(title_text="Hit Rate (%)", row=1, col=1)
                fig_cache.update_yaxes(title_text="Total Delay (slots)", row=1, col=2)
                
                st.plotly_chart(fig_cache, use_container_width=True)
            
            # ==================== DOWNLOAD OPTIONS ====================
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = schedule_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Schedule (CSV)",
                    data=csv,
                    file_name="broadcast_schedule.csv",
                    mime="text/csv"
                )
            
            with col2:
                stats_df = pd.DataFrame(stats_data)
                csv_stats = stats_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Statistics (CSV)",
                    data=csv_stats,
                    file_name="item_statistics.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Create summary report
                report = f"""
BROADCAST DISK SCHEDULING REPORT
================================

Configuration:
- Number of Disks: {len(scheduler.disks)}
- Total Items: {sum(len(items) for items in scheduler.disks.values())}
- Cycle Length: {scheduler.cycle_length} slots

Disk Configuration:
"""
                for disk_id in scheduler.disks:
                    items = [item for item, _ in scheduler.disks[disk_id]]
                    report += f"- Disk {disk_id}: Items {items}, Frequency: {scheduler.relative_frequencies[disk_id]}\n"
                
                report += f"""
Performance:
- Multi-Disk Expected Delay: {multi_disk_delay:.3f} slots
- Flat Broadcast Delay: {flat_delay:.3f} slots
- Improvement: {improvement:.1f}%

Broadcast Sequence:
{' -> '.join(sequence_items)}
"""
                
                st.download_button(
                    label="📝 Download Report (TXT)",
                    data=report,
                    file_name="broadcast_report.txt",
                    mime="text/plain"
                )
    
    else:
        st.warning("⚠️ Please add at least one disk with items to generate a schedule.")
    
    # ==================== HELP SECTION ====================
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        ### Multi-Disk Broadcast Algorithm
        
        **Concept:**
        - Data items are assigned to virtual "disks" spinning at different speeds
        - Hot (frequently accessed) items go on faster disks
        - Cold (rarely accessed) items go on slower disks
        
        **Algorithm Steps:**
        1. **Assign items to disks** based on access probability
        2. **Set relative frequencies** for each disk
        3. **Calculate chunks** using LCM of frequencies
        4. **Interleave chunks** to create the broadcast cycle
        
        **Benefits:**
        - Reduces average wait time for popular items
        - Maintains periodic broadcast (no variance in inter-arrival times)
        - Allows power-saving through predictable scheduling
        
        **Cache Policies:**
        - **LRU**: Evicts least recently used (doesn't consider broadcast structure)
        - **PIX**: Evicts item with lowest P/X (probability / frequency)
        - **LIX**: Practical approximation of PIX using access history
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built based on *'Broadcast Disks: Data Management for Asymmetric Communication'* "
        "by Acharya, Alonso, Franklin, and Zdonik (SIGMOD '95)"
    )


if __name__ == "__main__":
    main()
