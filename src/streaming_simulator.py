import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import clear_output
import numpy as np
from datetime import datetime
import sys
import time

# Detect if running as standalone script
STANDALONE_MODE = __name__ == '__main__'

# If standalone, use interactive backend for separate windows
if STANDALONE_MODE:
    matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5
    print("🎯 Running in STANDALONE mode - plots will open in separate windows")
else:
    print("📓 Running in NOTEBOOK mode - plots will appear inline")

class StreamingSimulator:
    """
    Simulates real-time data streaming from manufacturing robot controller.
    Reads CSV data point-by-point, stores in database, and visualizes in real-time.
    
    Data format: Current measurements from 14 robot axes with timestamps
    """
    
    def __init__(self, csv_path, db_config):
        """
        Initialize the streaming simulator.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing robot controller data
        db_config : dict
            Database configuration with keys: host, database, user, password
        """
        self.csv_path = csv_path
        self.db_config = db_config
        self.df = None
        self.current_index = 0
        self.data_buffer = []
        self.axis_columns = []
        
        # Load CSV into memory
        self._load_csv()
        
        # Initialize database table
        self._init_database()
        
    def _load_csv(self):
        """Load CSV file into pandas DataFrame"""
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # Identify axis columns
            self.axis_columns = [col for col in self.df.columns if col.startswith('Axis #')]
            
            # Convert Time to datetime
            self.df['Time'] = pd.to_datetime(self.df['Time'])
            
            print(f"✅ Loaded {len(self.df)} records from {self.csv_path}")
            print(f"📊 Columns: {list(self.df.columns)}")
            print(f"🔧 Detected {len(self.axis_columns)} robot axes")
            print(f"📅 Time range: {self.df['Time'].min()} to {self.df['Time'].max()}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise
    
    def _init_database(self):
        """Initialize database and create table if it doesn't exist"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create table with columns for all 14 axes
            create_table_query = """
            CREATE TABLE IF NOT EXISTS robot_data (
                id SERIAL PRIMARY KEY,
                trait VARCHAR(50),
                axis_1 FLOAT,
                axis_2 FLOAT,
                axis_3 FLOAT,
                axis_4 FLOAT,
                axis_5 FLOAT,
                axis_6 FLOAT,
                axis_7 FLOAT,
                axis_8 FLOAT,
                axis_9 FLOAT,
                axis_10 FLOAT,
                axis_11 FLOAT,
                axis_12 FLOAT,
                axis_13 FLOAT,
                axis_14 FLOAT,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            cursor.execute(create_table_query)
            
            # Create index on timestamp for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_robot_data_timestamp 
                ON robot_data(timestamp);
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            print("✅ Database table initialized")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            raise
    
    def nextDataPoint(self, visualize=True, plot_style='combined'):
        """
        Read next data point from CSV, insert into database, and visualize.
        
        Parameters:
        -----------
        visualize : bool
            Whether to display visualization (default: True)
        plot_style : str
            'combined' - All axes on one graph (default)
            'separate' - Individual subplots for each axis
        
        Returns:
        --------
        dict : Current data point or None if end of data
        """
        if self.current_index >= len(self.df):
            print("⚠️ End of data stream reached")
            return None
        
        # Get current record
        record = self.df.iloc[self.current_index]
        
        # Insert into database
        self._insert_to_db(record)
        
        # Add to buffer for visualization
        self.data_buffer.append(record)
        
        # Visualize
        if visualize:
            if plot_style == 'combined':
                self._plot_dashboard(record)
            elif plot_style == 'separate':
                self._plot_dashboard_separate(record)
            else:
                print(f"⚠️ Unknown plot_style '{plot_style}'. Using 'combined'.")
                self._plot_dashboard(record)
        
        # Increment index
        self.current_index += 1

        return record.to_dict()

    def streamBatch(self, num_points, batch_size=500, visualize_every=1000, plot_style='combined'):
        """
        Stream multiple data points efficiently using batch database inserts.

        This is ~100x faster than nextDataPoint() for large datasets.

        Parameters:
        -----------
        num_points : int
            Number of data points to stream
        batch_size : int
            Number of records to insert per database batch (default: 500)
        visualize_every : int
            Visualize every Nth point (default: 1000, use 0 to disable)
        plot_style : str
            'combined' or 'separate' visualization style

        Returns:
        --------
        dict : Summary statistics of the streaming operation
        """
        start_time = time.time()
        points_to_process = min(num_points, len(self.df) - self.current_index)

        if points_to_process <= 0:
            print("⚠️ No more data to stream")
            return None

        print(f"🚀 Batch streaming {points_to_process:,} data points...")
        print(f"   Batch size: {batch_size}, Visualize every: {visualize_every}")

        # Timing accumulators for diagnostics
        time_db = 0
        time_viz = 0
        time_prep = 0

        # Open a single database connection for all inserts
        t0 = time.time()
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        print(f"   DB connection established in {time.time() - t0:.2f}s")

        points_processed = 0
        batches_inserted = 0
        current_batch = []

        # Pre-extract data using numpy arrays (FASTEST approach)
        t0 = time.time()
        end_idx = self.current_index + points_to_process
        slice_df = self.df.iloc[self.current_index:end_idx]

        # Extract columns as numpy arrays
        axis_cols = ['Axis #1', 'Axis #2', 'Axis #3', 'Axis #4', 'Axis #5', 'Axis #6', 'Axis #7',
                     'Axis #8', 'Axis #9', 'Axis #10', 'Axis #11', 'Axis #12', 'Axis #13', 'Axis #14']

        # Get trait values (or default)
        traits = slice_df['Trait'].values if 'Trait' in slice_df.columns else ['current'] * len(slice_df)
        times = slice_df['Time'].values

        # Get axis values as numpy arrays
        axis_arrays = []
        for col in axis_cols:
            if col in slice_df.columns:
                axis_arrays.append(slice_df[col].values)
            else:
                axis_arrays.append(np.full(len(slice_df), np.nan))

        # Build all values tuples
        all_values = []
        for i in range(len(slice_df)):
            values = [traits[i] if isinstance(traits, np.ndarray) else traits]
            for arr in axis_arrays:
                val = arr[i]
                values.append(float(val) if pd.notna(val) else None)
            values.append(times[i])
            all_values.append(tuple(values))

        time_prep = time.time() - t0
        print(f"   Data prepared in {time_prep:.2f}s ({len(all_values):,} records)")

        try:
            # Insert in batches
            for batch_start in range(0, len(all_values), batch_size):
                batch_end = min(batch_start + batch_size, len(all_values))
                current_batch = all_values[batch_start:batch_end]

                t0 = time.time()
                self._insert_batch(cursor, current_batch)
                conn.commit()
                time_db += time.time() - t0

                batches_inserted += 1
                points_processed = batch_end

                # Progress update every 5 batches
                if batches_inserted % 5 == 0:
                    elapsed = time.time() - start_time
                    rate = points_processed / elapsed if elapsed > 0 else 0
                    remaining = (points_to_process - points_processed) / rate if rate > 0 else 0
                    print(f"   ✓ {points_processed:,}/{points_to_process:,} ({points_processed/points_to_process*100:.1f}%) "
                          f"- {rate:.0f} pts/sec - ETA: {remaining:.1f}s")

            # Add all records to buffer at once (for visualization/analysis)
            self.data_buffer.extend(slice_df.to_dict('records'))

            # Update current index
            self.current_index += points_processed

        finally:
            cursor.close()
            conn.close()

        elapsed_time = time.time() - start_time
        rate = points_processed / elapsed_time if elapsed_time > 0 else 0

        # Final visualization (only once at the end)
        if visualize_every > 0 and len(self.data_buffer) > 0:
            t0 = time.time()
            record = self.df.iloc[self.current_index - 1]
            if plot_style == 'combined':
                self._plot_dashboard_fast(record)
            else:
                self._plot_dashboard_separate(record)
            time_viz = time.time() - t0

        summary = {
            'points_processed': points_processed,
            'batches_inserted': batches_inserted,
            'elapsed_seconds': elapsed_time,
            'points_per_second': rate,
            'current_index': self.current_index,
            'buffer_size': len(self.data_buffer),
            'time_breakdown': {
                'data_prep': time_prep,
                'database': time_db,
                'visualization': time_viz
            }
        }

        print(f"\n✅ Batch streaming complete!")
        print(f"   Points processed: {points_processed:,}")
        print(f"   Time elapsed: {elapsed_time:.2f} seconds")
        print(f"   Rate: {rate:.0f} points/second")
        print(f"   Database batches: {batches_inserted}")
        print(f"\n   ⏱️ Time breakdown:")
        print(f"      Data prep:     {time_prep:.2f}s")
        print(f"      Database:      {time_db:.2f}s")
        print(f"      Visualization: {time_viz:.2f}s")

        return summary

    def _plot_dashboard_fast(self, current_record):
        """Lightweight visualization for batch streaming - uses sampling for large buffers"""
        if not STANDALONE_MODE:
            clear_output(wait=True)

        fig = plt.figure(figsize=(16, 8))

        # Sample buffer if too large (max 2000 points for visualization)
        buffer_size = len(self.data_buffer)
        if buffer_size > 2000:
            # Use last 2000 points only for visualization
            sample_indices = list(range(max(0, buffer_size - 2000), buffer_size))
            buffer_sample = [self.data_buffer[i] for i in sample_indices]
        else:
            buffer_sample = self.data_buffer

        buffer_df = pd.DataFrame(buffer_sample)

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                 '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']

        ax = fig.add_subplot(111)

        # Plot only first 8 axes for speed
        for idx, axis in enumerate(self.axis_columns[:8]):
            if axis in buffer_df.columns:
                ax.plot(buffer_df[axis].values, color=colors[idx], linewidth=1, alpha=0.7, label=axis)

        ax.set_title(f'Robot Current Monitoring - {self.current_index:,} points streamed', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index (last 2000 points)', fontsize=11)
        ax.set_ylabel('Current (A)', fontsize=11)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # Add progress text
        progress_pct = (self.current_index / len(self.df)) * 100
        ax.text(0.02, 0.98, f'Progress: {self.current_index:,}/{len(self.df):,} ({progress_pct:.1f}%)',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def _insert_batch(self, cursor, batch):
        """Insert a batch of records using execute_values for maximum speed"""
        insert_query = """
        INSERT INTO robot_data
        (trait, axis_1, axis_2, axis_3, axis_4, axis_5, axis_6, axis_7,
         axis_8, axis_9, axis_10, axis_11, axis_12, axis_13, axis_14, timestamp)
        VALUES %s
        """
        execute_values(cursor, insert_query, batch, page_size=len(batch))

    def _insert_to_db(self, record):
        """Insert a single record into the database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Prepare INSERT query
            insert_query = """
            INSERT INTO robot_data 
            (trait, axis_1, axis_2, axis_3, axis_4, axis_5, axis_6, axis_7, 
             axis_8, axis_9, axis_10, axis_11, axis_12, axis_13, axis_14, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Extract values - handle NaN values
            values = (
                record.get('Trait', 'current'),
                float(record['Axis #1']) if pd.notna(record.get('Axis #1')) else None,
                float(record['Axis #2']) if pd.notna(record.get('Axis #2')) else None,
                float(record['Axis #3']) if pd.notna(record.get('Axis #3')) else None,
                float(record['Axis #4']) if pd.notna(record.get('Axis #4')) else None,
                float(record['Axis #5']) if pd.notna(record.get('Axis #5')) else None,
                float(record['Axis #6']) if pd.notna(record.get('Axis #6')) else None,
                float(record['Axis #7']) if pd.notna(record.get('Axis #7')) else None,
                float(record['Axis #8']) if pd.notna(record.get('Axis #8')) else None,
                float(record['Axis #9']) if pd.notna(record.get('Axis #9')) else None,
                float(record['Axis #10']) if pd.notna(record.get('Axis #10')) else None,
                float(record['Axis #11']) if pd.notna(record.get('Axis #11')) else None,
                float(record['Axis #12']) if pd.notna(record.get('Axis #12')) else None,
                float(record['Axis #13']) if pd.notna(record.get('Axis #13')) else None,
                float(record['Axis #14']) if pd.notna(record.get('Axis #14')) else None,
                record['Time']
            )
            
            cursor.execute(insert_query, values)
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Database insert error at index {self.current_index}: {e}")
    
    def _plot_dashboard(self, current_record):
        """Real-time visualization dashboard - all axes on one graph"""
        # Only clear output in notebook mode
        if not STANDALONE_MODE:
            clear_output(wait=True)
        
        # In standalone mode, reuse the same figure
        if STANDALONE_MODE:
            plt.ion()  # Interactive mode
            if plt.fignum_exists(1):
                plt.figure(1)
                plt.clf()
            else:
                plt.figure(1, figsize=(18, 10))
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(18, 10))
        
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
        
        fig.suptitle('Robot Controller Current Monitoring - Real-Time Stream (All Axes)', 
                     fontsize=16, fontweight='bold')
        
        if len(self.data_buffer) > 0:
            buffer_df = pd.DataFrame(self.data_buffer)
            
            # Main plot - All axes on one graph
            ax_main = fig.add_subplot(gs[0])
            
            # Define colors for all 14 axes
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
                     '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
                     '#F06292', '#AED581', '#FFD54F', '#90CAF9',
                     '#FFAB91', '#CE93D8']
            
            # Use Time for x-axis
            time_data = buffer_df['Time']
            
            # Plot all axes
            lines = []
            labels = []
            for idx, axis in enumerate(self.axis_columns):
                if axis in buffer_df.columns:
                    axis_data = buffer_df[axis].dropna()
                    
                    if len(axis_data) > 0:
                        # Plot line using Time as x-axis
                        line, = ax_main.plot(time_data, buffer_df[axis], 
                                           color=colors[idx % len(colors)], 
                                           linewidth=2, alpha=0.7, 
                                           label=axis)
                        lines.append(line)
                        labels.append(axis)
                        
                        # Highlight current point
                        current_val = current_record.get(axis, 0)
                        if pd.notna(current_val) and current_val > 0:
                            ax_main.scatter(current_record['Time'], current_val, 
                                          color=colors[idx % len(colors)], 
                                          s=80, zorder=5, 
                                          edgecolors='black', linewidth=1.5)
            
            # Formatting
            ax_main.set_title('Current Draw - All Robot Axes', 
                            fontsize=14, fontweight='bold', pad=15)
            ax_main.set_xlabel('Time', fontsize=12)
            ax_main.set_ylabel('Current (Amperes)', fontsize=12)
            ax_main.grid(True, alpha=0.3, linestyle=':', linewidth=1)
            
            # Format x-axis to show time nicely
            import matplotlib.dates as mdates
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax_main.tick_params(axis='x', rotation=45)
            
            # Legend - split into two columns if many axes
            if len(labels) > 7:
                ax_main.legend(lines, labels, loc='upper left', 
                             fontsize=9, ncol=2, framealpha=0.9,
                             bbox_to_anchor=(0, 1), borderaxespad=0)
            else:
                ax_main.legend(lines, labels, loc='upper left', 
                             fontsize=9, framealpha=0.9)
            
            # Add reference line for total current
            total_current_series = buffer_df[self.axis_columns].sum(axis=1, skipna=True)
            ax_main_twin = ax_main.twinx()
            ax_main_twin.plot(time_data, total_current_series, 
                            color='black', linewidth=2.5, alpha=0.4, 
                            linestyle='--', label='Total Current')
            ax_main_twin.set_ylabel('Total Current (A)', fontsize=11, color='black')
            ax_main_twin.tick_params(axis='y', labelcolor='black')
            ax_main_twin.legend(loc='upper right', fontsize=9, framealpha=0.9)
            
            # Summary statistics panel
            ax_summary = fig.add_subplot(gs[1])
            ax_summary.axis('off')
            
            # Calculate detailed stats
            active_axes_count = sum(1 for ax in self.axis_columns 
                                  if ax in buffer_df.columns 
                                  and pd.notna(buffer_df[ax].iloc[-1]) 
                                  and buffer_df[ax].iloc[-1] > 0.1)
            
            total_current = total_current_series.iloc[-1]
            avg_total_current = total_current_series.mean()
            max_single_axis = max((buffer_df[ax].iloc[-1] for ax in self.axis_columns 
                                 if ax in buffer_df.columns and pd.notna(buffer_df[ax].iloc[-1])), 
                                default=0)
            
            # Find axis with max current
            max_axis = None
            max_axis_val = 0
            for ax in self.axis_columns:
                if ax in buffer_df.columns and pd.notna(buffer_df[ax].iloc[-1]):
                    val = buffer_df[ax].iloc[-1]
                    if val > max_axis_val:
                        max_axis_val = val
                        max_axis = ax
            
            summary_text = f"""
    📊 REAL-TIME STREAMING STATUS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Progress: {self.current_index + 1:,} / {len(self.df):,} records ({((self.current_index + 1) / len(self.df) * 100):.1f}%)  |  Timestamp: {current_record['Time']}
    
    Current Metrics:                              Statistical Summary:
      • Total Current:     {total_current:>6.2f} A          • Avg Total Current:  {avg_total_current:>6.2f} A
      • Max Axis Current:  {max_axis_val:>6.2f} A ({max_axis})    • Active Axes:       {active_axes_count:>2d} / {len(self.axis_columns)}
      • Buffer Size:       {len(buffer_df):>6,} pts         • Data Quality:       {(1 - buffer_df[self.axis_columns].isna().sum().sum() / (len(buffer_df) * len(self.axis_columns))) * 100:.1f}%
            """
            
            ax_summary.text(0.02, 0.5, summary_text, 
                          fontsize=9.5, family='monospace',
                          verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightblue', 
                                  alpha=0.3, edgecolor='steelblue', linewidth=2))
        
        plt.tight_layout()
        
        # Different display method for standalone vs notebook
        if STANDALONE_MODE:
            plt.draw()
            plt.pause(0.001)
            # Print progress to console
            print(f"\r📊 Streaming: {self.current_index + 1}/{len(self.df)} ({((self.current_index + 1) / len(self.df) * 100):.1f}%)", end='', flush=True)
        else:
            plt.show()
    
    def _plot_dashboard_separate(self, current_record):
        """Real-time visualization dashboard - separate subplots for each axis"""
        clear_output(wait=True)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Robot Controller Current Monitoring - Individual Axes View', 
                     fontsize=16, fontweight='bold')
        
        if len(self.data_buffer) > 0:
            buffer_df = pd.DataFrame(self.data_buffer)
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
                     '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
                     '#F06292', '#AED581', '#FFD54F', '#90CAF9',
                     '#FFAB91', '#CE93D8']
            
            # Use Time for x-axis
            time_data = buffer_df['Time']
            
            # Import for time formatting
            import matplotlib.dates as mdates
            
            # Plot each axis in separate subplot
            for idx, axis in enumerate(self.axis_columns):
                if idx >= 14:  # Only plot first 14 axes
                    break
                    
                row = idx // 4
                col = idx % 4
                ax = fig.add_subplot(gs[row, col])
                
                if axis in buffer_df.columns:
                    axis_data = buffer_df[axis].dropna()
                    
                    if len(axis_data) > 0:
                        # Plot line using Time as x-axis
                        ax.plot(time_data, buffer_df[axis], 
                               color=colors[idx], linewidth=2, alpha=0.7)
                        
                        # Highlight current point
                        current_val = current_record.get(axis, 0)
                        if pd.notna(current_val) and current_val > 0:
                            ax.scatter(current_record['Time'], current_val, 
                                      color='red', s=80, zorder=5, 
                                      edgecolors='black', linewidth=1.5)
                        
                        # Add mean line
                        mean_val = axis_data.mean()
                        if pd.notna(mean_val):
                            ax.axhline(mean_val, color='green', linestyle='--', 
                                      linewidth=1, alpha=0.5, 
                                      label=f'μ={mean_val:.2f}A')
                        
                        ax.set_title(f'{axis}', fontsize=10, fontweight='bold')
                        ax.set_xlabel('Time', fontsize=8)
                        ax.set_ylabel('Current (A)', fontsize=8)
                        ax.grid(True, alpha=0.3, linestyle=':')
                        ax.legend(fontsize=7, loc='upper right')
                        ax.tick_params(labelsize=7, axis='x', rotation=45)
                        
                        # Format x-axis
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        plt.tight_layout()
        plt.show()
    
    def get_all_data_from_db(self):
        """Retrieve all data from database for analysis"""
        try:
            conn = psycopg2.connect(**self.db_config)
            query = "SELECT * FROM robot_data ORDER BY timestamp"
            df = pd.read_sql(query, conn)
            conn.close()
            print(f"✅ Retrieved {len(df)} records from database")
            return df
        except Exception as e:
            print(f"❌ Error retrieving data: {e}")
            return None
    
    def detect_anomalies(self, threshold_multiplier=2.5):
        """
        Detect anomalies in robot axis currents using statistical methods.
        
        Parameters:
        -----------
        threshold_multiplier : float
            Number of standard deviations for anomaly detection
        
        Returns:
        --------
        dict : Dictionary of anomalies by axis
        """
        if len(self.data_buffer) < 30:
            print("⚠️ Not enough data for reliable anomaly detection (need 30+ points)")
            return {}
        
        buffer_df = pd.DataFrame(self.data_buffer)
        anomalies = {}
        
        for axis in self.axis_columns:
            if axis in buffer_df.columns:
                # Get non-null values
                axis_data = buffer_df[axis].dropna()
                
                if len(axis_data) < 10:
                    continue
                
                mean = axis_data.mean()
                std = axis_data.std()
                
                if std == 0:  # Avoid division by zero
                    continue
                
                threshold_upper = mean + (threshold_multiplier * std)
                threshold_lower = mean - (threshold_multiplier * std)
                
                # Find anomalies
                anomaly_mask = (buffer_df[axis] > threshold_upper) | (buffer_df[axis] < threshold_lower)
                anomaly_indices = buffer_df[anomaly_mask].index.tolist()
                
                if anomaly_indices:
                    anomalies[axis] = {
                        'indices': anomaly_indices,
                        'values': buffer_df.loc[anomaly_indices, axis].tolist(),
                        'mean': mean,
                        'std': std,
                        'threshold_upper': threshold_upper,
                        'threshold_lower': threshold_lower,
                        'severity': 'HIGH' if len(anomaly_indices) > 5 else 'MEDIUM'
                    }
        
        return anomalies
    
    def analyze_axis_health(self):
        """
        Analyze overall health of each robot axis based on current patterns.
        
        Returns:
        --------
        dict : Health status for each axis
        """
        if len(self.data_buffer) < 50:
            print("⚠️ Need at least 50 data points for health analysis")
            return {}
        
        buffer_df = pd.DataFrame(self.data_buffer)
        health_report = {}
        
        for axis in self.axis_columns:
            if axis not in buffer_df.columns:
                continue
            
            axis_data = buffer_df[axis].dropna()
            
            if len(axis_data) < 10:
                continue
            
            # Calculate health metrics
            mean_current = axis_data.mean()
            std_current = axis_data.std()
            max_current = axis_data.max()
            cv = (std_current / mean_current * 100) if mean_current > 0 else 0  # Coefficient of variation
            
            # Determine health status
            if cv > 50:
                status = 'CRITICAL - High variability'
                color = '🔴'
            elif cv > 30:
                status = 'WARNING - Moderate variability'
                color = '🟡'
            elif max_current > 30:
                status = 'CAUTION - High current draw'
                color = '🟠'
            else:
                status = 'NORMAL'
                color = '🟢'
            
            health_report[axis] = {
                'status': status,
                'color': color,
                'mean_current': mean_current,
                'std_current': std_current,
                'max_current': max_current,
                'coefficient_of_variation': cv,
                'data_points': len(axis_data)
            }
        
        return health_report
    
    def reset(self):
        """Reset the simulator to start from beginning"""
        self.current_index = 0
        self.data_buffer = []
        print("🔄 Simulator reset to beginning")
    
    def get_statistics(self):
        """Get comprehensive statistics for the current data buffer"""
        if len(self.data_buffer) == 0:
            print("⚠️ No data in buffer")
            return None
        
        buffer_df = pd.DataFrame(self.data_buffer)
        
        stats = {
            'total_points': len(buffer_df),
            'time_range': {
                'start': buffer_df['Time'].min(),
                'end': buffer_df['Time'].max(),
                'duration': (buffer_df['Time'].max() - buffer_df['Time'].min())
            },
            'axes_stats': {}
        }
        
        for axis in self.axis_columns:
            if axis in buffer_df.columns:
                axis_data = buffer_df[axis].dropna()
                if len(axis_data) > 0:
                    stats['axes_stats'][axis] = {
                        'mean': axis_data.mean(),
                        'median': axis_data.median(),
                        'std': axis_data.std(),
                        'min': axis_data.min(),
                        'max': axis_data.max(),
                        'non_null_count': len(axis_data)
                    }
        
        return stats


# ============================================================================
# STANDALONE SCRIPT MODE
# Run this file directly: python StreamingSimulator.py
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 ROBOT CONTROLLER STREAMING SIMULATOR - STANDALONE MODE")
    print("=" * 80)
    print("\n📌 This will open plots in a SEPARATE WINDOW")
    print("   (Not inline like in Jupyter notebook)\n")
    
    # Example configuration - MODIFY THESE FOR YOUR SETUP
    db_config = {
        'host': 'ep-polished-snow-ahx3qiod-pooler.c-3.us-east-1.aws.neon.tech',
        'database': 'neondb',
        'user': 'neondb_owner',
        'password': 'npg_JlIENr3i4AbL',
        'port': 5432,
        'sslmode': 'require'
    }
    
    csv_path = 'RMBR4-2_export_test.csv'  # Path to your CSV file
    
    print("⚙️ Configuration:")
    print(f"   Database: {db_config['host']}")
    print(f"   CSV File: {csv_path}")
    print()
    
    # Ask user if they want to continue
    try:
        response = input("Continue with these settings? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    
    print("\n" + "=" * 80)
    print("Initializing Simulator...")
    print("=" * 80)
    
    try:
        # Initialize simulator
        ss = StreamingSimulator(csv_path=csv_path, db_config=db_config)
        
        # Ask how many points to stream
        print(f"\nTotal records available: {len(ss.df):,}")
        try:
            num_points = input("\nHow many points to stream? (default: 100): ")
            num_points = int(num_points) if num_points.strip() else 100
        except:
            num_points = 100
        
        try:
            viz_freq = input("Visualize every Nth point (default: 10): ")
            viz_freq = int(viz_freq) if viz_freq.strip() else 10
        except:
            viz_freq = 10
        
        print("\n" + "=" * 80)
        print(f"🎬 Starting streaming: {num_points} points, visualizing every {viz_freq}th")
        print("=" * 80)
        print("\n💡 Look for the plot window - it will update as data streams!")
        print("   Press Ctrl+C to stop\n")
        
        # Stream data
        for i in range(num_points):
            try:
                data_point = ss.nextDataPoint(
                    visualize=(i % viz_freq == 0), 
                    plot_style='combined'
                )
                
                if data_point is None:
                    print("\n\n⚠️ Reached end of data")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Streaming interrupted by user")
                break
        
        print("\n\n" + "=" * 80)
        print("✅ STREAMING COMPLETE")
        print("=" * 80)
        print(f"   Points processed: {ss.current_index}")
        print(f"   Points in buffer: {len(ss.data_buffer)}")
        
        # Keep window open
        print("\n💡 Plot window will stay open until you close it")
        print("   Press Enter to exit or close the plot window...")
        
        try:
            input()
        except KeyboardInterrupt:
            pass
        
        plt.close('all')
        print("\n✅ Done!")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: CSV file not found: {csv_path}")
        print("   Please update the csv_path variable in this script")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)