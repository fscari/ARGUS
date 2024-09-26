import pandas as pd
import plotly.graph_objects as go

# Read the CSV file
df = pd.read_csv('C:\\Users\\localadmin\\PycharmProjects\\Argus\\lidar_data\\Lidar_data_2024-09-26_161608.csv')

# Convert 'epoch' to datetime (assuming epoch is in nanoseconds)
df['epoch'] = pd.to_datetime(df['epoch'], unit='ns')

# Sort the data by 'epoch'
df = df.sort_values('epoch').reset_index(drop=True)

# Get unique timestamps
timestamps = df['epoch'].unique()

# Create frames for the animation
frames = []
count = 1
for timestamp in timestamps:
    df_time = df[df['epoch'] == timestamp]

    scatter = go.Scatter3d(
        x=df_time['x'],
        y=df_time['y'],
        z=df_time['z'],
        mode='markers',
        marker=dict(
            size=2,
            color=['rgb({},{},{})'.format(r, g, b) for r, g, b in zip(df_time['r'], df_time['g'], df_time['b'])]
        )
    )

    frames.append(go.Frame(data=[scatter], name=str(timestamp)))
    print(f"Done: {count} out of {len(timestamps)}")
    print(f'percentage: {round(count/len(timestamps) * 100)}')
    count += 1

# Create the initial plot
df_initial = df[df['epoch'] == timestamps[0]]

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=df_initial['x'],
            y=df_initial['y'],
            z=df_initial['z'],
            mode='markers',
            marker=dict(
                size=2,
                color=['rgb({},{},{})'.format(r, g, b) for r, g, b in zip(df_initial['r'], df_initial['g'], df_initial['b'])]
            )
        )
    ],
    layout=go.Layout(
        title='Point Cloud Visualization with Time Slider',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, {"frame": {"duration": 500, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}])
                ],
                x=0.1,
                y=0,
                xanchor='right',
                yanchor='top'
            )
        ],
        sliders=[{
            'steps': [{'args': [[str(timestamp)], {'frame': {'duration': 0, 'redraw': True},
                                                   'mode': 'immediate',
                                                   'transition': {'duration': 0}}],
                       'label': str(timestamp),
                       'method': 'animate'} for timestamp in timestamps],
            'transition': {'duration': 0},
            'x': 0,
            'y': -0.1,
            'currentvalue': {'font': {'size': 12}, 'prefix': 'Time: ', 'visible': True, 'xanchor': 'right'},
            'len': 1.0
        }]
    ),
    frames=frames
)

# Display the figure
fig.show()
