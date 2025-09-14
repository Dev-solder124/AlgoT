import matplotlib.pyplot as plt
from datetime import datetime

# Your list of datetime strings
datetime_strings = [
    '02-01-2023 09:15', '02-01-2023 09:16', '02-01-2023 09:17', '02-01-2023 09:18',
    '02-01-2023 09:19', '02-01-2023 09:20', '02-01-2023 09:21', '02-01-2023 09:22',
    '02-01-2023 09:23', '02-01-2023 09:24', '02-01-2023 09:25', '02-01-2023 09:26',
    '02-01-2023 09:27', '02-01-2023 09:28', '02-01-2023 09:29', '02-01-2023 09:30',
    '02-01-2023 09:31'
]

# Convert the datetime strings to datetime objects
datetime_objects = [datetime.strptime(dt, '%d-%m-%Y %H:%M') for dt in datetime_strings]

# Example values corresponding to each datetime
values = [10, 12, 15, 14, 16, 18, 20, 22, 24, 23, 25, 27, 30, 32, 31, 29, 28]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(datetime_objects, values, marker='o', linestyle='-')

# Formatting the plot
plt.title('Datetime vs Values')
plt.xlabel('Datetime')
plt.ylabel('Values')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

# Show the plot
plt.show()