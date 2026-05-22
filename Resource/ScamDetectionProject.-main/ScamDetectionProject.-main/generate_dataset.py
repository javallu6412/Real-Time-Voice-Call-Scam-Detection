import pandas as pd
import random

# Number of samples
n_samples = 150

# Generate sample data
caller_id = [random.randint(9000000000, 9999999999) for _ in range(n_samples)]
duration = [random.randint(10, 300) for _ in range(n_samples)]
time = [f"{random.randint(0,23):02d}:{random.randint(0,59):02d}" for _ in range(n_samples)]
frequency = [random.randint(1,5) for _ in range(n_samples)]
locations = ['Chennai','Delhi','Mumbai','Bangalore','Kolkata']
location = [random.choice(locations) for _ in range(n_samples)]
call_type = [random.choice(['scam','legit']) for _ in range(n_samples)]

# Create DataFrame
df = pd.DataFrame({
    'caller_id': caller_id,
    'duration': duration,
    'time': time,
    'frequency': frequency,
    'location': location,
    'call_type': call_type
})

# Save CSV
df.to_csv('call_metadata.csv', index=False)
print("Sample dataset created with 100 call records!")
