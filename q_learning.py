import random
import torch

# Define the global environmental variables
DR = 0.99   # Discount Rate
LR = 0.1    # Learning Rate
ER_DECAY = 0.98
MIN_ER = 0.1

# Number of actions and states for experiences
# and for restaurants, respectively
NAE = 12
NSE = 6
NAR = 28
NSR = 6

# Array of each experience and restaurant type,
# respectively
EXPERIENCES = [
    'Movie',
    'Ice Skate',
    'Escape Room',
    'Paintball',
    'Park',
    'Zoo',
    'Theme parks',
    'Water parks',
    'Mall',
    'Gym',
    'Museums',
    'Arcade'
]

RESTAURANTS = [
    'Acai Bowls',
    'Bagels',
    'Bakeries',
    'Breweries',
    'Bubble Tea',
    'Chimney Cakes',
    'Coffee & Tea',
    'Coffee Roasteries',
    'Convenience Stores',
    'Cupcakes',
    'Desserts',
    'Do-It-Yourself Food',
    'Donuts',
    'Empanadas',
    'Food Trucks',
    'Gelato',
    'Ice Cream & Frozen Yogurt',
    'Juice Bars & Smoothies',
    'Kombucha',
    'Patisserie/Cake Shop',
    'Piadina',
    'Poke',
    'Pretzels',
    'Shaved Ice',
    'Shaved Snow',
    'Smokehouse',
    'Street Vendors',
    'Tea Rooms'
]

# Determine the state from the price and time
def get_state(price, time):
    # Breakfast
    if time > 400 and time < 1000:
        return 1 if price else 0 # Return bougie morning or bargain morning
    # Lunch
    elif time >= 1000 and time < 1600:
        return 3 if price else 2
    # Dinner
    else:
        return 5 if price else 4

# Returns an unbiased argmax for a given tensor
def unbiased_argmax(tensor, na):
    # Determine if the the tensor row consists solely
    # of zeros
    if torch.equal(tensor, torch.zeros(na, dtype=torch.float)):
        return random.randint(0, na - 1)
    else:
        return torch.argmax(tensor)

# For first time algorithm initialization
def initialize_q_learning(price, time):
    # Create a q table with the appropriate state action dimensions
    # for both restaurants and 
    q_table_r, q_table_e = torch.ones(NSR, NAR), torch.ones(NSE, NAE)
    
    # Fetch the current state
    state = get_state(price, time)

    # Set the episode to zero
    episode = 0

    # Randomly choose both an experience and a restaurant
    experience = random.randint(0, NAE - 1)
    restaurant = random.randint(0, NAR - 1)

    # Convert the q_tables to lists (for JSON)
    q_table_r, q_table_e = q_table_r.tolist(), q_table_e.tolist()

    # Convert the restaurant and experience to string form
    experience = EXPERIENCES[experience]
    restaurant = RESTAURANTS[restaurant]

    return q_table_e, q_table_r, experience, restaurant, episode, state

# Iterate through one step of the learning process
def step_q_learning(q_table_e, q_table_r, last_s, last_e, last_r, satisfaction, episode, price, time):
    # Convert each q table back into torch tensors
    q_table_e = torch.tensor(q_table_e, dtype=torch.float).view(NSE, NAE)
    q_table_r = torch.tensor(q_table_r, dtype=torch.float).view(NSR, NAR)

    # Fetch the current state
    state = get_state(price, time)

    # Update the q table based on the last state
    # Convert the string of the experience and restaurant from
    # string form to numeric form
    last_e = EXPERIENCES.index(last_e)
    last_r = RESTAURANTS.index(last_r)

    # Determine the reward based on the satisfaction
    reward = 1 if satisfaction else -1

    # Update the appropriate values based on the bellman equation
    q_table_e[last_s, last_e] = (1 - LR) * q_table_e[last_s, last_e] + LR * (reward + DR * torch.max(q_table_e[state, :]))
    q_table_r[last_s, last_r] = (1 - LR) * q_table_r[last_s, last_r] + LR * (reward + DR * torch.max(q_table_r[state, :]))
    
    # Determine the exploration rate based on the current
    # iteration
    er = ER_DECAY ** episode

    # Determines if the next action should be greedy or 
    # exploring
    epsilon = random.uniform(0,1)
    if epsilon > er:
        experience = int(unbiased_argmax(q_table_e[state, :], NAE))
        restaurant = int(unbiased_argmax(q_table_r[state, :], NAR))
    else:
        experience = random.randint(0, NAE - 1)
        restaurant = random.randint(0, NAR - 1)
    # Step the episode
    episode += 1
    
    # Flatten the q table to return to the user
    q_table_r, q_table_e = q_table_r.tolist(), q_table_e.tolist()

    # Convert the restaurant and experience to string form
    experience = EXPERIENCES[experience]
    restaurant = RESTAURANTS[restaurant]

    return q_table_e, q_table_r, experience, restaurant, episode, state

