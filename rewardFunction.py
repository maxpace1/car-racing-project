"""
We assume our vehicle to be an electric RWD, powered by a single motor without a differential mechanism. This gives us scope to create a 'reward' function that assesses mechanical and electrical sources of loss given current action. This loss is invariant non-positive. 

Vehicle efficiency is maximized using a coast-and-burn strategy to limit losses in the form of braking (wasted heat energy), severe steering adjustments (increasing lateral wheel load, common-axle scrubbing), and excess time spent in lower rev-ranges (when mechanical losses dominate) and redline rev-ranges (where rotor windage losses increase). Maximizing efficiency therefore comes down to:
- penalizing use of braking proportional to engagement, at throttle level
- penalizing steering proportional to steering lock, at throttle level
- penalizing rev-ranges (0,0.5], (0.9,1.0]

The weightages of these factors are configurable, and overall losses are to be considered alongside the goal of completing the track as quickly as possible, without leaving it.
"""

# All weights in range [0,10], severity proportional to weight
steering_penalty_weight = 4
low_rev_penalty_weight = 2
high_rev_penalty_weight = 1
braking_penalty_weight = 8

def computeLosses(action):
  """ Returns the non-positive loss (efficiency reward) attained by the current set of actions.
  
  Parameters:
  action (list): Holds [steering [-1,+1], throttle [0,+1], braking [0,+1]] actuation

  Returns:
  int: reward of current action
  """
  steerActuation = action[0]
  throttleActuation = action[1]
  brakeActuation = action[2]

  steering_loss = steering_penalty_weight * abs(steerActuation) * throttleActuation
  
  if throttleActuation < 0.5:
    throttle_loss = low_rev_penalty_weight * throttleActuation
  elif throttleActuation > 0.9:
    throttle_loss = high_rev_penalty_weight * throttleActuation
  else:
    throttle_loss = 0

  braking_loss = braking_penalty_weight * brakeActuation * throttleActuation

  cumulative_losses = -(steering_loss + throttle_loss + braking_loss)

  return cumulative_losses


