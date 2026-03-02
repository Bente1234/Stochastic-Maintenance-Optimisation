

import numpy as np
import pandas as pd
from math import factorial
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
import math

#Maintenance Highway
# Read Data
df = pd.read_csv('highway_data.csv')

# Parameters
c_i = 10 # Inspection cost (thousands euros)
c_minor = 0 # Repair cost for minor damage  (thousands euros)
c_moderate = 1 # Repair cost for moderate damage (thousands euros)
c_major = 10 # Repair cost for major damage (thousands euros)
penalty = 500 # Penalty cost (thousands euros)
penalty_threshold = 20 # Threshold of total severity score for penalty

#Preparing the data
damage_cols = ['Damage 1','Damage 2','Damage 3','Damage 4','Damage 5','Damage 6']
#Number of damages per row
df['k'] = df[damage_cols].notna().sum(axis=1)
#Mapping month names
month_number= {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
df['Month_num'] = df['Month'].map(month_number)
df['Date'] = pd.to_datetime(dict(year=df['Year'], month=df['Month_num'], day=1))
df['T'] = df['Date'].dt.days_in_month

# Task a
#Here lambda is the damage arrival rate, x is the damage count and T is the number of days
def likelihood (lam, x, T):
    #The likelihood of a Poisson distribution, a multiplication of the PMF
    likelihood = ((lam * T) ** x) * np.exp(-lam * T) / [math.factorial(int(i)) for i in x]
    return np.prod(likelihood)

#To maximise the likelihood the log function of the likelihood is created
def mle_lam(x, T):

    def log_likelihood(lam):
        if lam <= 0:
            return -np.inf  #incase
        return np.sum(x * np.log(lam * T) - lam * T - [math.lgamma(int(i)+1) for i in x])

    #Maximize log-likelihood is the same as minimizing the negative log-likelihood, this is done because scipy does not have a maximize function
    res = minimize_scalar(lambda lam: -log_likelihood(lam),
                          bounds=(1e-6, 10), method='bounded')

    return res.x
#Computing the observed damages (x) and the exposure time (T)
x = df['k'].values
T = df['T'].values

print(f'(a) MLE of lambda: {mle_lam(x, T):.4f} damages/day')


# Task b
#To estimate the probability per severity of damage, the number of that severity sort is divided by the total amount of damages 
prob_severity = 0
severity_values = df[damage_cols].values.ravel()
severity_values = severity_values[~pd.isna(severity_values)]

#Sum number of damages per severity level
n_minor = np.sum(severity_values == 0.1)
n_moderate = np.sum(severity_values == 1)
n_major = np.sum(severity_values == 10)

n_total = n_minor + n_moderate + n_major

#Creating a list of the probability per severity level
prob_severity = np.array([n_minor/n_total, n_moderate/n_total, n_major/n_total])

print(f'(b) P_minor = {prob_severity[0]:.4f}, P_moderate = {prob_severity[1]:.4f}, P_major = {prob_severity[2]:.4f}')

# Task c
#A Monte Carlo simulation is used to estimate the expected monthly maintenance cost under the current maintenance policy
def simulate_month(lam, prob_severity):
    #Defining all costs and the damage threshold 
    inspection_cost = 10000
    repair_costs = {0.1: 0, 1: 1000, 10: 10000}
    penalty_cost = 500000
    L = 20

    #Simulation of the poisson proccess with inter arrival rate lambda
    k = np.random.poisson(lam)

    #Computing based on the probabilities from task b, per severity the number of damages per severity level based on k arrivals
    severities = np.random.choice([0.1, 1, 10], size=k, p=prob_severity)

    #Summing the total repair costs based on, per damage sort, the number of damages
    repair_total = sum(repair_costs[s] for s in severities)

    #Sum the severity score over all  occured damages
    severity_score = np.sum(severities)
    #Add penalty if the severity score is equal to or exceeds the threshold of L = 20
    penalty = penalty_cost if severity_score >= L else 0

    #Sum all three different cost types to find the total cost per month
    return inspection_cost + repair_total + penalty

#Simulate the calculation above 10000 times based on the poisson process with inter arrival time lambda
def expected_monthly_cost(lam, prob_severity, n_sim=10000):
    costs = [simulate_month(lam, prob_severity) for _ in range(n_sim)]
    return np.mean(costs)

#Running the simulation above with the following parameters:
lam = mle_lam(x, 1) #Lambda calculted in task a with an inspection interval of 1
prob_severity = prob_severity #Probabilities that were computed in task b
exp_cost = expected_monthly_cost(lam, prob_severity)
print(f"(c) Monthly maintenance cost = {exp_cost}")

# Task d
#Using the same function as in task c, but now adapting it such that different inspection interval's can be tested
def simulate_cycle(lam_month, prob_severity, tau):
    inspection_cost = 10000
    repair_costs = {0.1: 0, 1: 1000, 10: 10000}
    penalty_cost = 500000
    threshold = 20

    #Creating poission arrivals according to lambda
    k = np.random.poisson(lam_month * tau)

    severities = np.random.choice([0.1, 1, 10], size=k, p=prob_severity)

    repair_total = sum(repair_costs[s] for s in severities)
    severity_score = np.sum(severities)
    penalty = penalty_cost if severity_score > threshold else 0

    return inspection_cost + repair_total + penalty

#Simulating the expected costs per month based on a tau that can be selected manually
def expected_monthly_cost_tau(lam_month, prob_severity, tau, n_sim=10000):
    costs = [simulate_cycle(lam_month, prob_severity, tau) for _ in range(n_sim)]
    avg_cycle_cost = np.mean(costs)
    return avg_cycle_cost / tau

#Taking taus in the range 0-1 with steps of 0.1 (this range was defined after trial and error), and performing the simulation defined above on each of these tau's
taus = np.arange(0.1, 1.1, 0.1) # must inspect at least twice a year
results = {tau: expected_monthly_cost_tau(lam, prob_severity, tau)
           for tau in taus}

#Out of the all avergage costs per tau, selecting the tau for which the costs are the smallest
tau_star = min(results, key=results.get)

print("Results per τ (months):")

for tau, cost in results.items():
    print(f"  τ={tau}: Expected monthly cost ≈ €{cost:,.2f}")

print(f"(d) Optimal inspection interval = {tau_star} months")

