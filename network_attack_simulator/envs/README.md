# Network Attack Simulator

A environment for testing AI agents against a simple simulated computer network with subnetworks and machines with different vulnerabilities.

The aim is to retrieve sensitive documents located on certain machines on the network without being caught. Each document provides a large reward to the agent.

The attack terminates in one of the following ways:
1. the agent collects all sensitive documents = goal
2. the agent reaches its time limite of steps = loss

The agent receives information about the network topology, specifically:
1. the machines and subnets in the network

The agent does not know which services are running on which machine, and hence which machines are vulnerable to which exploits.

The actions available to the agent are exploits and scan.
- exploits:
    - there is one exploit action for each possible service running, which is a environment parameter
- scan:
    - reveals which services are present and absent on a target machine (i.e. inspired by Nmap behaviour)

Each action must be launched against a specific machine, but actions will only possibly work on machines that are reachable.

A machine is reachable if:
1. it is on exposed subnet (i.e. this would be the machines available to public, e.g. webserver)
2. it is on a subnet connected to a compromised subnet, where a subnet is considered compromised when at least one machine within it has been compromised by agent

## Dependencies
- Python >3.5
- Numpy

For rendering:
- matplotlib
- networkX

## Specifying an environment

The environment is defined by:
- subnets: list of subnets and how many machines on each subnet
- Network topology: the connectivity of each subnet (i.e. which subnets are connected to each other, which are connected to the public internet)
- sensitive machines: the list of sensitive machines and their value
- number of services: the number of services available (i.e. how many possible exploits are available)
- service exploits: the cost and success probability of each service exploit
- machine configurations: which services are running on each machine in network
- firewall: which service traffic is permitted for inter subnet connection on network

There are two options for generating a new environement:

#### Option 1: autogenerated
Generate an environment automatically based on test environment from literature [1][2].

Environment required parameters:
- number of machines
- number of services

Environment optional parameters (if none supplied, uses default values):
- reward for exploiting sensitive machine
- reward for exploiting user machine
- exploit cost
- scan cost
- exploit success probabilities
- machine config distribution and related hyperparameters
- restrictiveness of the firewalls

The generated network has the following structure:
- Machines distributed across each subnet following a set rule:
    - DMZ (subnet 0) - with one machine
    - sensitive (subnet 1) - with one machine
    - user (subnets 2+) - all other machines distributed in tree of subnets with each subnet containing up to 5 machines.

Two machines on network contain Sensitive documents (aka the rewards):
    - One in sensitive subnet machine = r_sensitive
    - One on machine in a leaf subnet of the user subnets tree = r_user
    - any machine with no sensitive docs = 0

e.g. To generate an environment with 5 machines and running a possible 3 services with deterministic exploits.

```
env = NetworkAttackSimulator.from_params(5, 3)
```

This will produce a network with 3 subnets:
    1. subnet 0: containing 1 machine, exposed to public and connected to subnets 1 & 2
    2. subnet 1: containing 1 machine with sensitive info (r_sensitive) and connected to subnets 1 & 2 (but not public)
    3. subnet 2: containing 3 machines, 1 which has sensitive info (r_user) and connected to subnets 1 & 2 (but not public)

The generated environment will follow a predictable network topology with the first subnet being the public subnet accessible from outside and which is connected to the 2nd and 3rd subnets. The 2nd and 3rd subnets are also connected, and any remaining subnets are connected in a tree structure from the 3rd subnet (i.e. think of these as user subnets).

The configurations of each machine are randomly generated, with two options: i) uniform at random (uniform=True), ii) using nested dirichlet process (uniform=False). The behavior of the second option is controlled by a number of hyperparameters (alpha_H, alpha_V, lambda_V), see generator.generate_config method description for more info.


The success probability of each exploit can be set in one of the following i) none, ii) "mixed", iii) single float, iv) list of floats. see generator.generate_config method description for more info on the behavior of each option.

#### Option 2: custom configuration
Generate an environment from a configuration file. The configuration file must be a .yaml file with the following properties:

- subnets: a list of number of machines in each subnet
- topology: an adjacency matrix of connections between each subnet and also the outside world
- services: number of possible services running on any given machine as an integer
- sensitive_machines: a list of lists where each list is the addresses of machines on network that contain sensitive information and the value of each machine (as float or int)
    - i.e. each list is of form: \[subnetID, machineID, value\]

**Example configuration file:**

The following specifies a network with:
- 3 subnets with 1, 1, 1 machines on each of subnets 1 to 3 respectively, for a total of 3 machines on the network
- The following connectivity:
    - Subnet 1: connected to public and subnets 2 and 3
    - Subnet 2: connected to subnets 1 and 3
    - Subnet 3: connected to subnets 1 and 2
- 1 service possibly running on each machine: ssh
- ssh service exploit has an 0.8 probability of success and costs 1 to use
- Sensitive information on (machines are zero indexed):
    - subnet 2 on machine 0 with value of 10
    - subnet 3 on machine 0 with value of 10
- A firewalls along each inter-subnet connection which allows ssh traffic for all connections, except betwen subnet 1 and 2 and from 1 to external network

```
subnets: [1, 1, 1]
topology: [[ 1, 1, 0, 0],
           [ 1, 1, 1, 1],
           [ 0, 1, 1, 1],
           [ 0, 1, 1, 1]]
sensitive_machines: [[2, 0, 10],
                     [3, 0, 10]]
num_services: 1
service_exploits:
  ssh:
    - 0.8
    - 1
machine_configurations:
  (1, 0): [ssh]
  (2, 0): [ssh]
  (3, 0): [ssh]
firewall:
  (0, 1): [ssh]
  (1, 0): []
  (1, 2): []
  (2, 1): [ssh]
  (1, 3): [ssh]
  (3, 1): [ssh]
  (2, 3): [ssh]
  (3, 2): [ssh]
```

To load a new environment from a config file use the from_file classmethod:
```
env = NetworkAttackSimulator.from_file("path/to/configfile")
```

#### Machine configurations

The configurations of each machine (i.e. which services are present/absent) are allocated randomly but deterministically, so that a network initialized with the same number of machines and services will always produce the same network. This is done to keep consistency for agents trained on a given network size.

The distribution of configurations of each machine in the network are generated using a Nested Dirichlet Process, so that across the network machines will have corelated configurations (i.e. certain services/configurations will be more common across machines on the network), the degree of correlation is controlled by alpha_H and alpha_V, with lower values leading to greater corelation. This design is based off of work in [2].

lambda_V controls the average number of services running per machine. Higher values will mean more services (so more vulnerable) machines on average.


## Interacting with the environment

Once an environment has been initialized, interacting with it is easy. There are only a few methods for interaction:
1. **reset()** : which resets the environment back to the initial state (i.e. no machines compromised and state of each machine is unknown) and returns the initial state
2. **step(action)** : which takes an action and performs one step in the environment, applying action and returning the new state, a reward and whether the new state is the goal state or not
3. **render(mode)**: for rendering the environment. See function comments for more details of options. Also see render_episode for rendering an entire episode sequence that can be stepped through action by action.


## References

[1] C. Sarraute, O. Buffet, and J. Hoffmann, “POMDPs Make Better Hackers: Accounting for Uncertainty in Penetration Testing.,” in AAAI, 2012.

[2] M. Backes, J. Hoffmann, R. Künnemann, P. Speicher, and M. Steinmetz, “Simulated Penetration Testing and Mitigation Analysis,” arXiv:1705.05088 [cs], May 2017.