# Overview

Building on previous work showing that Karma can contribute to fairer traffic networks in systems composed exclusively of human drivers, a research question emerges: **Can Karma serve as an effective resource allocation mechanism in systems composed solely of autonomous vehicles?**

# Contents


```
./
├── evaluation/ 
│   └── ...
├── income_data/ 
│   └── ...
├── network_analysis/ 
│   └── ...
├── routerl_aec_environment/
│    └── ...
├── routerl_karma_parallel_env/
│    └── ...
├── scenarios/
│   └── ...
├── server_scripts/
│   └── ...
```


`network_analysis` directory contains the selected network, along with analyses of its key characteristics, as well as scripts and plots that illustrate the reasoning behind its selection. The network was chosen because it effectively demonstrates the trade-off between route length and capacity: **Route 0** is the shortest but becomes congested quickly, **Route 1** is slightly longer yet offers greater capacity with two lanes, and **Route 2** is significantly longer, however provides the highest capacity with five lanes.


The `routerl_karma_parallel_env` directory contains an updated version of the [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL) framework, in which each agent’s actions involve selecting both a **route** and a **departure time**. To accommodate this MultiDiscrete action space, this version employs the **Parallel API** from the PettingZoo library instead of the **AEC API** used in the original RouteRL implementation. Furthermore, it introduces **monetary pricing** and **karma-based mechanisms** within the route choice context.


The `routerl_aec_environment` contains a version of the [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL) framework, where agents are assigned an income value, and travel route costs are incorporated into the agent's reward function based on each agent’s Value of Time (VoT).


`scenarios` folder the different scenarios that are going to be tested during the work. Specifically, 

- [`google maps`](https://github.com/AnastasiaPsarou/RouteRL-Karma/tree/main/scenarios/google_maps) contains scripts where AV agents act independently, learn optimal routes and departure times. Each agent acts selfishly, aiming to minimize its own travel time.
- [`benevolent_dictator`](https://github.com/AnastasiaPsarou/RouteRL-Karma/tree/main/scenarios/benevolent_dictator) contains the implementation of the **benevolent dictator**, a centralized multi-agent reinforcement learning (MARL) scenario where the agents aim to minimize the system travel time.
- [`monetary_pricing`](https://github.com/AnastasiaPsarou/RouteRL-Karma/tree/main/scenarios/benevolent_dictator) contains scripts where agents choose routes aiming to minimize their cost, which consists of route fees and their VoT.

  
The `server_scripts` folder contains scripts for running experiments using the GMUM servers.

<!--
## Installation

```pip3 install git+https://github.com/COeXISTENCE-PROJECT/RouteRL.git@dev ``` 

```pip3 install git+https://github.com/COeXISTENCE-PROJECT/Janux.git@dev ``` -->

## Path generation


Check the paths generated [here](benevolent_dictator/plots_simple_network_mappo_benevolent_dictator_10_agents).

## References
```
Riehl, K., Kouvelas, A. & Makridis, M.A. Karma economies for sustainable urban mobility – a fair approach to public good value pricing. npj. Sustain. Mobil. Transp. 1, 14 (2024). https://doi.org/10.1038/s44333-024-00014-4
```

```
Akman*, A. O., Psarou*, A., Gorczyca, Ł., Varga, Z. G., Jamróz, G., & Kucharski, R. (2025). RouteRL: Multi-agent reinforcement learning framework for urban route choice with autonomous vehicles. SoftwareX, 31, 102279. https://doi.org/10.1016/j.softx.2025.102279
```






