# Learning Bidding Strategies for Karma Economies in Realistic Traffic Settings with Multi-Agent Reinforcement Learning

Karma is a non-monetary resource-allocation mechanism that prioritizes users' needs rather than their financial power. Monetary pricing can effectively reduce congestion by imposing charges on specific road segments, but it may be unfair by favoring higher-income individuals. Prior work has shown that in this context, Karma can achieve similar efficiency while yielding fairer outcomes; however, demonstrated only in a deterministic setting. Demonstrating Karma's applicability under more realistic traffic conditions is therefore important for real-world implementation. Additionally, experimental evidence suggests that humans may struggle to execute optimal bidding strategies in Karma economies. In this paper, we demonstrate the use of Multi-Agent Reinforcement Learning (MARL) to train automated bidding agents for travelers. In a microscopic traffic simulation case study, we show that MARL agents learn effective bidding strategies that yield fairer travel outcomes for drivers than those achieved under monetary pricing schemes 

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
├── routerl_aec_environment_karma/
│    └── ...
├── scenarios/
│   └── ...
├── server_scripts/
│   └── ...
```


`network_analysis` directory contains the selected network, along with analyses of its key characteristics, as well as scripts and plots that illustrate the reasoning behind its selection. The network was chosen because it effectively demonstrates the trade-off between route length and capacity: **Route 0** is the shortest but becomes congested quickly, **Route 1** is slightly longer yet offers greater capacity with two lanes, and **Route 2** is significantly longer, however provides the highest capacity with five lanes.


The `routerl_aec_environment` contains a version of the [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL) framework, where agents are assigned an income value, and travel route costs are incorporated into the agent's reward function based on each agent’s Value of Time (VoT).

The `routerl_aec_environment_karma` directory contains an updated version of the [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL) framework, which we extend by modifying agents’ reward functions and introducing bidding and auction mechanisms. 


`scenarios` folder the different scenarios that are going to be tested during the work. Specifically, 

- [`monetary_pricing`](https://github.com/AnastasiaPsarou/RouteRL-Karma/tree/main/scenarios/monetary_pricing) contains scripts where agents choose routes aiming to minimize their cost, which consists of route fees and their VoT.
- [`karma_pricing`](https://github.com/AnastasiaPsarou/RouteRL-Karma/tree/main/scenarios/karma_pricing) contains scripts where agents learn optimal bidding strategies in Karma economies.

  
The `server_scripts` folder contains scripts for running experiments using the GMUM servers.

<!--
## Installation

```pip3 install git+https://github.com/COeXISTENCE-PROJECT/RouteRL.git@dev ``` 

```pip3 install git+https://github.com/COeXISTENCE-PROJECT/Janux.git@dev ``` -->


## References
```
Riehl, K., Kouvelas, A. & Makridis, M.A. Karma economies for sustainable urban mobility – a fair approach to public good value pricing. npj. Sustain. Mobil. Transp. 1, 14 (2024). https://doi.org/10.1038/s44333-024-00014-4
```

```
Akman*, A. O., Psarou*, A., Gorczyca, Ł., Varga, Z. G., Jamróz, G., & Kucharski, R. (2025). RouteRL: Multi-agent reinforcement learning framework for urban route choice with autonomous vehicles. SoftwareX, 31, 102279. https://doi.org/10.1016/j.softx.2025.102279
```






