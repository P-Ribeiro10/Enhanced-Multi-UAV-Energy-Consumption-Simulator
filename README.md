# Enhanced Multi-UAV Energy Consumption Simulator
The Enhanced Multi-UAV Energy Consumption (eMUAVE) Simulator is a Python-based simulator that allows to compute the energy consumption of multiple Unmanned Aerial Vehicles (UAVs), acting as Flying Access Points (FAPs) to provide wireless connectivity to Ground Users (GUs).
It computes the energy consumption for two UAV type: rotary-wing and fixed-wing.

As input, it receives a .txt file named "GUs.txt" that identifies the number of groups of GUs, the total number of GUs, their spatial positions, and their traffic demand.

As output, it provides the energy consumption per hour of the FAPs (for the two UAV types) following trajectories defined by state of the art algorithms.
