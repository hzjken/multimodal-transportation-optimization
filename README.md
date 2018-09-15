# multimodal-transportation-optimization
A project on using mathematical programming to solve multi-modal transportation cost minimization in goods delivery and supply chain management.
## Project Overview
In delivery services, many different transportation tools such as trucks, airplanes and ships are available. Different choices of routes and transporation tools will lead to different costs. To minimize cost, we should consider goods consolidation (occassions when different goods share a journey together.), different transportation costs and delivery time constraints etc. This project uses mathematical programming to model such situation and solves for cost minimization solution. The model is implemented with **IBM Cplex API** and **numpy** matrixing in Python.

<p align="center"><img src="https://user-images.githubusercontent.com/30411828/45585955-c6311e80-b920-11e8-95c9-bc90089446b4.jpg"></p>

## Assumptions
Before model building, some assumptions should be made to simplify the case because real-world delivery problems consist of too many unmeasurable factors that can affect the delivery process and final outcomes. Here are the assumptions:<br>
1. The delivery process is **deterministic**, no random effect will appear on delivery time and cost etc. 
2. Goods can be transported in **normal container**, no special containers (refrigerated, thermostatic etc.) will be needed.
3. Container only constraints on the good's **volume**, and all goods are **divisible in terms of volume**. (no bin packing problem needed to be considered.) 
4. The model only evaluates the **major carriage routes**. The first and last mile between end user and origin/destination shipping point are not considered. (**From warehouse to warehouse**.)
5. There is **only one transportation tool available between each two ports**. For instance, we can only directly go from one airport to the other airport in different cities by flight, while direct journey by ship or railway or truck is infeasible.
6. Overall cost is restricted to the most important 3 parts, **transportation cost**, **warehouse cost** and **goods tariff**.

## Parameters

## Decision Variables

## Mathematical Modelling

## Optimization Result & Solution
