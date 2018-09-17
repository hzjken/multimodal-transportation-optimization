# Multi-modal Transportation Optimization
A project on using mathematical programming to solve multi-modal transportation cost minimization in goods delivery and supply chain management.
## Project Overview
In delivery services, many different transportation tools such as trucks, airplanes and ships are available. Different choices of routes and transporation tools will lead to different costs. To minimize cost, we should consider goods consolidation (Occassions when different goods share a journey together.), different transportation costs and delivery time constraints etc. This project uses mathematical programming to model such situation and solves for cost minimization solution. The model is implemented with **IBM Cplex API** and **numpy** matrixing in Python.

<p align="center"><img src="https://user-images.githubusercontent.com/30411828/45585955-c6311e80-b920-11e8-95c9-bc90089446b4.jpg"></p>

## Assumptions
Before model building, some assumptions should be made to simplify the case because real-world delivery problems consist of too many unmeasurable factors that can affect the delivery process and final outcomes. Here are the main assumptions:<br>
1. The delivery process is **deterministic**, no random effect will appear on delivery time and cost etc. 
2. Goods can be transported in **normal container**, no special containers (refrigerated, thermostatic etc.) will be needed.
3. Container only constraints on the good's **volume**, and all goods are **divisible in terms of volume**. (No bin packing problem needed to be considered.) 
4. The model only evaluates the **major carriage routes**. The first and last mile between end user and origin/destination shipping point are not considered. (**From warehouse to warehouse**.)
5. There is **only one transportation tool available between each two ports**. For instance, we can only directly go from one airport to the other airport in different cities by flight, while direct journey by ship or railway or truck is infeasible.
6. Overall cost is restricted to the most important 3 parts, **transportation cost**, **warehouse cost** and **goods tariff**.

## Decision Variables
In order to fit such problem into the framework of mathematical programming and simplify the later model building part, we need to use the concept of **variable matrix**, a list of variables deployed in the form of a matrix or multi-dimensional array. In our modelling, 3 variable matrices will be introduced.<br>
1. **Decision Variable Matrix:** &nbsp;&nbsp; ***X***<br>
The most important variable matrix in the model. It's a 4 dimensional matrix, each dimension representing start port, end port, time and goods respectively. Each element in the matrix is a binary variable, representing whether a route is taken by a specific goods. For example, element ***X<sub>i,j,t,k</sub>*** represents whether **goods k** travels from **port i** to **port j** at **time t**.
```python
  varList1 = model.binary_var_list(portDim * portDim * timeDim * goodsDim,name = 'x')
  x = np.array(varList1).reshape(portDim, portDim, timeDim, goodsDim)
```

2. **Container Number Matrix:** &nbsp;&nbsp; ***Y***<br>
A variable matrix used to support the decision variable matrix. It's a 3 dimensional matrix, with each dimension representing start port, end port and time respectively. Each element in the matrix is an integer variable, representing the number of containers needed in a specific route. For example, ***Y<sub>i,j,t</sub>*** represents the number of containers needed to load all the goods travelling simultaneously from **port i** to **port j** at **time t**. Such matrix is introduced to make up for the limitation of "linear operator only" in mathematical programming, when we need a **roundup()** method in direct calculation of the container number.
```python
  varList2 = model.integer_var_list(portDim * portDim * timeDim,name = 'y')
  y = np.array(varList2).reshape(portDim, portDim, timeDim)
  ```

3. **Route Usage Matrix:** &nbsp;&nbsp; ***Z***<br>
A variable matrix used to support the decision variable matrix. It's a 3 dimensional matrix, with each dimension representing start port, end port and time respectively. Each element in the matrix is a binary variable, representing whether a route is used or not. For instance, ***Z<sub>i,j,t</sub>*** represents whether the route from **port i** to **port j** at **time t** is used or not (no matter which goods). It's introduced with similar purpose to ***Y<sub>i,j,t</sub>*** .
```python
  varList3 = model.binary_var_list(portDim * portDim * timeDim,name = 'z')
  z = np.array(varList3).reshape(portDim, portDim, timeDim)        
```

## Parameters
Similar to the decision variables mentioned above, the following parameters or parameter matrices are introduced for the sake of later model building:<br>
1. **Per Container Cost:** &nbsp;&nbsp; ***C***<br>
A 3 dimensional parameter matrix, each dimension representing start port, end port and time. ***C<sub>i,j,t</sub>*** in the matrix represents the overall transportation cost per container from **port i** to **port j** at **time t**. This overall cost includes handling cost, bunker/fuel cost, documentation cost, equipment cost and extra cost from **model data.xlsx**. For infeasible route, the cost element will be set to be big M (an extremely large number), making the choice infeasible.

2. **Route Fixed Cost:** &nbsp;&nbsp; ***FC***<br>
A 3 dimensional parameter matrix, each dimension representing start port, end port and time. ***FC<sub>i,j,t</sub>*** in the matrix represents the fixed transportation cost to travel from **port i** to **port j** at **time t**, regardless of goods number or volume. For infeasible route, the cost element will be set to be big M as well.

3. **Warehouse Cost:** &nbsp;&nbsp; ***wh***<br>
A one dimension array with length equaling the number of ports in the data. ***wh<sub>i</sub>*** represents the warehouse cost per cubic meter per day at **port i**. Warehouse cost for ports with no warehouse function (like airport, railway station etc.) is set to be big M.

4. **Transportation Time:** &nbsp;&nbsp; ***T***<br>
A 3 dimensional parameter matrix, each dimension representing start port, end port and time. ***T<sub>i,j,t</sub>*** in the matrix represents the overall transportation time from **port i** to **port j** at **time t**. This overall time includes custom clearance time, handling time, transit time and extra time from **model data.xlsx**. For infeasible route, the time element will be set to be big M as well.

## Mathematical Modelling

## Optimization Result & Solution

## Model Use & Extension Guide
