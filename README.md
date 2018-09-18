# Multi-modal Transportation Optimization
A project on using mathematical programming to solve multi-modal transportation cost minimization in goods delivery and supply chain management.
## Project Overview
In delivery services, many different transportation tools such as trucks, airplanes and ships are available. Different choices of routes and transporation tools will lead to different costs. To minimize cost, we should consider goods consolidation (Occassions when different goods share a journey together.), different transportation costs and delivery time constraints etc. This project uses mathematical programming to model such situation and solves for overall cost minimization solution. The model is implemented with **IBM Cplex API** and **numpy** matrixing in Python.

<p align="center"><img src="https://user-images.githubusercontent.com/30411828/45585955-c6311e80-b920-11e8-95c9-bc90089446b4.jpg"></p>

## Problem Statement
In our simulated case, there are 8 goods, 4 cities/countries (Shanghai, Wuxi, Singapore, Malaysia), 16 ports and 4 transportation tools. The 8 goods originate from different cities and have different destinations. Each city/country has 4 ports, the airport, railway station, seaport and warehouse. There are some routes connecting different ports. Each route has a specific transportation tool, transportation cost, transit time and weekly schedule. Warehouse in each city allows goods to be deposited for a period of time so as to fit certain transportation schedules or wait for other goods to be transported together. All goods might have different order dates and different delivery deadlines. With all these criteria, how can we find out solution routes for all goods that minimize the overall cost? 

<p align="center"><img  height="350" src="https://user-images.githubusercontent.com/30411828/45628501-d125b380-bac6-11e8-8bd8-b909ac2a257e.png"></p>

## Assumptions
Before model building, some assumptions should be made to simplify the case because real-world delivery problems consist of too many unmeasurable factors that can affect the delivery process and final outcomes. Here are the main assumptions:<br>
1. The delivery process is **deterministic**, no random effect will appear on delivery time and cost etc. 
2. Goods can be transported in **normal container**, no special containers (refrigerated, thermostatic etc.) will be needed.
3. Container only constraints on the good's **volume**, and all goods are **divisible in terms of volume**. (No bin packing problem needed to be considered.) 
4. The model only evaluates the **major carriage routes**. The first and last mile between end user and origin/destination shipping point are not considered. (**From warehouse to warehouse**.)
5. There is **only one transportation tool available between each two ports**. For instance, we can only directly go from one airport to the other airport in different cities by flight, while direct journey by ship or railway or truck is infeasible.
6. Overall cost is restricted to the most important 3 parts, **transportation cost**, **warehouse cost** and **goods tariff**.
7. The minimum unit for time is **day** in the model, and there is **at most one transit in a route in one day**. 

## Dimension & Matrixing
In order to make the criteria logic clearer and the calculation more efficient, we use the concept of matrixing to build the necessary components in the model. In our case, there are totally 4 dimensions:<br>
1. **Start Port:** &nbsp;&nbsp; ***i***<br>
Indicating the start port of a direct transport route. The dimension length equals the total number of ports in the data.
2. **End Port:** &nbsp;&nbsp; ***j***<br>
Indicating the end port of a direct transport route. The dimension length equals the total number of ports in the data.
1. **Time:** &nbsp;&nbsp; ***t***<br>
Indicating the departure time of a direct transport. The dimension length equals the total number of days between the earliest order date and the latest delivery deadline date of all goods in the data.
1. **Goods:** &nbsp;&nbsp; ***k***<br>
Indicating the goods to be transported. The dimension length equals the total number of goods in the data.

All the variable or parameter matrices to be introduced in the later parts will have one or more of these 4 dimensions.
## Decision Variables
As mentioned above, we will use the concept of **variable matrix**, a list of variables deployed in the form of a matrix or multi-dimensional array. In our model, 3 variable matrices will be introduced:<br>
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
Similar to the decision variables, the following parameter arrays or matrices are introduced for the sake of later model building:<br>

1. **Per Container Cost:** &nbsp;&nbsp; ***C***<br>
A 3 dimensional parameter matrix, each dimension representing start port, end port and time. ***C<sub>i,j,t</sub>*** in the matrix represents the overall transportation cost per container from **port i** to **port j** at **time t**. This overall cost includes handling cost, bunker/fuel cost, documentation cost, equipment cost and extra cost from [**model data.xlsx**](https://github.com/hzjken/multimodal-transportation-optimization/blob/master/model%20data.xlsx). For infeasible route, the cost element will be set to be big M (an extremely large number), making the choice infeasible.

2. **Route Fixed Cost:** &nbsp;&nbsp; ***FC***<br>
A 3 dimensional parameter matrix, each dimension representing start port, end port and time. ***FC<sub>i,j,t</sub>*** in the matrix represents the fixed transportation cost to travel from **port i** to **port j** at **time t**, regardless of goods number or volume. For infeasible route, the cost element will be set to be big M as well.

3. **Warehouse Cost:** &nbsp;&nbsp; ***wh***<br>
A one dimension array with dimension start port. ***wh<sub>i</sub>*** represents the warehouse cost per cubic meter per day at **port i**. Warehouse cost for ports with no warehouse function (like airport, railway station etc.) is set to be big M.
 
4. **Transportation Time:** &nbsp;&nbsp; ***T***<br>
A 3 dimensional parameter matrix, each dimension representing start port, end port and time. ***T<sub>i,j,t</sub>*** in the matrix represents the overall transportation time from **port i** to **port j** at **time t**. This overall time includes custom clearance time, handling time, transit time and extra time from [**model data.xlsx**](https://github.com/hzjken/multimodal-transportation-optimization/blob/master/model%20data.xlsx). For infeasible route, the time element will be set to be big M.

5. **Tax Percentage:** &nbsp;&nbsp; ***tax***<br>
A one dimension array with dimension goods. ***tax<sub>k</sub>*** represents the tax percentage for **goods k** imposed by its destination country. If the goods only goes through a domestic transit, the tax percentage for such goods will be set as 0.

6. **Transit Duty:** &nbsp;&nbsp; ***td***<br>
A two dimensional matrix, each dimension representing start port and end port. ***td<sub>i,j</sub>*** represents the transit duty (tax imposed on goods passing through a country) percentage for goods to go from **port i** to **port j**. If port i and port j belong to the same country, transit duty percentage is set to be 0. For simplicity purpose, transit duty is set to be equal among all goods. (can be extended easily) 

7. **Container Volume:** &nbsp;&nbsp; ***ctnV***<br>
A two dimensional matrix, each dimension representing start port and end port. ***ctnV<sub>i,j</sub>*** represents the volume of container in the route from **port i** to **port j**.

8. **Goods Volume:** &nbsp;&nbsp; ***V***<br>
A one dimension array with dimension goods. ***V<sub>k</sub>*** represents the volume of **goods k**.

9. **Goods Value:** &nbsp;&nbsp; ***val***<br>
A one dimension array with dimension goods. ***val<sub>k</sub>*** represents the value of **goods k**.

10. **Order Date:** &nbsp;&nbsp; ***ord***<br>
A one dimension array with dimension goods. ***ord<sub>k</sub>*** represents the order date of **goods k**.

11. **Deadline Date:** &nbsp;&nbsp; ***ddl***<br>
A one dimension array with dimension goods. ***ddl<sub>k</sub>*** represents the deadline delivery date of **goods k**.

12. **Origin Port:** &nbsp;&nbsp; ***OP***<br>
A one dimension array with dimension goods. ***OP<sub>k</sub>*** represents the port where **goods k** starts from.

13. **Destination Port:** &nbsp;&nbsp; ***DP***<br>
A one dimension array with dimension goods. ***DP<sub>k</sub>*** represents the port where **goods k** ends up to be in.

The data of all the above parameter matrices will be imported from **model data.xlsx** with function **transform()** and **set_param()**. For more details, please refer to the codes in [**multi-modal transpotation.py**](https://github.com/hzjken/multimodal-transportation-optimization/blob/master/multi-modal%20transpotation.py).

## Mathematical Modelling
With all the variables and parameters defined above, we can build up the objectives and constraints to form an integer programming model.
### Objective
<p align="center"><img width ="500" src="https://user-images.githubusercontent.com/30411828/45684896-56b66b80-bb7a-11e8-8d6b-0da2d9ec709e.png"></p>
<p align="center"><img width ="600" src="https://user-images.githubusercontent.com/30411828/45684632-82852180-bb79-11e8-8fd9-547623e9ab66.png"></p>

The objective of the model is to minimize the overall cost, which includes 3 parts, **transportation cost**, **warehouse cost** and **tax cost**. Firstly, the **transportation cost** includes container cost and route fixed cost. Container cost equals the number of containers used in each route times per container cost while route fixed cost equals the sum of fixed cost of all used routes. Secondly, the **warehouse cost** equals all goods' sum of volume times days of storage times warehouse fee per cubic meter per day in each warehouse. Finally, the **tax cost** equals the sum of import tariff and transit duty of all goods.
```python
transportCost = np.sum(y*perCtnCost) + np.sum(z*tranFixedCost)
warehouseCost = warehouse_fee(x)[0] #For details ,pleas refer to function warehouse_fee() in code.
taxCost = np.sum(taxPct*kValue) + np.sum(np.sum(np.dot(x,kValue),axis=2)*transitDuty)
model.minimize(transportCost + warehouseCost + taxCost)
```
### Constraints


## Optimization Result & Solution

## Model Use & Extension Guide
