#Author: Fatma Al Lawati
from __future__ import annotations
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from model import trafficModel
from agents import roadCell, trafficLight, carAgent

#Portrayal for canvas grid
def agent_portrayal(agent):
    if isinstance(agent, roadCell):
        #Draw the road background tile
        if agent.oriantation == "NS":
            color = "#9e9e9e"

        elif agent.oriantation == "EW":
            color = "#9e9e9e"

        #Intersection
        else:
            color = "#8f8f8f"
        
        return {
            "Shape": "rect",
            "w": 1,
            "h": 1,
            "Filled": "true",
            "Color": color,
            "Layer": 0,
        }
    
    if isinstance(agent, trafficLight):
        #Show a square with color reflecting which direction has green
        if agent.phase == 0:
            #NS green
            color = "#28a745"  #green
            text = "NS"
        else:
            color = "#dc3545"
            text = "EW"
        return {
            "Shape": "rect",
            "w": 0.7,
            "h": 0.7,
            "Filled": "true",
            "Color": color,
            "Layer": 1,
            "text": text,
            "text_color": "white",
        }

    if isinstance(agent, carAgent):
        #Color by direction for fun
        dir_color = {
            "N": "#e53935",
            "S": "#fb8c00",
            "E": "#1e88e5",
            "W": "#8e24aa",
        }.get(agent.dir, "#000000")

        return {
            "Shape": "circle",
            "r": 0.35,
            "Filled": "true",
            "Color": dir_color,
            "Layer": 2,
        }

    return {}

#UI controls
grid_width = 24
grid_height = 24
canvas = CanvasGrid(agent_portrayal, grid_width, grid_height, 600, 600)

chart = ChartModule(
    [
        {"Label": "Throughput", "Color": "black"},
        {"Label": "CarsAlive", "Color": "red"},
        {"Label": "AvgWaitCurrentCars", "Color": "blue"},
    ],
    data_collector_name="datacollector",
)

model_params = {
    "width": grid_width,
    "height": grid_height,
    "n_h_roads": UserSettableParameter("slider", "Horizontal Roads", 3, 1, 5, 1),
    "n_v_roads": UserSettableParameter("slider", "Vertical Roads", 3, 1, 5, 1),
    "ns_green": UserSettableParameter("slider", "NS Green (steps)", 12, 3, 40, 1),
    "ew_green": UserSettableParameter("slider", "EW Green (steps)", 12, 3, 40, 1),
    "spawn_rate": UserSettableParameter("slider", "Spawn Rate per Point", 0.15, 0.0, 0.6, 0.01),
    "max_cars": UserSettableParameter("slider", "Max Cars", 400, 50, 1200, 10),
    "seed": UserSettableParameter("number", "Random Seed (optional)", 1),
}

server = ModularServer(
    trafficModel,
    [canvas, chart],
    "Smart Traffic Manager - Simulation",
    model_params,
)

if __name__ == "__main__":
    #Opens http://127.0.0.1:8521 by default
    server.port = 8521  #8522
    server.launch()