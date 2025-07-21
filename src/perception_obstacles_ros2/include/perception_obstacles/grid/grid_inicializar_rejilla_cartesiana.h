#pragma once

#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <string>

#include <yaml-cpp/yaml.h>

#include "perception_obstacles/grid/grid_types.h"
#include "perception_obstacles/grid/grid_utils.h"

void grid_inicializar_rejilla_cartesiana(GRID_TYPES::CART_Data* grid_cart_data);
