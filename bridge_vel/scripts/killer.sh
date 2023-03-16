#!/bin/bash

# sleep 5

echo "Killing twist_mux"
rosnode kill twist_mux

echo "Killing twist_marker"
rosnode kill twist_marker

# sleep 5