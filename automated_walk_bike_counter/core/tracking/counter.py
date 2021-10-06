# Copyright (c) Data Science Research Lab at California State University Los
# Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

import copy
import csv
import datetime
import logging
import os
import threading

from ..configuration import config


class ObjectCounter:
    Motorbikes = {}
    Duplicates = {}

    def __init__(self):
        self.COUNTER_m = 0
        self.COUNTER_p = 0
        self.COUNTER_c = 0
        self.COUNTER_o = 0
        self.COUNTER_car = 0
        self.COUNTER_bus = 0
        self.COUNTER_truck = 0

        self.Cars = {}
        self.Buses = {}
        self.Pedestrians = {}
        self.Cyclists = {}
        self.Trucks = {}
        self.predicted_as_cyclist = {}

        self.output_counter_file_name = "counter"
        self.last_exported_ped_counter = 0
        self.last_exported_cyclist_counter = 0
        self.last_exported_car_counter = 0
        self.last_exported_bus_counter = 0
        self.last_exported_truck_counter = 0
        self.export_counter = 0
        self.counter_thread = None
        self.valid_selected_objects = []
        self.line_of_interest_is_active = False
        self.video = None
        self.detailed_counters = {}

    def add_new_moving_object_for_counting(self, obj, position_new, postprocessed):
        cur_detected_object = obj.last_detected_object
        cont_m = cur_detected_object.mess

        print(
            f"\tObject {obj.id} identified as {cont_m}, and has been counted "
            f"{obj.counted} times"
        )

        # for duplicated detection for bikers, when biker and motorbikers get detected
        # as pedestrian first
        if cont_m == "person" or cont_m == "bicycle" or cont_m == "motorbike":

            if obj.id in self.Pedestrians.keys():

                if cont_m == "bicycle" and obj.counted_biker >= 2:
                    if self.check_object_can_be_counted(obj):
                        # this is probably a biker not a pedestrian
                        self.COUNTER_p -= 1
                        self.COUNTER_c += 1
                        self.Cyclists[obj.id] = self.COUNTER_c
                        self.Pedestrians.pop(obj.id)
                        logging.debug(
                            f"\tPerson {obj.id} has been counted as a bicycle"
                            "2 or more times, and has been re-identified as a cyclist"
                        )
                        self.predicted_as_cyclist[obj.id] = True

                elif cont_m == "bicycle" and obj.counted_biker < config.count_threshold_bike:
                    # increase counter
                    obj.counted_biker += 1
                    logging.debug(
                        f"\tPerson {obj.id} identified as a bicycle"
                        f"{obj.counted_biker} times."
                    )

                if (
                    cont_m == "motorbike"
                    and obj.counted_moter >= config.count_threshold_bike
                    and self.check_object_can_be_counted(obj)
                ):
                    # this is probably a moterbiker
                    self.COUNTER_p -= 1
                    self.COUNTER_o += 1
                    self.Motorbikes[obj.id] = self.COUNTER_o
                    self.Pedestrians.pop(obj.id)

                elif cont_m == "motorbike" and obj.counted_moter < 2:
                    obj.counted_moter += 1

            if (
                obj.id not in self.Pedestrians.keys()
                and obj.id not in self.Cyclists.keys()
                and obj.id not in self.Motorbikes.keys()
            ):

                if (
                    cont_m == "person"
                    and obj.counted >= config.count_threshold
                    and self.check_object_can_be_counted(obj)
                ):
                    logging.debug(
                        f"Starting to track pedestrian {obj.id} "
                        "as it as passed the count threshold."
                    )

                    (position_x, position_y) = obj.position[-1]
                    self.COUNTER_p += 1
                    self.update_detailed_counter(obj, cont_m)
                    self.Pedestrians[obj.id] = self.COUNTER_p
                    obj.pedestrian_id = 1
                    # mark the moving object with the id
                elif cont_m == "bicycle" and obj.counted >= config.count_threshold_bike:
                    # ever detected as pedestrian, added 4/18 for prevent detecting
                    # bicycle without rider
                    self.predicted_as_cyclist[obj.id] = True
                    if obj.pedestrian_id == 1 and self.check_object_can_be_counted(obj):
                        logging.debug(
                            f"Starting to track cyclist {obj.id} "
                            "as it as passed the count threshold."
                        )

                        self.COUNTER_c += 1
                        self.Cyclists[obj.id] = self.COUNTER_c
                        # mark the moving object with the id
                # added on 7/23
                elif (
                    cont_m == "motorbike"
                    and obj.counted >= config.count_threshold_motor
                    and self.check_object_can_be_counted(obj)
                ):
                    logging.debug(
                        f"Starting to track motorbike {obj.id} "
                        "as it as passed the count threshold."
                    )
                    self.COUNTER_o += 1
                    self.Motorbikes[obj.id] = self.COUNTER_o

        else:

            if (
                (obj.id not in self.Cars.keys())
                and (obj.id not in self.Buses.keys())
                and (obj.id not in self.Trucks.keys())
            ):

                if (
                    cont_m == "car"
                    and obj.counted >= config.count_threshold_car
                    and self.check_object_can_be_counted(obj)
                ):
                    logging.debug(
                        f"Starting to track car {obj.id} "
                        "as it as passed the count threshold."
                    )
                    self.COUNTER_car += 1
                    self.Cars[obj.id] = self.COUNTER_car

                elif (
                    cont_m == "bus"
                    and obj.counted >= config.count_threshold_bus
                    and self.check_object_can_be_counted(obj)
                ):
                    logging.debug(
                        f"Starting to track bus {obj.id} "
                        "as it as passed the count threshold."
                    )
                    self.COUNTER_bus += 1
                    self.Buses[obj.id] = self.COUNTER_bus

                elif (
                    cont_m == "truck"
                    and obj.counted >= config.count_threshold_truck
                    and self.check_object_can_be_counted(obj)
                ):
                    logging.debug(
                        f"Starting to track truck {obj.id} "
                        "as it as passed the count threshold."
                    )
                    self.COUNTER_truck += 1
                    self.Trucks[obj.id] = self.COUNTER_truck

            else:

                if obj.id in self.Trucks.keys():
                    if (
                        cont_m == "bus"
                        and obj.counted_bus >= 3
                        and self.check_object_can_be_counted(obj)
                    ):
                        # this is probably a bus not a truck
                        self.COUNTER_truck -= 1
                        self.COUNTER_bus += 1
                        self.Buses[obj.id] = self.COUNTER_bus
                        self.Trucks.pop(obj.id)
                        logging.debug(
                            f"\tTruck {obj.id} has been counted as a bus"
                            "3 or more times, and has been re-identified as a bus"
                        )

                    elif (
                        cont_m == "car"
                        and obj.counted_car >= 3
                        and self.check_object_can_be_counted(obj)
                    ):
                        # this is probably a car not a truck
                        self.COUNTER_truck -= 1
                        self.COUNTER_car += 1
                        self.Cars[obj.id] = self.COUNTER_car
                        self.Cars.pop(obj.id)
                        logging.debug(
                            f"\tTruck {obj.id} has been counted as a car"
                            "3 or more times, and has been re-identified as a car"
                        )

                    elif (
                        cont_m == "bus"
                        and obj.counted_bus < 3
                        and self.check_object_can_be_counted(obj)
                    ):
                        obj.counted_bus += 1
                        logging.debug(
                            f"\tObject {obj.id} identified as a bus"
                            f"{obj.counted_bus} times."
                        )

                if obj.id in self.Cars.keys():
                    if (
                        cont_m == "bus"
                        and obj.counted_bus >= 3
                        and self.check_object_can_be_counted(obj)
                    ):
                        # this is probably a bus not a truck
                        self.COUNTER_car -= 1
                        self.COUNTER_bus += 1
                        self.Buses[obj.id] = self.COUNTER_bus
                        self.Cars.pop(obj.id)
                        logging.debug(
                            f"\tCar {obj.id} has been counted as a bus"
                            "3 or more times, and has been re-identified as a bus"
                        )

                    elif (
                        cont_m == "bus"
                        and obj.counted_bus < 3
                        and self.check_object_can_be_counted(obj)
                    ):
                        obj.counted_bus += 1
                        logging.debug(
                            f"\tObject {obj.id} identified as a bus"
                            f"{obj.counted_bus} times."
                        )

                if obj.id in self.Buses.keys():
                    if (
                        cont_m == "truck"
                        and obj.counted_truck < 3
                        and self.check_object_can_be_counted(obj)
                    ):
                        obj.counted_truck += 1
                        logging.debug(
                            f"\tObject {obj.id} identified as a truck"
                            f"{obj.counted_bus} times."
                        )

    def export_counter_initialization(self):

        header = (
            ["Time"]
            + self.valid_selected_objects
            + self.create_detailed_counters_headers()
        )

        self.output_counter_file_name = self.output_counter_file_name + ".csv"

        if os.path.isfile(self.output_counter_file_name):
            os.remove(self.output_counter_file_name)

        with open(self.output_counter_file_name, "w", newline="") as csvfile:
            counters = csv.DictWriter(csvfile, fieldnames=header, lineterminator="\n")
            counters.writeheader()

    def export_counter_threading(self):

        self.counter_thread = threading.Thread(
            target=self.counter_export, args=(), daemon=True
        )
        self.counter_thread.start()

    def counter_export(self):

        cur_detailed_counter = copy.deepcopy(self.detailed_counters)
        self.detailed_counters = {}

        header = (
            ["Time"]
            + self.valid_selected_objects
            + self.create_detailed_counters_headers()
        )

        self.export_counter += 1

        ped_output_counter = 0
        cyclist_output_counter = 0
        car_output_counter = 0
        truck_output_counter = 0
        bus_output_counter = 0
        cur_ped_counter = 0
        cur_cyclist_counter = 0
        cur_car_counter = 0
        cur_truck_counter = 0
        cur_bus_counter = 0

        for item in header:
            if item.lower() == "pedestrian":
                cur_ped_counter = self.COUNTER_p
                ped_output_counter = cur_ped_counter - self.last_exported_ped_counter
                if ped_output_counter < 0:
                    ped_output_counter = 0
            elif item.lower() == "cyclist":
                cur_cyclist_counter = self.COUNTER_c
                cyclist_output_counter = (
                    cur_cyclist_counter - self.last_exported_cyclist_counter
                )
                if cyclist_output_counter < 0:
                    cyclist_output_counter = 0
            elif item.lower() == "car":
                cur_car_counter = self.COUNTER_car
                car_output_counter = cur_car_counter - self.last_exported_car_counter
                if car_output_counter < 0:
                    car_output_counter = 0
            elif item.lower() == "truck":
                cur_truck_counter = self.COUNTER_truck
                truck_output_counter = (
                    cur_truck_counter - self.last_exported_truck_counter
                )
                if truck_output_counter < 0:
                    truck_output_counter = 0
            elif item.lower() == "bus":
                cur_bus_counter = self.COUNTER_bus
                bus_output_counter = cur_bus_counter - self.last_exported_bus_counter
                if bus_output_counter < 0:
                    bus_output_counter = 0

        video_counted_minutes = config.periodic_counter_time * self.export_counter
        timedelta = datetime.timedelta(minutes=video_counted_minutes)

        with open(self.output_counter_file_name, "a+", newline="") as csvfile:
            counters = csv.DictWriter(csvfile, fieldnames=header, lineterminator="\n")
            data_object = {}

            # for item in ["Time"] + self.valid_selected_objects:
            for item in header:
                if item.lower() == "time":
                    data_object[item] = str(timedelta)
                elif item.lower() == "pedestrian":
                    data_object[item] = str(ped_output_counter)
                elif item.lower() == "cyclist":
                    data_object[item] = str(cyclist_output_counter)
                elif item.lower() == "car":
                    data_object[item] = str(car_output_counter)
                elif item.lower() == "truck":
                    data_object[item] = str(truck_output_counter)
                elif item.lower() == "bus":
                    data_object[item] = str(bus_output_counter)
                else:
                    if item in list(cur_detailed_counter.keys()):
                        data_object[item] = str(cur_detailed_counter[item])
                    else:
                        data_object[item] = 0

            data = [data_object]
            counters.writerows(data)

        self.last_exported_ped_counter = cur_ped_counter
        self.last_exported_cyclist_counter = cur_cyclist_counter
        self.last_exported_car_counter = cur_car_counter
        self.last_exported_truck_counter = cur_truck_counter
        self.last_exported_bus_counter = cur_bus_counter

        print("Counter exported to CSV")

    def check_object_can_be_counted(self, obj):
        return not self.line_of_interest_is_active or obj.object_passed_loi

    def update_detailed_counter(self, obj, object_name):
        if object_name == "person":
            object_name = "Pedestrian"
        elif object_name == "bicycle":
            object_name = "Cyclist"

        key = object_name + "_" + obj.moving_direction
        if key in self.detailed_counters.keys():
            self.detailed_counters[key] += 1
        else:
            self.detailed_counters[key] = 1

    def create_detailed_counters_headers(self):

        if self.video is not None and self.video.line_of_interest_info is not None:

            header_list = []
            for item in self.valid_selected_objects:
                if item in ["pedestrian", "cyclist"]:

                    header_list.append(
                        item.capitalize()
                        + "_from_"
                        + self.video.line_of_interest_info.side_A_name
                        + "_to_"
                        + self.video.line_of_interest_info.side_B_name
                    )

                    header_list.append(
                        item.capitalize()
                        + "_from_"
                        + self.video.line_of_interest_info.side_B_name
                        + "_to_"
                        + self.video.line_of_interest_info.side_A_name
                    )

            return header_list
        else:
            return []
