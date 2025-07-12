#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC

from src.actions.base_action import Advance, Sleep, SpinAntiClockwise, Stop, SpinClockwise, CustomAction, ShiftLeft, ShiftRight, TurnRight


class ComplexAction(ABC):
    def __init__(self, ):
        # 当前动作是否强制执行，默认为强制执行
        self.force = True
        # 当前动作序列是否更新控制器记录的速度，默认为不更新
        self.update_controller_speed = False
        pass


class TurnLeftInPlace(ComplexAction):
    def __init__(self):
        super().__init__()
        #print("左转！")
        self.action_seq = [
            Advance(speed=45),
            Sleep(0.35),
            SpinAntiClockwise(speed=50),
            Sleep(0.875),
            Stop()
        ]


class TurnRightInPlace(ComplexAction):
    def __init__(self):
        super().__init__()
        self.action_seq = [
            Advance(speed=50),
            #Sleep(0.675),
            Sleep(0.4),
            SpinClockwise(speed=50),
            Sleep(0.66),
            Stop(),
            Sleep(1)
        ]


class TurnAround(ComplexAction):
    def __init__(self):
        super().__init__()
        self.action_seq = [
            Stop(),
            Sleep(0.5),
            Advance(speed=45),
            Sleep(1.4),#1.55,2
            Stop(),
            SpinAntiClockwise(speed=50),
            Sleep(0.685),
            Advance(speed=45),
            Sleep(0.4),#0.9
            SpinAntiClockwise(speed=50),
            Sleep(0.685), #0.45
            Stop(),
            Sleep(1)

        ]


class Start(ComplexAction):
    def __init__(self):
        super().__init__()
        self.update_controller_speed = True
        self.action_seq = [
            Advance(speed=35),
            Sleep(0.2),
            Advance(speed=25)
        ]

class Stopobs2(ComplexAction):
    def __init__(self):
        super().__init__()
        print("执行停车！")
        self.action_seq = [
            Stop(),
            #Advance(speed=35),
            Sleep(2)
        ]

class Sto(ComplexAction):
    def __init__(self):
        super().__init__()
        self.update_controller_speed = True
        self.action_seq = [
            Advance(speed=45),
            Sleep(0.6),
            SpinClockwise(speed=50),
            Sleep(0.7),
            Advance(speed=45),
            Sleep(0.5),
            SpinAntiClockwise(speed=50),
            Sleep(0.75),
            Advance(0),
            Sleep(1.5),
            SpinAntiClockwise(speed=50),
            Sleep(0.9),
            Advance(speed=45),
            Sleep(0.5),
            SpinClockwise(speed=50),
            Sleep(0.7),
            Stop()
        ]
        
class Parking(ComplexAction):
    def __init__(self):
        super().__init__()
        self.action_seq = [
            Stop(),
            Sleep(1),

            CustomAction(motor_setting=[-85, 65, 60, -55]),
            Sleep(0.75),
            Stop(),

            Sleep(2),
            CustomAction(motor_setting=[65, -58, -55, 55]),
            Sleep(1),
            Stop()
        ]
