#!/usr/bin/env python
# -*- coding: utf-8 -*-

# GNU General Public License v3.0
# @knedl1k 2026

from manim import *
import numpy as np

class ArcFaceAnimation(Scene):
    def construct(self):
        CLASS_1_COLOR = BLUE
        CLASS_2_COLOR = RED
        MARGIN_COLOR = GRAY
        AXIS_COLOR = GREY_B
        
        THETA_1 = 30 * DEGREES
        THETA_2 = 120 * DEGREES
        
        axes = Axes(
            x_range=[-1.5, 3, 1],
            y_range=[-1.5, 3, 1],
            axis_config={"color": AXIS_COLOR},
            tips=True
        )
        labels = axes.get_axis_labels(x_label="x", y_label="y")

        vec1 = Arrow(axes.c2p(0, 0), axes.c2p(2.5 * np.cos(THETA_1), 2.5 * np.sin(THETA_1)), buff=0, color=CLASS_1_COLOR)
        vec2 = Arrow(axes.c2p(0, 0), axes.c2p(1.8 * np.cos(THETA_2), 1.8 * np.sin(THETA_2)), buff=0, color=CLASS_2_COLOR)
        
        label_w1 = MathTex(r"\mathbf{W}_1", color=CLASS_1_COLOR).next_to(vec1.get_end(), RIGHT)
        label_w2 = MathTex(r"\mathbf{W}_2", color=CLASS_2_COLOR).next_to(vec2.get_end(), LEFT)

        points_group_1 = VGroup()
        points_group_2 = VGroup()
        
        np.random.seed(50)
        
        for _ in range(6):
            r = np.random.uniform(1.5, 2.8)
            angle = THETA_1 + np.random.uniform(-0.15, 0.15)
            dot = Dot(axes.c2p(r * np.cos(angle), r * np.sin(angle)), color=CLASS_1_COLOR, radius=0.08)
            points_group_1.add(dot)

        for _ in range(6):
            r = np.random.uniform(1.0, 2.2)
            angle = THETA_2 + np.random.uniform(-0.2, 0.2)
            dot = Dot(axes.c2p(r * np.cos(angle), r * np.sin(angle)), color=CLASS_2_COLOR, radius=0.08)
            points_group_2.add(dot)

        mid_angle = (THETA_1 + THETA_2) / 2
        boundary_line = DashedLine(
            start=axes.c2p(0, 0),
            end=axes.c2p(3 * np.cos(mid_angle), 3 * np.sin(mid_angle)),
            color=GRAY
        )
        boundary_label = Text("Decision Boundary", font_size=24, color=GRAY).next_to(boundary_line.get_end(), RIGHT)

        title_softmax = Text("Standard Softmax Loss", font_size=36).to_edge(DOWN)

        self.play(Create(axes), Write(labels))
        self.play(GrowArrow(vec1), GrowArrow(vec2), Write(label_w1), Write(label_w2))
        self.play(FadeIn(points_group_1), FadeIn(points_group_2))
        self.play(Create(boundary_line), Write(boundary_label))
        self.play(Write(title_softmax))
        self.wait(2)

        title_arcface = Text("ArcFace Loss", font_size=36).to_edge(DOWN)
        
        circle = Circle(radius=2.5, color=WHITE, stroke_opacity=0.8).move_to(axes.c2p(0,0))
        label_hypersphere = Text("Hypersphere", font_size=28).next_to(circle, UP)

        new_vec1_end = axes.c2p(2.5 * np.cos(THETA_1), 2.5 * np.sin(THETA_1))
        new_vec2_end = axes.c2p(2.5 * np.cos(THETA_2), 2.5 * np.sin(THETA_2))
        
        anims_normalize = []
        
        for dot in points_group_1:
            x, y, z = dot.get_center()
            current_angle = np.arctan2(y, x)
            target_pos = 2.5 * np.array([np.cos(current_angle), np.sin(current_angle), 0])
            anims_normalize.append(dot.animate.move_to(target_pos))

        for dot in points_group_2:
            x, y, z = dot.get_center()
            current_angle = np.arctan2(y, x)
            target_pos = 2.5 * np.array([np.cos(current_angle), np.sin(current_angle), 0])
            anims_normalize.append(dot.animate.move_to(target_pos))

        self.play(
            ReplacementTransform(title_softmax, title_arcface),
            FadeOut(axes), FadeOut(labels), FadeOut(boundary_label),
            Create(circle), Write(label_hypersphere),
            Transform(vec1, Arrow(axes.c2p(0,0), new_vec1_end, buff=0, color=CLASS_1_COLOR)),
            Transform(vec2, Arrow(axes.c2p(0,0), new_vec2_end, buff=0, color=CLASS_2_COLOR)),
            label_w1.animate.next_to(new_vec1_end, RIGHT),
            label_w2.animate.next_to(new_vec2_end, LEFT),
            *anims_normalize
        )
        self.wait(1)

        MARGIN_ANGLE = 15 * DEGREES
        
        margin_sector = AnnularSector(
            inner_radius=0, outer_radius=2.5,
            start_angle=mid_angle - MARGIN_ANGLE,
            angle=2 * MARGIN_ANGLE,
            color=MARGIN_COLOR, fill_opacity=0.3, stroke_width=0
        )
        
        label_margin_text = Text("Margin\nGap", font_size=20, color=BLACK).move_to(margin_sector.get_center() * 1.5)
        
        arc_m1 = Arc(radius=1.5, start_angle=mid_angle, angle=-MARGIN_ANGLE, color=RED)
        arc_m2 = Arc(radius=1.5, start_angle=mid_angle, angle=MARGIN_ANGLE, color=BLUE)
        label_m = MathTex("m", color=RED).next_to(arc_m1, LEFT, buff=0.05)

        anims_compact = []
        
        for dot in points_group_1:
            x, y, z = dot.get_center()
            curr_ang = np.arctan2(y, x)
            new_ang = curr_ang + (THETA_1 - curr_ang) * 0.7
            target = 2.5 * np.array([np.cos(new_ang), np.sin(new_ang), 0])
            anims_compact.append(dot.animate.move_to(target))

        for dot in points_group_2:
            x, y, z = dot.get_center()
            curr_ang = np.arctan2(y, x)
            new_ang = curr_ang + (THETA_2 - curr_ang) * 0.7
            target = 2.5 * np.array([np.cos(new_ang), np.sin(new_ang), 0])
            anims_compact.append(dot.animate.move_to(target))

        self.play(FadeIn(margin_sector), Write(label_margin_text))
        self.play(Create(arc_m1), Create(arc_m2), Write(label_m))
        
        self.play(
            *anims_compact,
            run_time=2
        )
        
        self.wait(2)
