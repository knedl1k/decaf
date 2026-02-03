from manim import *
import numpy as np


class SoftmaxVsArcFace(Scene):
    def construct(self):
        CLASS_1_COLOR = BLUE
        CLASS_2_COLOR = RED
        MARGIN_COLOR = GREY

        W1_DEG = 30
        W2_DEG = 150
        BOUNDARY_DEG = (W1_DEG + W2_DEG) / 2
        MARGIN_SIZE = 20

        VISUAL_RADIUS = 2.2

        ORIGIN_POINT = ORIGIN

        # --- Softmax ---

        title = Tex("Standard Softmax Loss").to_edge(UP)
        self.add(title)

        def get_pos(r, degrees):
            rad = np.deg2rad(degrees)
            return ORIGIN_POINT + np.array([np.cos(rad), np.sin(rad), 0]) * r

        w1_vec = Arrow(ORIGIN_POINT, get_pos(1.8, W1_DEG), buff=0, color=CLASS_1_COLOR, stroke_width=6)
        w2_vec = Arrow(ORIGIN_POINT, get_pos(1.6, W2_DEG), buff=0, color=CLASS_2_COLOR, stroke_width=6)

        w1_label = MathTex("W_1", color=CLASS_1_COLOR).next_to(w1_vec.get_end(), UR, buff=0.4)
        w2_label = MathTex("W_2", color=CLASS_2_COLOR).next_to(w2_vec.get_end(), UL, buff=0.4)

        boundary_line = DashedLine(ORIGIN_POINT, get_pos(1.7, BOUNDARY_DEG), color=GRAY)

        # self.play(Create(axes))
        self.play(GrowArrow(w1_vec), Write(w1_label), GrowArrow(w2_vec), Write(w2_label))
        self.play(Create(boundary_line))

        points_group = VGroup()
        c1_dots = []
        c2_dots = []

        # np.random.seed(42)

        for _ in range(10):
            # Class 1
            r = np.random.uniform(1.2, 2.0)
            a = W1_DEG + np.random.uniform(-15, 15)
            dot = Dot(get_pos(r, a), color=CLASS_1_COLOR, radius=0.08)
            c1_dots.append(dot)
            points_group.add(dot)

            # Class 2
            r = np.random.uniform(1.2, 2.0)
            a = W2_DEG + np.random.uniform(-15, 15)
            dot = Dot(get_pos(r, a), color=CLASS_2_COLOR, radius=0.08)
            c2_dots.append(dot)
            points_group.add(dot)

        self.play(FadeIn(points_group))
        self.wait(1)

        # --- ArcFace ---

        arcface_title = Tex("ArcFace Loss").to_edge(UP)
        subtitle = Tex(r"1. Normalization ($||x||=s, ||W||=1$)", font_size=36, color=YELLOW).next_to(
            arcface_title, DOWN
        )

        hypersphere = Circle(radius=VISUAL_RADIUS, color=WHITE, stroke_opacity=0.8).move_to(ORIGIN_POINT)

        w1_vec_new = Arrow(ORIGIN_POINT, get_pos(VISUAL_RADIUS, W1_DEG), buff=0, color=CLASS_1_COLOR, stroke_width=6)
        w2_vec_new = Arrow(ORIGIN_POINT, get_pos(VISUAL_RADIUS, W2_DEG), buff=0, color=CLASS_2_COLOR, stroke_width=6)

        self.play(Transform(title, arcface_title), FadeIn(subtitle))

        c1_dots_norm = []
        c2_dots_norm = []

        for dot in c1_dots:
            vec = dot.get_center() - ORIGIN_POINT
            angle_rad = np.arctan2(vec[1], vec[0])
            angle_deg = np.degrees(angle_rad)

            new_pos = get_pos(VISUAL_RADIUS, angle_deg)
            new_dot = Dot(new_pos, color=CLASS_1_COLOR, radius=0.08)
            c1_dots_norm.append(new_dot)

        for dot in c2_dots:
            vec = dot.get_center() - ORIGIN_POINT
            angle_rad = np.arctan2(vec[1], vec[0])
            angle_deg = np.degrees(angle_rad)

            new_pos = get_pos(VISUAL_RADIUS, angle_deg)
            new_dot = Dot(new_pos, color=CLASS_2_COLOR, radius=0.08)
            c2_dots_norm.append(new_dot)

        point_anims = [Transform(d, n) for d, n in zip(c1_dots, c1_dots_norm)] + [
            Transform(d, n) for d, n in zip(c2_dots, c2_dots_norm)
        ]

        self.play(
            Create(hypersphere),
            # FadeOut(axes),
            Transform(w1_vec, w1_vec_new),
            Transform(w2_vec, w2_vec_new),
            w1_label.animate.move_to(get_pos(VISUAL_RADIUS + 0.6, W1_DEG)),
            w2_label.animate.move_to(get_pos(VISUAL_RADIUS + 0.6, W2_DEG)),
            *point_anims,
            run_time=2,
        )
        self.wait(1)

        # --- Margin ---

        subtitle_margin = Tex(r"2. Additive Angular Margin ($m$)", font_size=36, color=YELLOW).next_to(
            arcface_title, DOWN
        )
        self.play(ReplacementTransform(subtitle, subtitle_margin))

        start_angle_rad = np.deg2rad(BOUNDARY_DEG - MARGIN_SIZE)
        angle_diff_rad = np.deg2rad(2 * MARGIN_SIZE)

        margin_sector = AnnularSector(
            inner_radius=0,
            outer_radius=VISUAL_RADIUS,
            angle=angle_diff_rad,
            start_angle=start_angle_rad,
            color=MARGIN_COLOR,
            fill_opacity=0.4,
        ).move_to(UP + np.array([0, 0.1, 0]))

        margin_text = Tex("Margin", font_size=24, color=WHITE).move_to(get_pos(VISUAL_RADIUS * 0.6, BOUNDARY_DEG))

        self.play(FadeOut(boundary_line), FadeIn(margin_sector), Write(margin_text))

        c1_dots_compact = []
        c2_dots_compact = []

        for _ in range(len(c1_dots)):
            new_a = W1_DEG + np.random.uniform(-6, 6)
            new_pos = get_pos(VISUAL_RADIUS, new_a)
            new_dot = Dot(new_pos, color=CLASS_1_COLOR, radius=0.08)
            c1_dots_compact.append(new_dot)

        for _ in range(len(c2_dots)):
            new_a = W2_DEG + np.random.uniform(-6, 6)
            new_pos = get_pos(VISUAL_RADIUS, new_a)
            new_dot = Dot(new_pos, color=CLASS_2_COLOR, radius=0.08)
            c2_dots_compact.append(new_dot)

        compact_anims = [Transform(d, n) for d, n in zip(c1_dots, c1_dots_compact)] + [
            Transform(d, n) for d, n in zip(c2_dots, c2_dots_compact)
        ]

        self.play(*compact_anims, run_time=2, rate_func=smooth)

        final_text = Tex("High Intra-class Compactness", font_size=32, color=GREEN).to_edge(DOWN)
        self.play(Write(final_text))

        self.wait(3)
