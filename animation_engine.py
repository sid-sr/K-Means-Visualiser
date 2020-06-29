from manimlib.imports import *
import csv
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class KMeansClustering(object):

    def __init__(self, k):
        self.k = k
        self.centers = None
        self.clusters = None
        self.cluster_vals = []
        self.center_vals = []

    def plot_state(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.clusters)
        plt.scatter(self.centers[:, 0], self.centers[:, 1], c='r', s=70)
        plt.grid()

    def fit(self, X, runs=10, plot=False, plot_final=True, num_iter=100, plot_freq=0.1):
        best_var = 10**9
        for _ in range(runs):
            fail = False
            centers = np.random.randn(self.k, X.shape[1]) * 3
            for iter_ in range(num_iter):
                arr = np.zeros((X.shape[0], self.k))
                for i, center in enumerate(centers, 0):
                    arr[:, i] = (((X - center) ** 2).sum(axis=1) ** 0.5)

                self.clusters = np.argmin(arr, axis=1)
                self.cluster_vals.append(self.clusters)

                if plot and iter_ % int(num_iter * plot_freq) == 0:
                    self.plot_state(X)
                    plt.title("Iteration " + str(iter_))
                    plt.show()

                for cno in range(self.k):
                    try:
                        centers[cno] = X[self.clusters == cno, :].mean(axis=0)
                    except:
                        fail = True
                        break
                self.center_vals.append(np.copy(centers))

            if fail:
                continue

            var = 0
            for center in centers:
                var += np.mean(((X - center) ** 2)) ** 0.5
            if var < best_var:
                best_var = var
                self.centers = centers

        if plot_final and not fail:
            self.plot_state(X)
            plt.title("Iteration " + str(iter_ + 1))
            plt.show()


class KMeansAnim(GraphScene):
    CONFIG = {
        "x_min": -5,
        "x_max": 5,
        "y_min": -5,
        "y_max": 5,
        "graph_origin": ORIGIN,
        "function_color": WHITE,
        "axes_color": WHITE
    }

    CLUSTER_COLORS = [RED, GREEN, BLUE]

    def __init__(self, **kwargs):
        self.coords = []
        self.load_data('manim_test_data.csv')
        self.num_iter = 8
        GraphScene.__init__(self, **kwargs)

    def gen_dots(self, t_stamp):
        ret = []
        for coord, color in zip(self.coords, self.model.cluster_vals[t_stamp]):
            dot = SmallDot(coord[:2]+[0])
            dot.set_color(KMeansAnim.CLUSTER_COLORS[color])
            ret.append(dot)
        return ret

    def gen_centers(self, t_stamp):
        cents = []
        for center in self.model.center_vals[t_stamp]:
            point = Dot(list(center[:2]) + [0])
            cents.append(point)
        return cents

    def ret_centers_formatted(self, t_stamp):
        ret = []
        centers = self.model.center_vals[t_stamp]
        for center in centers:
            st = f"[{round(center[0], 2)}, {round(center[1], 2)}]"
            ret.append(st)
        return ret

    def gen_ctexts(self, t_stamp):
        ctext = self.ret_centers_formatted(t_stamp)
        center_text_0 = TextMobject(ctext[0], color=RED)
        center_text_0.scale(0.5)
        center_text_1 = TextMobject(ctext[1], color=GREEN)
        center_text_1.scale(0.5)
        center_text_2 = TextMobject(ctext[2], color=BLUE)
        center_text_2.scale(0.5)

        center_text_0.shift(2*DOWN, 6*LEFT)
        center_text_1.next_to(center_text_0, DOWN)
        center_text_2.next_to(center_text_1, DOWN)
        return center_text_0, center_text_1, center_text_2

    def disp_text(self, text, pos1, pos2, col1=WHITE, col2=None):
        ktitle = TextMobject(text)
        ktitle.shift(pos1, pos2)
        ktitle.scale(0.7)
        if not col2:
            col2 = col1
        ktitle.set_color_by_gradient(col1, col2)
        self.play(Write(ktitle))

    def construct(self):
        # self.setup_axes(animate=True)
        self.run_kmeans()

        title = TextMobject("K-Means Clustering:")
        title.set_color_by_gradient(BLUE, PURPLE)
        title.shift(3.5*UP)
        self.play(Write(title))

        # initial points
        old_dot_list = self.gen_dots(0)
        old_dots = VGroup(*old_dot_list)
        self.play(ShowCreation(old_dots))

        # initial centers:
        old_centers_list = self.gen_centers(0)
        old_centers = VGroup(*old_centers_list)
        self.play(ShowCreation(old_centers))

        self.disp_text("Centers:", 1.5*DOWN, 6*LEFT)
        self.disp_text("k=3", 1.5*DOWN, 6*RIGHT)
        self.disp_text("n=100", 2*DOWN, 6*RIGHT)

        old_c0, old_c1, old_c2 = self.gen_ctexts(0)
        self.play(Write(old_c0))
        self.play(Write(old_c1))
        self.play(Write(old_c2))

        old_vg = VGroup(old_dots, old_centers, old_c0, old_c1, old_c2)

        # transformation of the points
        for t in range(1, self.num_iter):
            dot_list = self.gen_dots(t)
            dots = VGroup(*dot_list)
            centers_list = self.gen_centers(t)
            centers = VGroup(*centers_list)
            c0, c1, c2 = self.gen_ctexts(t)
            vg = VGroup(dots, centers, c0, c1, c2)

            self.play(ReplacementTransform(old_vg, vg))
            old_vg = vg
            self.wait(0.5)

    def load_data(self, file_name):
        with open(f'{file_name}', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x, y = row
                self.coords.append([float(x)/3, float(y)/3, 0])
        file.close()

    def run_kmeans(self):
        self.model = KMeansClustering(3)
        self.model.fit(np.array(self.coords)[:, :2], plot_final=False,
                       num_iter=self.num_iter, runs=1)
