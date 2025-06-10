from tools.plot_common import Figdata, show_figs

class AnmScoreFigData():
    def __init__(self):
        self.figdatas = []

    def anm_score_to_figdata(self, scores, title=""):
        anm_scores = [x[1] for x in scores if x[0] == 1]  # anomaly first
        nml_scores = [x[1] for x in scores if x[0] == 0]

        figdata = Figdata(
            data=anm_scores,
            data2=nml_scores,
            type="boxplot",
            labels=["anm", "nml"],          # match order left→right
            ylabel="score ↑ (higher=worse)",
            title=title
        )
        return figdata


    def append_figdata(self, figdata):
        self.figdatas.append(figdata)


    def show_fig(self, title="anm_score", export_dir="results", is_display_console=False):
        show_figs(
            *self.figdatas,
            fold_interval=len(self.figdatas),
            sup_title=title,
            export_path=f"{export_dir}/{title}.png",
            is_display_console=is_display_console
        )
