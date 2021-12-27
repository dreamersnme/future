import time

import numpy as np
import pandas as pd


def CHECK_MPI_SINGLE():
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0
    if rank != 0:
        raise Exception("CANNOT USE MPI")





def isnan(xx):
    xx =  np.array(xx) if type(xx)==list else np.array([xx])
    return np.isnan(xx).any()


def check_NAN(item, see = []):
    for i in item :
        if isnan(i):
            see = see + item
            for s in see:
                print("================================================")
                print("================================================", isnan(s))
                print(s)
            raise Exception("NNNNNNNNNNNNNAAAAAAAAAAAAAANNNNNNN")




import matplotlib.pyplot as plt
import seaborn as sns

PLOTS={}

class BoxPlot:
    def __init__(self, name, ax, labels):
        self.ax = ax
        self.labels = labels
        PLOTS[name] = self

    def update(self, box_plot_data):
        self.ax.cla()
        self.ax.tick_params(axis='y', labelsize=6)
        self.ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        sample_sizes = [len(s) for s in box_plot_data ]
        widths = [0.9 * s / max(sample_sizes) for s in sample_sizes]
        self.ax.boxplot(box_plot_data, whis=2, patch_artist=True,labels=self.labels, widths = widths, flierprops={'marker': '+'})

#
# class BoxPlot:
#     def __init__(self, name, ax, labels):
#         self.ax = ax
#         self.labels = labels
#         PLOTS[name] = self
#
#     def update(self, box_plot_data):
#         self.ax.cla()
#         data = pd.DataFrame(dict(zip(self.labels, box_plot_data)))
#         sns.violinplot(y='Asset',
#               data=box_plot_data,
#               scale='count', split=True,
#               palette='seismic',
#               inner='quartile',ax=self.ax)




class APlot():

    def __init__(self, name, ax, line_def, legend =True):
        self.ax = ax
        PLOTS[name] = self
        self.lines =[]
        for l in line_def:
            line,  = ax.plot([], [], **l)
            self.lines.append(line)
        if legend: ax.legend()
        self.ax.tick_params(axis='y', labelsize=6)

    def update(self, l_idx, x, y):
        line = self.lines[l_idx]
        line.set_data(x, y)

    def getLims(self):
        xlim = (1e6, -1e6)
        ylim = (1e6, -1e6)
        for line in self.lines:
            xx, yy = line.get_data()
            xlim = (min(xx[0], xlim[0]), max(xx[-1], xlim[1]))
            ylim = (min(min(yy), ylim[0]), max(max(yy), ylim[1]))

        xpad = (xlim[1] - xlim[0]) * 0.05
        ypad = (ylim[1] - ylim[0]) * 0.1
        xlim = (xlim[0]- xpad, xlim[1]+xpad)
        ylim = (ylim[0]- ypad, ylim[1] +ypad)
        return xlim, ylim

    def refresh(self):

        x, y = self.getLims()
        self.ax.set_xlim(x[0], x[1])
        self.ax.set_ylim(y[0], y[1])



class ResultPlot:
    def __init__(self):
        self.rwd=[]
        self.asset =[]
        self.risk =[]
        plt.ion()
        self.fig = plt.figure(figsize=(8, 9))
        self.init_fig()

    def init_fig(self):
        lines =[
            {'color':'g', 'label':'Account', 'alpha':0.8},
            {'color': 'r', 'label': 'Unrealized', 'alpha': 0.8},
            {'color': 'b', 'label': 'Total', 'alpha': 0.6},
            {'color': 'gray', 'label': 'Reward', 'alpha': 0.6}]
        APlot("all",  plt.subplot2grid((5,2), (0, 0), colspan=2, rowspan=2), lines)


        lines = [{'label': 'Position', 'lw':0.5}]
        pp=APlot("pos",  plt.subplot2grid((5,2), (2, 0), colspan=2), lines, legend=False)
        lines = [{'color':'r', 'label': 'Unit', 'lw':0.5}]
        APlot("unit", pp.ax.twinx(), lines)

        # lines =[{'label':'Reward'}, {'label':'Asset'} ]
        lines = [{'label': 'Reward'}]
        pp = APlot("rwd", plt.subplot2grid((5, 2), (3, 0)), lines, legend=False)
        lines = [{'label': 'Asset', 'color':'orange'}]
        APlot("ass", pp.ax.twinx(),  lines)

        lines = [{'label': 'Reward'}]
        pp=APlot("rwd_d", plt.subplot2grid((5, 2), (3, 1)), lines)
        lines = [{'label': 'Asset', 'color':'orange'}]
        APlot("ass_d", pp.ax.twinx(), lines, legend=False)

        BoxPlot("posneg", plt.subplot2grid((5, 2), (4, 0)), ['Pos', 'Neg'])

        lines = [{'label': 'Risk', 'color': 'orange'}]
        APlot("risk_d", plt.subplot2grid((5, 2), (4, 1)), lines)

        self.fig.tight_layout()

    def _update_line(self, name, l_idx, x, y):
        if (isinstance(y, pd.Series)): y = y.tolist()
        x = range(1, len(x)+1)

        PLOTS[name].update(l_idx, x, y)

    def _update_box(self, name, data):
        PLOTS[name].update(data)


    def refresh(self):
        for key, ax in PLOTS.items():
            if isinstance(ax, APlot): ax.refresh()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.1)
        
    def update(self, iteration, info):
        trading_book = info['log']
        risk = info['risk']
        pos = info['pos']
        neg = info['neg']
        self.rwd.append(trading_book["CumReward"][-1])
        self.asset.append(trading_book["Total asset"][-1])
        self.risk.append(risk)



        self.fig.suptitle('Train iteration {}'.format(iteration), fontsize=11)
        self._update_line("all",0,trading_book.index, trading_book["Cash balance"])
        self._update_line("all",1,trading_book.index, trading_book["Unrealized value"])
        self._update_line("all",2,trading_book.index, trading_book["Total asset"])
        self._update_line("all",3,trading_book.index, trading_book["CumReward"])

        self._update_line('pos', 0, trading_book.index, trading_book["Position"])
        self._update_line("unit", 0, trading_book.index, trading_book["Unit"])

        iter_range = range(0, iteration)
        self._update_line('rwd',0, iter_range,  self.rwd)
        self._update_line('ass',0, iter_range,  self.asset)

        detail_cnt = min(20, iteration)
        detail_start = iteration-detail_cnt
        detail_range = range(detail_start, iteration)
        self._update_line('rwd_d', 0, detail_range, self.rwd[-detail_cnt:])
        self._update_line('ass_d', 0, detail_range, self.asset[-detail_cnt:])
        self._update_line('risk_d', 0, detail_range, self.risk[-detail_cnt:])


        self._update_box('posneg', [pos, neg])

        self.refresh()
        self.fig.savefig('./train_result/fig/evolution_train{}.png'.format(iteration))

#
#     # def update(self, iteration, trading_book):
#     #
#     #     rwd = trading_book["CumReward"][-1]
#     #     asset = trading_book["Total asset"][-1]
#     #     self.rwd.append(rwd)
#     #     self.asset.append(asset)
#     #
#     #     self.line11.set_data(trading_book.index, trading_book["Cash balance"])
#     #     self.line12.set_data(trading_book.index, trading_book["Unrealized value"])
#     #     self.line13.set_data(trading_book.index, trading_book["Total asset"])
#     #     self.line14.set_data(trading_book.index, trading_book["CumReward"])
#     #
#     #     self.line21.set_data(range(0, iteration),  self.rwd)
#     #     self.line21.set_data(range(0, iteration),  self.asset)
#     #
#     #     self.line31.set_data(trading_book.index,  trading_book["Position"])
#     #     self.fig.canvas.draw()
#     #     self.fig.canvas.flush_events()
#     #     time.sleep(0.1)
#     #     self.fig.savefig('./train_result/evolution_train{}.png'.format(iteration))
#
# class ResultPlot2:
#     def __init__(self):
#         self.rwd=[]
#         self.asset =[]
#     def update(self, iteration, trading_book):
#         plt.close('all')
#
#         plt.figure(figsize=(5, 8))
#         rwd = trading_book["CumReward"][-1]
#         asset = trading_book["Total asset"][-1]
#         self.rwd.append(rwd)
#         self.asset.append(asset)
#
#         plt.subplot(211)
#         # Visualize results
#         plt.title('Train iteration {}'.format(iteration), fontsize=11)
#         plt.plot(trading_book.index, trading_book["Cash balance"], 'g', label='Account cash balance', alpha=0.8)
#         plt.plot(trading_book.index, trading_book["Unrealized value"], 'r', label='Unrealized value', alpha=0.8)
#         plt.plot(trading_book.index, trading_book["Total asset"], 'b', label='Total asset', alpha=0.6)
#         plt.plot(trading_book.index, trading_book["CumReward"], 'gray', label='Reward', alpha=0.6)
#
#         plt.xlabel('Timeline', fontsize=10)
#         plt.ylabel('Value', fontsize=10)
#
#         plt.subplot(413)
#
#         plt.plot(self.rwd, label='Reward')
#         plt.plot(self.asset, label='Asset')
#         plt.subplot(414)
#
#         plt.plot(trading_book.index,  trading_book["Position"], label='Position',)
#         plt.legend()
#         plt.tight_layout()
#         plt.show(block=False)
#         plt.pause(0.1)
#         plt.savefig('./train_result/evolution_train{}.png'.format(iteration))
#         # plt.close()