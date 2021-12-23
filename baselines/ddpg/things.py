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

class ResultPlot:
    def __init__(self):
        self.rwd=[]
        self.asset =[]
        plt.ion()
        self.fig = plt.figure(figsize=(5, 8))
        self.init_fig()
        plt.legend()

    def init_fig(self):
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(413)
        ax3 = self.fig.add_subplot(414)


        line11, = ax1.plot([], [], 'g', label='Account cash balance', alpha=0.8)
        line12, = ax1.plot([], [], 'r', label='Unrealized value', alpha=0.8)
        line13, = ax1.plot([], [], 'b', label='Total asset', alpha=0.6)
        line14, = ax1.plot([], [], 'gray', label='Reward', alpha=0.6)
        line21, = ax2.plot([], [], label='Reward')
        line22, = ax2.plot([], [], label='Asset')
        line31, = ax3.plot([], [], label='Position')

        ax1.legend()
        ax2.legend()
        ax3.legend()
        self.plots = {ax1:[line11, line12, line13, line14],
                    ax2:[line21, line22],
                    ax3:[line31]}
        self.axes = [ax1, ax2, ax3]

    def _update_data(self, a_idx, l_idx, x, y):
        if (isinstance(y, pd.Series)): y = y.tolist()
        ax = self.axes[a_idx]
        line = self.plots[ax][l_idx]
        x = range(0, len(x))
        line.set_data(x, y)

    def refresh(self):

        def lim(lines):
            xlim = (0, 0)
            ylim = (0, 0)
            for line in lines:
                xx, yy = line.get_data()

                xlim = (min(xx[0], xlim[0]), max(xx[-1], xlim[1]))
                ylim = (min(min(yy), ylim[0]), max(max(yy), ylim[1]))

            xpad = max(1, (xlim[1] - xlim[0]) * 0.1)
            ypad = max(1, (ylim[1] - ylim[0]) * 0.1)
            xlim = (xlim[0]- xpad, xlim[1]+xpad)
            ylim = (ylim[0] -ypad, ylim[1] +ypad)
            return xlim, ylim

        for ax in self.axes:
            xlim, ylim = lim(self.plots[ax])
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.1)

    def update(self, iteration, trading_book):

        rwd = trading_book["CumReward"][-1]
        asset = trading_book["Total asset"][-1]
        self.rwd.append(rwd)
        self.asset.append(asset)
        self.axes[0].set_title('Train iteration {}'.format(iteration), fontsize=11)
        self._update_data(0,0,trading_book.index, trading_book["Cash balance"])
        self._update_data(0,1,trading_book.index, trading_book["Unrealized value"])
        self._update_data(0,2,trading_book.index, trading_book["Total asset"])
        self._update_data(0,3,trading_book.index, trading_book["CumReward"])
        self._update_data(1,0,range(0, iteration),  self.rwd)
        self._update_data(1,1,range(0, iteration),  self.asset)
        self._update_data(2,0,trading_book.index,  trading_book["Position"])
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