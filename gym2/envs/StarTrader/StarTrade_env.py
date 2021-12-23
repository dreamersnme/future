# --------------------------- IMPORT LIBRARIES -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from gym.utils import seeding
import gym
from gym import spaces
import data_preprocessing as dp
import math

# ------------------------- GLOBAL PARAMETERS -------------------------
# Start and end period of historical data in question
START_TRAIN = datetime(2008, 12, 31)
END_TRAIN = datetime(2017, 2, 12)
START_TEST = datetime(2017, 2, 12)
END_TEST = datetime(2019, 2, 22)

STARTING_ACC_BALANCE = 0
MAX_TRADE = 2


DJI = dp.DJI
DJI_N = dp.DJI_N
CONTEXT_DATA = dp.CONTEXT_DATA
CONTEXT_DATA_N = dp.CONTEXT_DATA_N


NUMBER_OF_STOCKS = len(DJI)

# # Pools of stocks to trade
# DJI = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ',
#        'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'UTX', 'UNH', 'VZ', 'WMT']
#
# DJI_N = ['3M','American Express', 'Apple','Boeing','Caterpillar','Chevron','Cisco Systems','Coca-Cola','Disney'
#          ,'ExxonMobil','General Electric','Goldman Sachs','Home Depot','IBM','Intel','Johnson & Johnson',
#          'JPMorgan Chase','McDonalds','Merck','Microsoft','NIKE','Pfizer','Procter & Gamble',
#          'United Technologies','UnitedHealth Group','Verizon Communications','Wal Mart']
#
# #Market and macroeconomic data to be used as context data
# CONTEXT_DATA = ['^GSPC', '^DJI', '^IXIC', '^RUT', 'SPY', 'QQQ', '^VIX', 'GLD', '^TYX', '^TNX' , 'SHY', 'SHV']




# # # DJIA component stocks
# DJI = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE', 'GS',
#           'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'UTX',
#           'UNH', 'VZ', 'WMT']
# DJI_N = ['3M','American Express', 'Apple','Boeing','Caterpillar','Chevron','Cisco Systems','Coca-Cola','Disney'
#          ,'ExxonMobil','General Electric','Goldman Sachs','Home Depot','IBM','Intel','Johnson & Johnson',
#          'JPMorgan Chase','McDonalds','Merck','Microsoft','NIKE','Pfizer','Procter & Gamble',
#          'United Technologies','UnitedHealth Group','Verizon Communications','Wal Mart']
#
# CONTEXT_DATA = ['^GSPC', '^DJI', '^IXIC', '^RUT', 'SPY', 'QQQ', '^VIX', 'GLD', '^TYX', '^TNX' , 'SHY', 'SHV']
#
# CONTEXT_DATA_N = ['S&P 500', 'Dow Jones Industrial Average', 'NASDAQ Composite', 'Russell 2000', 'SPDR S&P 500 ETF',
#  'Invesco QQQ Trust', 'CBOE Volatility Index', 'SPDR Gold Shares', 'Treasury Yield 30 Years',
#  'CBOE Interest Rate 10 Year T Note', 'iShares 1-3 Year Treasury Bond ETF', 'iShares Short Treasury Bond ETF']




# # DJIA component stocks




# ------------------------------ PREPROCESSING ---------------------------------
print ("\n")
print ("############################## Welcome to the playground of Star Trader!!   ###################################")
print ("\n")
print ("Hello, I am Star, I am learning to trade like a human. In this playground, I trade stocks and optimize my portfolio.")
print ("\n")

print ("Starting to pre-process data for trading environment construction ... ")
# Data Preprocessing
dataset = dp.DataRetrieval()



dow_stocks_train, dow_stocks_test = dataset.get_all()
dow_stock_volume = dataset.components_df_v[DJI]
portfolios = dp.Trading(dow_stocks_train, dow_stocks_test, dow_stock_volume.loc[START_TEST:END_TEST])
input_states = dataset.get_feature_dataframe (DJI)

if len(CONTEXT_DATA):
    context_df = dataset.get_feature_dataframe (CONTEXT_DATA)
    input_states = pd.concat([context_df, input_states], axis=1)
input_states = input_states.dropna()


input_states.to_csv('./data/ddpg_input_states.csv')
# Without context data
#input_states = feature_df
feature_length = len(input_states.columns)
data_length = len(input_states)
WORLD = dataset.components_df_o[DJI]
stock_volume = dataset.components_df_v[DJI]
WORLD.to_csv('./data/ddpg_WORLD.csv')
COMMITION = 0.2
SLIPPAGE = 1#1  # 상방 하방


print("Pre-processing and stock selection complete, trading starts now ...")
print("_______________________________________________________________________________________________________________")


# ------------------------------ CLASSES ---------------------------------

class StarTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, day = START_TRAIN):
        if NUMBER_OF_STOCKS !=1:
            raise Exception("NEED SINGLE TARGET")
        """
        Initializing the trading environment, trading parameters starting values are defined.
        """
        self.iteration = 0
        self.day = day

        # defined using Gym's Box action space function
        self.action_space = spaces.Box(low = -1, high = 1,shape = (NUMBER_OF_STOCKS,),dtype=np.int8)

        # [account balance]+[unrealized profit/loss] +[number of features, 36]+[portfolio stock of 5 stocks holdings]
        self.full_feature_length = 2 + feature_length
        print("full length", self.full_feature_length)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.full_feature_length + NUMBER_OF_STOCKS,))
        self.reset()
    def reset(self):
        """
        Reset the environment once an episode end.

        """
        self.done = False
        self.up_cnt = 0
        self.down_cnt =0
        self.total_neg = 0
        self.total_commition = 0
        self.acc_balance = [STARTING_ACC_BALANCE]
        self.total_asset = self.acc_balance
        self.reward_log = [0]
        self.position = np.zeros((1, NUMBER_OF_STOCKS)).flatten()
        self.position_log = [0]

        unrealized_pnl = 0.0
        self.unrealized_asset = [unrealized_pnl]
        self.buy_price = np.zeros((1, NUMBER_OF_STOCKS)).flatten()
        self.day = START_TRAIN

        self.day, self.data = self.skip_day (input_states)
        self.timeline = [self.day]
        self.state = self.acc_balance + [unrealized_pnl] + self.data.values.tolist() + [0 for i in range(NUMBER_OF_STOCKS)]
        self.iteration += 1
        self.reward = 0

        return self.state


    def skip_day(self, input_st):
        # Sliding the timeline window day-by-day, skipping the non-trading day as data is not available
        wrong_day = True
        add_day = 0
        while wrong_day:
            try:
                temp_date = self.day + timedelta(days=add_day)
                self_data = input_st.loc[temp_date]
                self_day = temp_date
                wrong_day = False
            except:
                add_day += 1
        return self_day, self_data


    def get_trade_num(self, normed_action, max_trade):
        action = normed_action + 1
        action = action * ( float(2*MAX_TRADE +1)/ 2.0) - MAX_TRADE
        action = math.floor(action)
        return min(action, MAX_TRADE)


    def _clean(self, cur_share, new_share):

        shift = new_share - cur_share
        direction = -1 if shift < 0 else 1 if shift > 0 else 0
        clean_all = False

        if cur_share == 0 or shift == 0:
            return direction, 0, shift, clean_all
        if (cur_share < 0 and shift < 0) or (cur_share > 0 and shift > 0):
            return direction, 0, shift, clean_all

        clean_all = (abs (cur_share) <= abs (shift))

        cleaned = -cur_share if clean_all else shift
        left = shift - cleaned

        return direction, cleaned, left, clean_all

    def _trade(self, idx, action):

        state_idx = idx + self.full_feature_length
        cur_share = self.state[state_idx]
        new_share = self.get_trade_num(action ,MAX_TRADE)

        buy_direction, cleaned, left_buy, cleaned_all = self._clean(cur_share, new_share)

        if buy_direction ==0:
            return cur_share


        assert cleaned + left_buy == (new_share - cur_share)

        commition = abs(cleaned + left_buy) * COMMITION
        self.state[0] -= commition
        self.total_commition += commition

        previous_price = self.buy_price[idx]
        transacted_price = WORLD.loc[self.day][idx] + (SLIPPAGE * buy_direction)  # 살땐 비싸게, 팔땐 싸게

        if cleaned == 0: #clean은 하지 않았으므로, 같은 방향의 변동
            assert cur_share+left_buy == new_share
            self.buy_price[idx] = (abs(cur_share) * previous_price +  abs(left_buy)*transacted_price) / abs(new_share)
        else:
            realized = -cleaned * (transacted_price - previous_price)
            thresold = abs(cleaned)*COMMITION
            if realized > thresold:
                self.up_cnt += abs(cleaned)
            elif realized < thresold:
                self.down_cnt += abs(cleaned)
            else: pass

            self.state[0] += realized #청산이익

            if not cleaned_all:
                pass
                # self.buy_price[idx] 안바뀜, 일부청산 이기 때문
            else:# 모두 청산하여 예전 가격 필요없음. 더 거래시 현재 가격
                self.buy_price[idx] = transacted_price

        self.state[state_idx] = new_share
        return new_share


    def step_done(self, actions):
        print ("@@@@@@@@@@@@@@@@@")
        print ("Iteration", self.iteration - 1)



        # Construct trading book and save to a spreadsheet for analysis
        trading_book = pd.DataFrame (index=self.timeline, columns=["Cash balance", "Unrealized value", "Total asset", "Rewards", "CumReward", "Position"])
        trading_book["Cash balance"] = self.acc_balance
        trading_book["Unrealized value"] = self.unrealized_asset
        trading_book["Total asset"] = self.total_asset
        trading_book["Rewards"] = self.reward_log
        trading_book["CumReward"] = trading_book["Rewards"].cumsum().fillna(0)
        trading_book["Position"]  = self.position_log
        trading_book.to_csv ('./train_result/trading_book_train_{}.csv'.format (self.iteration - 1))

        total_reward = trading_book["CumReward"][-1]
        total_asset = trading_book["Total asset"][-1]
        print("UP: {}, DOWN: {}, Commition: {}".format(self.up_cnt, self.down_cnt, self.total_commition))
        print("Acc: {}, Rwd: {}, Neg: {}".format(total_asset, total_reward, self.total_neg))

        kpi = dp.MathCalc.calc_kpi(trading_book)
        kpi.to_csv('./train_result/kpi_train_{}.csv'.format(self.iteration - 1))
        print("===============================================================================================")
        print(kpi)
        print("===============================================================================================")


        return self.state, self.reward, self.done, {"log": trading_book}

    def step(self, actions):
        # Episode ends when timestep reaches the last day in feature data
        self.done = self.day >= END_TRAIN
        # Uncomment below to run a quick test
        #self.done = self.day >= START_TRAIN + timedelta(days=10)

        # If it is the last step, plot trading performance

        if self.done:
            return self.step_done(actions)

        else:
            return self.step_normal(actions)


    def _cur_unrealized(self, cur_buy_stat):
        cur_buy_stat = np.array(cur_buy_stat)
        total = np.sum(abs(cur_buy_stat))
        if total ==0 : return 0
        now = np.sum (  (WORLD.loc[self.day] - np.array(self.buy_price)) * cur_buy_stat )
        now = now - (SLIPPAGE+COMMITION) * total
        return now


    def step_normal(self, actions):

        # Total asset is account balance + unrealized_pnl
        unrealized_pnl = self.state[1]
        total_asset_starting = self.state[0] + unrealized_pnl

        for stock_idx, action in enumerate(actions):
            position = self._trade (stock_idx, action)
        self.position_log = np.append (self.position_log, position)


        # Update date and skip some date since not every day is trading day
        self.day += timedelta (days=1)
        self.day, self.data = self.skip_day (input_states)
        cur_buy_stat = self.state[self.full_feature_length:]

        # Calculate unrealized profit and loss for existing stock holdings
        unrealized_pnl = self._cur_unrealized(cur_buy_stat)
        # print(unrealized_pnl)

        # next state space
        self.state = [self.state[0]] + [unrealized_pnl] + self.data.values.tolist () + cur_buy_stat

        total_asset_ending = self.state[0] + unrealized_pnl

        step_profit = total_asset_ending - total_asset_starting
        if step_profit <0: self.total_neg += step_profit


        # Update account balance statement
        self.acc_balance = np.append (self.acc_balance, self.state[0])
        self.unrealized_asset = np.append (self.unrealized_asset, unrealized_pnl)

        # Update total asset statement
        self.total_asset = np.append (self.total_asset, total_asset_ending)



        # Update timeline
        self.timeline = np.append (self.timeline, self.day)

        # Get the agent to consider gain-to-pain or lake ratio and be responsible for it if it has traded long enough
        self.reward = self.cal_reward(total_asset_starting, total_asset_ending, cur_buy_stat)
        # print (self.state)
        # print(self.reward)
        # print('p ', WORLD.loc[self.day])

        self.reward_log = np.append (self.reward_log, self.reward)

        return self.state, self.reward, self.done, {}

    def remain_risk(self, action_power):
        return 0.01 * (pow(action_power + 1, 3) -1)


    def cal_reward(self, total_asset_starting, total_asset_ending, cur_buy_stat):
        action = np.array(cur_buy_stat)
        action_power = np.mean(abs(action/ MAX_TRADE))
        profit = (total_asset_ending - total_asset_starting)/ MAX_TRADE
        if profit<0:
            profit = min(profit, -1 * pow(abs(profit), 1.7))
        else:
            profit = max(profit, pow(profit, 1.2))

        risk = self.remain_risk(action_power)
        return profit - risk + 0.1


        # stability_tick = 20
        # if len (self.total_asset) < stability_tick:
        #     return profit - risk
        #
        #
        # last_balance = self.total_asset[-stability_tick:]
        # last_profit = pd.Series (self.reward_log[-stability_tick:])
        #
        # GPR = dp.MathCalc.calc_gain_to_pain (last_profit)
        # LAKE = - dp.MathCalc.calc_lake_ratio (last_balance)
        # print("PPPPP", profit)
        # print("R", risk)
        # print("G ", GPR)
        # print("L", LAKE)
        #
        # reward = profit - risk + GPR + LAKE
            # + (50 * dp.MathCalc.sharpe_ratio(pd.Series(returns)))
        return reward


    
    def render(self, mode='human'):
        """
        Render the environment with current state.
        """
        return self.state

    def _seed(self, seed=None):
        """
        Seed the iteration.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
