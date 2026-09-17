# -*- coding: utf-8 -*-
import numpy as np
import pickle
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

# hyper param
INFO_LEN = 41
PAST_RACE = 7
HANDI_ADJUST = 0.2

# for file
DUMP_RACE_FILE = './data/dump_race2010_2025.pickle'
DUMP_HORSE_FILE = './data/dump_horse2010_2025.pickle'
DUMP_LIST_FILE = './data/dump_list2010_2025.pickle'
DUMP_KISYU_FILE = './data/dump_kisyu2010_2025.pickle'

class Magokoro():
    def __init__(self, fromdate='20181225', todate='20181231', train=True, dbcon="dbname=everydb2 user=postgres password=okabe1171"):
        #self.connection = psycopg2.connect(dbcon)
        self.fromdate = fromdate
        self.todate = todate
        self.racekey = {}
        self.horsekey = {}
        self.racelist = []
        self.kisyukey = {}

        self.classtable = [] #    G1,   G2,   G3, OPEN, 1600, 1000, 500, 未勝利or新馬
        self.classtable.append([-0.5,  0.2,  0.5,  0.7,  1.0,  1.0, 1.0, 1.9]) # 芝：2歳
        self.classtable.append([-1.7, -1.2, -1.0, -0.7, -0.7, -0.5, 0.2, 1.2]) # 芝：3歳
        self.classtable.append([-3.1, -2.6, -2.2, -1.9, -1.4, -0.7, 0.0, 0.0]) # 芝：3歳以上
        self.classtable.append([ 0.6,  0.6,  0.6,  0.6,  0.6,  0.6, 0.8, 1.6]) # ダート：2歳
        self.classtable.append([-1.7, -1.2, -1.2, -0.8, -0.8, -0.5, 0.3, 1.2]) # ダート：3歳
        self.classtable.append([-3.2, -2.8, -2.4, -2.0, -1.6, -0.8, 0.0, 0.0]) # ダート：3歳以上

        if train:
            self.load_dict()
        
    def make_race_dict(self, r):
        key = r['year'] + r['monthday'] + r['jyocd'] + r['racenum']
        if self.racekey.get(key) is None:
            self.racekey[key] = {}
            self.racekey[key]['ymd'] = r['year'] + r['monthday']
            self.racekey[key]['year'] = r['year']
            self.racekey[key]['monthday'] = r['monthday']
            self.racekey[key]['key'] = key
            self.racekey[key]['jyocd'] = int(r['jyocd'])
            self.racekey[key]['racenum'] = int(r['racenum'])
            self.racekey[key]['hondai'] = r['ryakusyo10']
            self.racekey[key]['trackcd'] = int(r['trackcd'])
            self.racekey[key]['gradecd'] = r['gradecd']
            self.racekey[key]['syubetucd'] = int(r['syubetucd'])
            self.racekey[key]['jyokencd1'] = int(r['jyokencd1'])
            self.racekey[key]['jyokencd2'] = int(r['jyokencd2'])
            self.racekey[key]['jyokencd3'] = int(r['jyokencd3'])
            self.racekey[key]['jyuryocd'] = int(r['jyuryocd'])
            self.racekey[key]['kyori'] = int(r['kyori'])
            self.racekey[key]['sibababacd'] = int(r['sibababacd'])
            self.racekey[key]['dirtbabacd'] = int(r['dirtbabacd'])
            self.racekey[key]['harontimes3'] = float(r['harontimes3']) / 10.0
            self.racekey[key]['harontimel3'] = float(r['harontimel3']) / 10.0
            self.racekey[key]['hassotime'] = r['hassotime']
            self.racekey[key]['torokutosu'] = int(r['torokutosu'])
            self.racekey[key]['syussotosu'] = int(r['syussotosu'])
            self.racekey[key]['nyusentosu'] = int(r['nyusentosu'])
            self.racekey[key]['horse'] = []

        return self.racekey[key]
        
    def make_horse_dict(self, h):
        key = h['year']+h['monthday']
        if self.horsekey.get(h['kettonum']) is None:
            self.horsekey[h['kettonum']] = {}

        if self.horsekey[h['kettonum']].get(key) is None:
            self.horsekey[h['kettonum']][key] = {}
            self.horsekey[h['kettonum']][key]['kettonum'] = h['kettonum']
            self.horsekey[h['kettonum']][key]['ymd'] = key
            self.horsekey[h['kettonum']][key]['year'] = h['year']
            self.horsekey[h['kettonum']][key]['monthday'] = h['monthday']
            self.horsekey[h['kettonum']][key]['racenum'] = int(h['racenum'])
            self.horsekey[h['kettonum']][key]['key'] = h['year'] + h['monthday'] + h['jyocd'] + h['racenum']
            self.horsekey[h['kettonum']][key]['jyocd'] = int(h['jyocd'])
            self.horsekey[h['kettonum']][key]['kakuteijyuni'] = int(h['kakuteijyuni'])
            self.horsekey[h['kettonum']][key]['odds'] = float(h['odds']) / 10.0
            self.horsekey[h['kettonum']][key]['time'] = self.get_time(h['time'])
            self.horsekey[h['kettonum']][key]['futan'] = float(h['futan']) / 10.0
            self.horsekey[h['kettonum']][key]['kyakusitukubun'] = int(h['kyakusitukubun'])
            self.horsekey[h['kettonum']][key]['umaban'] = int(h['umaban'])
            self.horsekey[h['kettonum']][key]['wakuban'] = int(h['wakuban'])
            self.horsekey[h['kettonum']][key]['bataijyu'] = h['bataijyu'] # 直前まではスペースのためINTしない
            self.horsekey[h['kettonum']][key]['blinker'] = int(h['blinker'])
            self.horsekey[h['kettonum']][key]['kisyucode'] = int(h['kisyucode'])
            self.horsekey[h['kettonum']][key]['bamei'] = h['bamei']
            self.horsekey[h['kettonum']][key]['barei'] = int(h['barei'])
            self.horsekey[h['kettonum']][key]['sexcd'] = int(h['sexcd'])
            self.horsekey[h['kettonum']][key]['ninki'] = int(h['ninki'])
            self.horsekey[h['kettonum']][key]['barei'] = int(h['barei'])
            self.horsekey[h['kettonum']][key]['harontimel3'] = float(h['harontimel3']) / 10.0
            self.horsekey[h['kettonum']][key]['dmtime'] = h['dmtime']
            self.horsekey[h['kettonum']][key]['dmgosap'] = h['dmgosap']
            self.horsekey[h['kettonum']][key]['dmgosam'] = h['dmgosam']

            #print(f"update kisyu {key} {h['jyocd']} {h['racenum']}")
            if int(h['kakuteijyuni']) > 0:
                _ = self.make_kisyu_dict(h)

        return self.horsekey[h['kettonum']][key]
    
    def make_kisyu_dict(self, h):
        key = h['kisyucode'] + h['year']
        donekey = key + h['monthday'] + h['racenum']
        jun = int(h['kakuteijyuni'])
        if self.kisyukey.get(key) is None:
            self.kisyukey[key] = {}
            self.kisyukey[key]['year'] = h['year']
            self.kisyukey[key]['kisyucode'] = h['kisyucode']
            self.kisyukey[key]['chaku1'] = 0
            self.kisyukey[key]['chaku2'] = 0
            self.kisyukey[key]['chaku3'] = 0
            self.kisyukey[key]['total'] = 0
            self.kisyukey[key]['done'] = []
        
        if donekey not in self.kisyukey[key]['done']:
            self.kisyukey[key]['done'].append(donekey)
            self.kisyukey[key]['chaku1'] += 1 if jun == 1 else 0
            self.kisyukey[key]['chaku2'] += 1 if jun == 2 else 0
            self.kisyukey[key]['chaku3'] += 1 if jun == 3 else 0
            self.kisyukey[key]['total'] += 1
            self.update = True

        return self.kisyukey[key]
    
    def make_pay_dict(self, r_dict, pay):
        if pay is not None:
            r_dict['paytansyopay'] = int(pay['paytansyopay1'])
            r_dict['payfukusyopay1'] = int(pay['payfukusyopay1'])
            r_dict['payfukusyopay2'] = int(pay['payfukusyopay2'])
            if pay['payfukusyopay3'].isdecimal():
                r_dict['payfukusyopay3'] = int(pay['payfukusyopay3'])
            r_dict['payumarenpay'] = int(pay['payumarenpay1'])
            r_dict['payumatanpay'] = int(pay['payumatanpay1'])
            r_dict['paywidepay1'] = int(pay['paywidepay1'])
            r_dict['paywidepay2'] = int(pay['paywidepay2'])
            r_dict['paywidepay3'] = int(pay['paywidepay3'])
            r_dict['paysanrenpukupay'] = int(pay['paysanrenpukupay1'])
            r_dict['paysanrentanpay'] = int(pay['paysanrentanpay1'])
    
    def save_dict(self):
        with open(DUMP_RACE_FILE, mode='wb') as fo:
            pickle.dump(self.racekey, fo)
        with open(DUMP_HORSE_FILE, mode='wb') as fo:
            pickle.dump(self.horsekey, fo)
        with open(DUMP_KISYU_FILE, mode='wb') as fo:
            pickle.dump(self.kisyukey, fo)
        with open(DUMP_LIST_FILE, mode='wb') as fo:
            pickle.dump(self.racelist, fo)
    
    def load_dict(self):
        with open(DUMP_RACE_FILE, mode='rb') as fi:
            self.racekey = pickle.load(fi)
        with open(DUMP_HORSE_FILE, mode='rb') as fi:
            self.horsekey = pickle.load(fi)
        with open(DUMP_KISYU_FILE, mode='rb') as fi:
            self.kisyukey = pickle.load(fi)
        with open(DUMP_LIST_FILE, mode='rb') as fi:
            self.racelist = pickle.load(fi)

    #--------------------------------------------
    # 芝／ダートを返す(0/1)
    #--------------------------------------------
    def get_rtype(self, tcd):
        if tcd >= 10 and tcd <=22:
            return 0 # turf
        if tcd >= 23 and tcd <=29:
            return 1 # dirt
        return -1
    
    #--------------------------------------------
    # 内回り／外回りを返す(0/1)
    #--------------------------------------------
    def get_inout(self, h):
        if h in (11,14,15,17,20,21,23,24,25,27,28):
            return 0 # 内回り、内外無しコース
        else:
            return 1 # 外回り、直線

    #--------------------------------------------
    # 右回り／左回りを返す(0/1)
    #--------------------------------------------
    def get_lr(self, h):
        if h['jyocd'] in (1,2,3,6,8,9,10):
            return 0 # 右回り
        else:
            return 1 # 左回り
        
    #--------------------------------------------
    # 直線坂の有無を返す(0/1)
    #--------------------------------------------
    def get_saka(self, h):
        if h['jyocd'] in (3,5,6,7,9):
            return 0 # 坂あり
        else:
            return 1 # 坂なし
        
    #--------------------------------------------
    # 小回りの有無を返す(0/1)
    #--------------------------------------------
    def get_komawari(self, h):
        if h['jyocd'] in (1,2,3,6,10):
            return 0 # 小回り
        else:
            return 1 # 広い
            
    #--------------------------------------------
    # 馬場状態を返す(1-4)
    #--------------------------------------------
    def get_baba(self, r):
        rtype = self.get_rtype(r['trackcd'])
        if rtype == 0:
            return r['sibababacd']
        else:
            return r['dirtbabacd']

    #--------------------------------------------
    # 今回のレースと対象レースの情報をカテゴリ特徴、連続値特徴、差分特徴に分けて生成
    #--------------------------------------------
    def get_adjust_point(self, r, h, rp, p, idx, pre_ymd, interval):

        gp1 = [] # カテゴリ特徴量Embedding(過去)2
        gp2 = [] # 連続値特徴量Linear(過去)4
        gp3 = [] # 差分カテゴリ特徴量Embedding8
        gp4 = [] # 差分連続値特徴量Linear9
        gp5 = [] # 全馬共通カテゴリ特徴量Embedding(今回)5
        gp6 = [] # 全馬共通連続値特徴量Linear(今回)5
        gp7 = [] # 能力値(過去)7
        # 合計40次元 2 4 8 9 5 5 7

        #####################################
        # gp1: カテゴリ特徴量Embedding(過去)2

        # 0 競馬場10
        gp1.append(rp['jyocd']-1)

        # 1 芝/ダート2
        gp1.append(self.get_rtype(rp['trackcd']))

        #####################################
        # gp2: 連続値特徴量Linear(過去)4

        # 2 馬場状態
        gp2.append((self.get_baba(rp)-1)/3.0)

        # 3 ペース
        mse = self.get_prepace(rp)
        gp2.append(((mse[0]*1.0+mse[1]*2.0+mse[2]*3.0+mse[3]*4.0) - 1.4)/2.45)

        # 4 前走脚質
        gp2.append((p['kyakusitukubun']-1)/3.0)

        # 5 距離
        gp2.append((float(rp['kyori']) - 1000.0) / (3600.0 - 1000.0))

        #####################################
        # gp3: 差分特徴量Embedding9

        # 6 コース形状2
        gp3.append(1.0 if (r['jyocd'] != rp['jyocd']) or (r['trackcd'] != rp['trackcd']) or (r['kyori'] != rp['kyori']) else 0.0)
            
        # 7 内回り/外回り2
        gp3.append(1.0 if self.get_inout(r['trackcd']) != self.get_inout(rp['trackcd']) else 0.0)

        # 8 同一騎手2
        gp3.append(1.0 if h['kisyucode'] != p['kisyucode'] else 0.0)
            
        # 9 ブリンカー有無2
        gp3.append(1.0 if h['blinker'] != p['blinker'] else 0.0)

        # 10 坂有無2
        gp3.append(abs(self.get_saka(r) - self.get_saka(rp)))
        
        # 11 左回り/右回り2
        gp3.append(abs(self.get_lr(r) - self.get_lr(rp)))
        
        # 12 小回り2
        gp3.append(abs(self.get_komawari(r) - self.get_komawari(rp)))

        # 13 芝/ダート2
        gp3.append(abs(self.get_rtype(r['trackcd']) - self.get_rtype(rp['trackcd'])))

        # 14 マイニング予想（最も遅い予想）からどの程度乖離しているか（不利等の考慮）
        dmt = self.get_dmtime(p['dmtime']) + self.get_dmtime('0' + p['dmgosam'])
        rt = p['time'] 
        trust = 0 if rt - dmt <= 0.0 else 1
        gp3.append(trust)

        #####################################
        # gp4: 差分特徴量Linear9

        # 15 距離
        dist_diff = (((r['kyori'] - rp['kyori']) / 2600.0) + 0.77) / 1.61
        dist_diff = max(0.25, dist_diff)
        dist_diff = min(0.75, dist_diff)
        gp4.append((dist_diff - 0.25) / 0.5)

        # 16 馬場状態
        gp4.append(abs(self.get_baba(r) - self.get_baba(rp)) / 4.0 * 1.33)

        # 17 枠
        umaban_p = (18 - int(p['umaban'])) if rp['jyocd'] == 4 and rp['kyori'] == 1000 else int(p['umaban'])
        umaban_h = (18 - int(h['umaban'])) if r['jyocd'] == 4 and r['kyori'] == 1000 else int(h['umaban'])
        gp4.append((umaban_p - umaban_h + 17) / 34.0)
           
        # 18 ペース
        p1 = self.get_current_pace(r)
        p2 = self.get_prepace(rp)
        mse = [(p1[0]-p2[0]), (p1[1]-p2[1]), (p1[2]-p2[2]), (p1[3]-p2[3])]
        gp4.append((((mse[0]*1.0+mse[1]*2.0+mse[2]*3.0+mse[3]*4.0)/10.0)+0.2)*2.5)

        # 19 前走脚質
        pre_leg = (p['kyakusitukubun']-1)/3.0
        cur_leg = (self.get_preleg(h) - 1)/3.0
        gp4.append((cur_leg - pre_leg + 1.0) / 2.0)
        
        # 20 出走頭数
        gp4.append((len(r['horse']) - len(rp['horse']) + 11) / 24.0)

        # 21 レース間隔
        pre_int = self.get_ymd(pre_ymd) - self.get_ymd(p['ymd'])
        gp4.append(min((pre_int - interval)/365.0, 1.0))

        # 22 負担重量
        futan = float(p['futan']) - float(h['futan']) + 9.0
        gp4.append(futan/18.0)

        # 23 条件
        classt = abs(self.get_class_time(rp) - self.get_class_time(r)) - 2.6
        classt = max(0, min(classt/2.4 + 1.08, 1.5))
        gp4.append(classt/1.5)

        #####################################
        # gp5: 全馬共通カテゴリ特徴量Embedding(今回)5

        # 24 競馬場10
        gp5.append(int(r['jyocd'])-1)

        # 25 芝/ダート2
        gp5.append(self.get_rtype(r['trackcd']))

        # 26 坂有無2
        gp5.append(self.get_saka(r))

        # 27 左回り/右回り2
        gp5.append(self.get_lr(r))

        # 28 小回り2
        gp5.append(self.get_komawari(r))

        #####################################
        # gp6: 全馬共通連続値特徴量Linear(今回)5

        # 29 馬場状態
        gp6.append((self.get_baba(r)-1) / 3.0)

        # 30 ペース
        mse = self.get_current_pace(r)
        gp6.append(((mse[0]*1.0+mse[1]*2.0+mse[2]*3.0+mse[3]*4.0) - 1.45) / 2.4)
 
        # 31 距離
        gp6.append((float(r['kyori']) - 1000.0) / (3600.0 - 1000.0))

        # 32 条件
        gp6.append((self.get_class_time(r) + 3.1) / 4.7)
        
        # 33 出走頭数
        gp6.append(len(r['horse']) / 18.0)

        #####################################
        # gp7: 能力値(過去)7

        # 34 枠の差（大きいほど良い）
        umaban_p = (18 - int(p['umaban'])) if rp['jyocd'] == 4 and rp['kyori'] == 1000 else int(p['umaban'])
        umaban_h = (18 - int(h['umaban'])) if r['jyocd'] == 4 and r['kyori'] == 1000 else int(h['umaban'])
        gp7.append((umaban_p - umaban_h + 17) / 34.0)

        # 35 騎手成績の差（大きいほど良い）
        kisyu_diff = self.get_kisyu_rate(h) - self.get_kisyu_rate(p)
        kisyu_diff = max(-0.35, kisyu_diff)
        kisyu_diff = min(0.35, kisyu_diff)
        gp7.append((kisyu_diff + 0.35) / 0.7)

        # 36  abs(前回脚質(逃0-1追) - ペース(早0-1遅))（大きいほど良い）
        mse = self.get_current_pace(r)
        pace_g7 = ((mse[0]*1.0+mse[1]*2.0+mse[2]*3.0+mse[3]*4.0) - 1.45) / 2.4
        preleg_g7 = (self.get_preleg(h) - 1)/3.0
        gp7.append(abs(pace_g7 - preleg_g7))

        # 36 距離/Time(1秒当たり何m)（大きいほど良い）
        #dpt = (float(rp['kyori']) / p['time'] - 9.0) / 9.6
        #dpt = (float(rp['kyori']) / self.get_dmtime(p['dmtime']) - 9.0) / 9.6
        #dpt = max(0.6, dpt) - 0.6
        #gp7.append(dpt / 0.4)

        # 37 能力値（大きいほど良い）
        baset =float(p['baset'])
        baset = (baset +6.5) /50.7
        baset = min(0.25, baset) / 0.25
        gp7.append(1.0 - baset)

        # 38 負担重量（大きいほど良い）
        futan = float(p['futan']) - float(h['futan']) + 9.0
        gp7.append(futan / 18.0)

        # 39 クラスタイム差（大きいほど良い）
        classt = self.get_class_time(rp) - self.get_class_time(r) + 2.6
        gp7.append(classt/7.6)

        # 40 上がり3ハロン
        gp7.append(max((50.0 - p['harontimel3']),0.0) / 20.0)

        # 41 馬心タイム値
        #gp7.append((max(self.get_race_point(p, rp, h, r),-10.0) + 10.0) / 22.0)

        return gp1+gp2+gp3+gp4+gp5+gp6+gp7
          
    #--------------------------------------------
    # タイム計算（単位：秒）
    #--------------------------------------------
    def get_time(self, t):
        return float(t[0]) * 60.0 + float(t[-3:]) / 10.0
    
    #--------------------------------------------
    # データマイニングタイム計算（単位：秒）
    #--------------------------------------------
    def get_dmtime(self, t):
        return float(t[0]) * 60.0 + float(t[-4:]) / 100.0
        
    #--------------------------------------------
    # YMD計算（単位：日）
    #--------------------------------------------
    def get_ymd(self, ymd):
        return float(ymd[:4]) * 365.0 + float(ymd[4:6]) * 30.0 + float(ymd[-2:])

    #--------------------------------------------
    # 対象レースの年齢、クラスを返す
    #--------------------------------------------
    def get_age_joken(self, r):
        agecd = int(r['syubetucd'])
        if agecd == 11:
            age = 0
            jokencd = int(r['jyokencd1'])
        elif agecd == 12:
            age = 1
            jokencd = int(r['jyokencd2'])
        else:
            age = 2
            jokencd = int(r['jyokencd3'])
        return age, jokencd
        
    #--------------------------------------------
    # レースクラスによるタイム差を返す
    #--------------------------------------------
    def get_class_time(self, r):
        age,jokencd = self.get_age_joken(r)
        joken = -1
        if jokencd == 701 or jokencd == 702 or jokencd == 703:
            joken = 7
        elif jokencd == 5:
            joken = 6
        elif jokencd == 10:
            joken = 5
        elif jokencd == 16:
            joken = 4
        elif jokencd == 999:
            gradecd = r['gradecd']
            if gradecd == 'A':
                joken = 0
            elif gradecd == 'B':
                joken = 1
            elif gradecd == 'C':
                joken = 2
            else:
                joken = 3
        else:
            print('joken error ',r)
            a
            
        return self.classtable[(self.get_rtype(int(r['trackcd']))) * 3 + age][joken]

    #--------------------------------------------
    # 過去対象馬の過去レース脚質
    #
    # 今回レースのペース予想:前走脚質prelegの集計
    # 今回レースRc - 出走馬Hc01 - 前走脚質preleg - Hc01の前走Hp01 - 前走脚質kyakusitsu
	#	            出走馬Hc02
    #                   :
	#                   :
	#	            出走馬Hc18
    #
    # 過去レースのペース予想:過去レース出走馬の脚質kyakusitsuの集計
    # 過去レースRp - 出走馬Hp01 - 過去脚質kyakusitsu
	#	            出走馬Hp02
    #                   :
	#                   :
	#	            出走馬Hp18
    #--------------------------------------------
    def get_preleg(self, h):
        if True:
        #if h.get('preleg') is None:  
            hlist = self.horsekey[h['kettonum']]
            sort_list = sorted(hlist.items(), reverse=True)
            pre = 3
            for p in sort_list:
                if h['ymd'] > p[0] and int(p[1]['kyakusitukubun']) > 0:
                    pre = int(p[1]['kyakusitukubun'])
                    break
            h['preleg'] = pre

        return h['preleg']
       
    #--------------------------------------------
    # 今回レースrでの予想ペースを全馬の前走脚質から計算
    #--------------------------------------------
    def get_current_pace(self, r):
        # 既に辞書登録済みのポイントか
        if True:
        #if r.get('curpace') is None:     
            # 対象レース出走全馬の脚質を取得            
            legc = [0., 0., 0., 0., 0.]
            for allh_key in r['horse']:
                allh = self.horsekey[allh_key][r['ymd']]
                legc[self.get_preleg(allh)] += 1.0

            total= sum(legc)
            if total > 0:
                for i in range(5):
                    legc[i] /= total
            r['curpace'] = legc[-4:]

        return r['curpace']
    
    #--------------------------------------------
    # 過去レースrでの予想ペースを全馬の脚質から計算
    #--------------------------------------------
    def get_prepace(self, r):
        # 既に辞書登録済みのポイントか
        if True:
        #if r.get('prepace') is None:     
            # 対象レース出走全馬の脚質を取得            
            legc = [0., 0., 0., 0., 0.]
            for allh_key in r['horse']:
                allh = self.horsekey[allh_key][r['ymd']]
                legc[allh['kyakusitukubun']] += 1.0
            total= sum(legc)
            if total > 0:
                for i in range(5):
                    legc[i] /= total
            r['prepace'] = legc[-4:]

        return r['prepace']

    #--------------------------------------------
    # 対象馬hの対象レースrでの評価ポイントを計算
    #--------------------------------------------
    def get_race_point(self, h, r, curh, curr):
        baset =float(h['baset']) # 小さいほど良い
        # 今回のレースの負担重量、クラスによるタイム差を計算
        futan = (float(curh['futan']) - float(h['futan'])) * HANDI_ADJUST # 小さいほど良い
        classt = self.get_class_time(curr) - self.get_class_time(r) # 大きいほど良い

        return 5.0 - (baset + futan - classt)
        
    #--------------------------------------------
    # 騎手成績計算
    #--------------------------------------------
    def get_kisyu_rate(self, h):      
        total = 0
        rate1 = 0
        rate2 = 0
        rate3 = 0
        for i in range(2):
            key = f"{int(h['kisyucode']):05}{int(h['ymd'][:4]) - i}"
            if self.kisyukey.get(key) is not None:
                rate1 += self.kisyukey[key]['chaku1']
                rate2 += self.kisyukey[key]['chaku2']
                rate3 += self.kisyukey[key]['chaku3']
                total += self.kisyukey[key]['total']

        if total > 0:
            #print(f'kisyu_rate={(rate1 + rate2/2 + rate3/3) / total}')
            return (rate1 + rate2/2 + rate3/3) / total
        else: # 新人等
            return 0.0

    #--------------------------------------------
    # 対象レースに出走している対象馬の評価
    #--------------------------------------------
    def evaluate_horse(self, r, h, use_701=True, is_training=True):
        pv = []
        padding_mask = []

        # hのrレース以前の同一トラックレースを最大n走抽出する
        hlist = self.horsekey[h['kettonum']]
        sort_list = sorted(hlist.items(), reverse=True)
        first = True

        idx = 0
        interval = 0
        pre_ymd = h['ymd']
        for p in sort_list:
            if h['ymd'] > p[0] and p[0] > '20150101':
                idx += 1
                # 過去のレース情報取得
                rp = self.racekey[p[1]['key']]

                if len(rp['horse']) > 18:
                    print(f"{rp['key']} horse_len={len(rp['horse'])}")

                if first and len(pv) == 0 and p[1]['kakuteijyuni'] > 0 and self.get_baba(rp) != 0 and 'baset' in p[1].keys():
                    interval = self.get_ymd(h['ymd']) - self.get_ymd(p[1]['ymd'])
                    interval /= 400.0
                    first = False

                _,jokencd = self.get_age_joken(rp)
                if (self.get_rtype(r['trackcd']) == self.get_rtype(rp['trackcd'])) and (use_701 or jokencd != 701) and p[1]['kakuteijyuni'] > 0 and self.get_baba(rp) != 0 and 'baset' in p[1].keys():
                    pv.append(self.get_adjust_point(r, h, rp, p[1], idx, pre_ymd, interval))
                    padding_mask.append(True)
                    if len(pv) == PAST_RACE:
                        break
                        
                pre_ymd = p[1]['ymd']
                            
        # 過去レースが無ければ評価できない
        if len(pv) == 0 and h['ninki'] == 1 or not is_training:
            return None
        #_,jokencd = self.get_age_joken(r)
        #if len(pv) == 0 and jokencd == 703 and h['odds'] < 4.0 and not is_training:
        #    return None
        #if len(pv) == 0 and jokencd != 703 and h['odds'] < 2.6 and not is_training:
        #    return None
            
        # 過去レースが7未満の場合は0データで埋める
        if len(pv) < PAST_RACE:
            for i in range(PAST_RACE - len(pv)):
                pv.append([0.0]*INFO_LEN)
                padding_mask.append(False)
 
        return (pv, padding_mask)
        
    def evaluate_one_race(self, r, is_training=True, use_701=True):
        hv = []
        odds = []
        result = []
        running = []
        padding = []

        no_cnt = 0
        # レースrに出走している全馬データを抽出
        for hkey in r['horse']:
            h = self.horsekey[hkey][r['ymd']]
            pt = self.evaluate_horse(r, h, use_701, is_training)
            if pt is None:
                no_cnt += 1
                p = []
                for _ in range(PAST_RACE):
                    p.append([0.0]*INFO_LEN)
                running.append(0)
                padding_mask = [False] * PAST_RACE
            else:
                running.append(1)
                p, padding_mask = pt

            hv.append(p)
            padding.append(padding_mask)
            odds.append(h['odds'])
            result.append(h['kakuteijyuni'])

        # 評価可能な馬が5頭未満は対象外、学習中は1頭でもいたら対象外
        if len(hv) - no_cnt < 5 or (is_training and no_cnt > 0):
            return None

        h_num = len(hv)
        for i in range(18-h_num):
            tmp = []
            for _ in range(PAST_RACE):
                tmp.append([0.0]*INFO_LEN)
            hv.append(tmp)
            odds.append(0.0)
            result.append(0)
            running.append(0)
            padding.append([False] * PAST_RACE)

        hv = np.asarray(hv, dtype=np.float32)
        odds = np.asarray(odds, dtype=np.float32)
        result = np.asarray(result, dtype=np.int32)
        running = np.asarray(running, dtype=np.int32)
        padding = np.asarray(padding)

        return (hv, odds, result, running, padding)

    def evaluate_all_race(self):
        dat = []
        label = []
        tan = []
        running_mask = []
        padding_mask = []

        for rkey in self.racelist:
            # 新馬と2歳オープン未満は除外 2026/9/13 Update
            r = self.racekey[rkey]
            age,jokencd = self.get_age_joken(r)
            if r['ymd'] >= self.fromdate and r['ymd'] <= self.todate and jokencd != 701:# and (age > 0 or jokencd == 999):
                x = self.evaluate_one_race(r)
                if x is not None:
                    cur_dat, odds, result, running, padding = x
                    if sum(result) == 0:
                        print(f"レース結果無し {r['ymd']} {r['jyocd']} {r['racenum']}R")
                        continue
                    
                    cur_result = np.where(result==1)[0][0]
                    cur_tan = odds[cur_result]
                    cur_dat = np.asarray(cur_dat, dtype=np.float32)
                    cur_tan = np.asarray(cur_tan, dtype=np.float32)

                    dat.append(cur_dat)
                    label.append(cur_result)
                    tan.append(cur_tan)
                    running_mask.append(running)
                    padding_mask.append(padding)

        return dat, label, tan, running_mask, padding_mask, self.racelist

class RaceDataAugmentation:
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std
    
    def __call__(self, horse_info, is_training=True):
        #if not is_training:
        if True:
            return horse_info
        
        # 連続値特徴にノイズを追加
        continuous_mask = [
            0,0,                    # gp1 2
            1,1,1,1,                # gp2 4
            0,0,0,0,0,0,0,0,0,      # gp3 8
            1,1,1,1,1,1,1,1,1,      # gp4 9
            0,0,0,0,0,              # gp5 5
            1,1,1,1,1,              # gp6 5
            1,1,1,1,1,1,1,          # gp7 9
                           
        ] # gp2, gp4, gp6, gp7の位置

        if len(continuous_mask) != len(horse_info[0][0]):
            print(f'diff noise mask length! noise={len(continuous_mask)} info={len(horse_info[0][0])}')

        noise = torch.randn_like(horse_info[..., continuous_mask]) * self.noise_std
        horse_info[..., continuous_mask] += noise
        horse_info = torch.clip(horse_info, 0.0, 1.0)
        
        return horse_info
    
class MagokoroDataset(Dataset):
    def __init__(self, fromdate='20140101', todate='20181231', is_training=True):
        m = Magokoro(fromdate, todate)
        dat, label, tan, running_mask, padding_mask, racelist = m.evaluate_all_race()
        self.dat = dat
        self.label = label
        self.tan = tan
        self.running_mask = running_mask
        self.padding_mask = padding_mask
        self.racelist = racelist
        self.is_training = is_training
        self.random_noise = RaceDataAugmentation()

        #self.report_dat()

    def report_dat(self):
        num_rows = (INFO_LEN+7) // 8
        num_cols = 8
        # fig: Figureオブジェクト (ウィンドウ全体)
        # axes: Axesオブジェクトの配列 (各グラフ領域)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 2.5))
        # グリッド間のスペースを自動調整
        plt.tight_layout()

        # 3. 各列（44次元）ごとにループ処理を行い、ヒストグラムを描画
        # axesは2次元配列として返されるため、reshape(-1)で1次元に平坦化すると扱いやすいです。
        axes_flat = axes.reshape(-1)

        npdat = np.array(self.dat)
        npdat = npdat.reshape((len(self.dat)*18*7, INFO_LEN))
        isvalid = np.array(self.padding_mask)
        isvalid = isvalid.reshape((len(isvalid)*18*7))
        histdat = npdat[isvalid]
        maxdat = histdat.max(axis=0)
        mindat = histdat.min(axis=0)
        print('# shape=',histdat.shape)
        for i in range(INFO_LEN):
            print(f'# {i}: min= {mindat[i]:.2f} max= {maxdat[i]:.2f}')

            ax = axes_flat[i]
            # 各列のデータを取得し、ヒストグラムを描画
            ax.hist(histdat[:, i], bins=30, color='skyblue', edgecolor='black')
            ax.set_title(f'Column {i}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, linestyle='--', alpha=0.6)

        for j in range(npdat.shape[1], len(axes_flat)):
            fig.delaxes(axes_flat[j])

        # 5. グラフを表示
        plt.show()

    def __getitem__(self, index):
        horse_info = self.dat[index]
        horse_info = torch.FloatTensor(horse_info)  # (18, 7, 86)
        winner_idx = self.label[index]  # (18,)
        winner_idx = torch.LongTensor([winner_idx])  # (18,)
        odds = torch.FloatTensor(self.tan[index])  # (18,)
        running_mask = self.running_mask[index]
        running_mask = torch.LongTensor(running_mask)  # (18,)
        padding_mask = self.padding_mask[index]
        padding_mask = torch.BoolTensor(padding_mask)  # (18,7)

        horse_info = self.random_noise(horse_info, is_training=self.is_training) # 連続値特徴にノイズを追加

        return horse_info, winner_idx.squeeze(), odds, running_mask, padding_mask

    def __len__(self):
        return len(self.dat)
