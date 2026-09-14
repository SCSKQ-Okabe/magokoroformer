import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from magokorodataset2026 import *
from magokoroformer2026 import *
import torch.nn.functional as F
import itertools
import psycopg2
import psycopg2.extras

LOG_FILE = 'report.csv'
MODEL_FN = '28-148-2cond.pth'

def logprint(s):
    with open(LOG_FILE, mode='a') as f:
        f.write(s+'\n')

def logprintHead(s):
    with open(LOG_FILE, mode='w') as f:
        f.write(s+'\n')

def predict(fromdate, todate):
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # モデル作成
    model = ImprovedHorseRacingTransformer()
    
    print(f'load model file : {MODEL_FN}')
    model.load_state_dict(torch.load(MODEL_FN))
    model = model.to(device)
    model.eval()

    csvhead = 'ymd,jyo,racenum,age,joken,track,kyori,baba,tosu,'
    csvhead += 'tan,fuk1,fuk2,fuk3,umaren,umatan,3renpuku,3rentan,wide1,wide2,wide3,'
    csvhead += 'y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,'
    csvhead += 'winrate,topodds'
    
    logprintHead(csvhead)

    m = Magokoro(fromdate, todate)

    with torch.no_grad():
        for rkey in m.racelist:
            raceinfo = m.racekey[rkey]
            age, joken = m.get_age_joken(raceinfo)
            if raceinfo['ymd'] >= fromdate and raceinfo['ymd'] <= todate and raceinfo.get('paytansyopay') is not None and joken != 701:
                x = m.evaluate_one_race(raceinfo)
                if x is not None:
                    horse_info, odds, result, running_mask, padding_mask = x
                    horse_info = torch.FloatTensor(horse_info)
                    running_mask = torch.LongTensor(running_mask)
                    padding_mask = torch.BoolTensor(padding_mask)

                    horse_info = horse_info.unsqueeze(0)
                    running_mask = running_mask.unsqueeze(0)
                    padding_mask = padding_mask.unsqueeze(0)

                    horse_info = horse_info.to(device)
                    running_mask = running_mask.to(device)
                    padding_mask = padding_mask.to(device)
                    predictions = model(horse_info, running_mask, padding_mask)

                    i_list = torch.argsort(predictions, dim=1, descending=True)
                    i_list = i_list.cpu().numpy()[0]

                    csvline = f"{raceinfo['ymd']},{raceinfo['jyocd']},{raceinfo['racenum']},"
                    print(csvline)
                    #age= int(raceinfo['syubetucd'])
                    #joken = int(raceinfo['jyokencd1']) + int(raceinfo['jyokencd2']) + int(raceinfo['jyokencd3'])
                    csvline += f"{age},{joken},"
                    csvline += f"{int(raceinfo['trackcd'])},{raceinfo['kyori']},"
                    csvline += f"{int(raceinfo['sibababacd'])+int(raceinfo['dirtbabacd'])},{len(raceinfo['horse'])},"
                    
                    csvline += f"{raceinfo['paytansyopay']},{raceinfo['payfukusyopay1']},{raceinfo['payfukusyopay2']},"
                    csvline += f"{raceinfo['payfukusyopay3'] if raceinfo.get('payfukusyopay3') else 0},"
                    csvline += f"{raceinfo['payumarenpay']},{raceinfo['payumatanpay']},{raceinfo['paysanrenpukupay']},"
                    csvline += f"{raceinfo['paysanrentanpay']},{raceinfo['paywidepay1']},{raceinfo['paywidepay2']},{raceinfo['paywidepay3']},"
                    
                    cnt5 = []
                    for i in range(len(raceinfo['horse'])):
                        h = m.horsekey[raceinfo['horse'][i_list[i]]][raceinfo['ymd']]
                        csvline += f"{h['kakuteijyuni']},"
                        if len(cnt5) < 5:
                            cnt5.append(int(h['umaban']))

                    for i in range(18 - len(raceinfo['horse'])):
                        csvline += "0,"

                    confidence = torch.softmax(predictions, dim=1)
                    csvline += f"{confidence[0][i_list[0]]:.3f},{m.horsekey[raceinfo['horse'][i_list[0]]][raceinfo['ymd']]['odds']},"

                    logprint(csvline)

# ===== メイン実行 =====
if __name__ == "__main__":
    predict('20250101', '20251231')
