'''
A simple demo of ASL_detection in terminal.
'''

from lib import ASLDetection

try:
    ad = ASLDetection()
    end_flg = ''
    while True:
        end_flg = input('Process [y/n]: ')
        if end_flg == 'n':
            break
        msg = ad.start_video()
        print(msg)
except Exception as e:
    print(e)