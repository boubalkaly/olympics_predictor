import pandas as pd


def custom_to_timedelta(time_str):
    if 'h' in time_str:
        hours, remainder = time_str.split('h')
        hours = int(hours)
        print(pd.to_timedelta(f'{hours}:{remainder}'))
    else:
        print(pd.to_timedelta(time_str))


custom_to_timedelta('02h10:31')
