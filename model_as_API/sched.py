from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
import pandas as pd
import dill

scheduler = BlockingScheduler(timezone=tzlocal.get_localzone_name())

df = pd.read_csv('model/data/homework.csv')
file_name = 'cars_best_pipe.pkl'
with open(file_name, 'rb') as file:
    model = dill.load(file)

@scheduler.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(n=5)
    data['predicted_price_cat'] = model['model'].predict(data)
    data['predicted_price_cat'] = data['predicted_price_cat'].apply(lambda x: 'low' if x == 0 else ('medium' if x == 1 else 'high'))

    print(data[['id', 'price', 'predicted_price_cat']])


if __name__ == '__main__':
    scheduler.start()