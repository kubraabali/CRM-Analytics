##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

df_= pd.read_csv("datasets/flo_data_20k.csv")
df= df_.copy()
pd.set_option("display.max_columns",500)
pd.set_option("display.width",500)


# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = (quartile3 + 1.5 * interquantile_range).round()
    low_limit = (quartile1 - 1.5 * interquantile_range).round()
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.

date= ["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online"]
for i in date:
    replace_with_thresholds(df,i)

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df.head()

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes # kullanacağımız tarih değişkenleri kategorik
df["first_order_date"] =df["first_order_date"].apply(pd.to_datetime)
df["last_order_date"] =df["last_order_date"].apply(pd.to_datetime)

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max()
today_date = dt.datetime(2021,7,2)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
df_cltv = pd.DataFrame({"customer_id": df["master_id"],
                       "recency_cltv_weekly": ((df["last_order_date"]-df["first_order_date"]).dt.days)/7,
                       "T_weekly": ((df["last_order_date"]-df["first_order_date"]).dt.days) /7,
                       "frequency": df["order_num_total"],
                       "monetary_cltv_avg" : df["customer_value_total"]/df["order_num_total"]
                       })
df_cltv.head()
df.head()

# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 1. BG/NBD modelini fit ediniz.

bgf= BetaGeoFitter(penalizer_coef=0.001)
#frequency, recency ve müşteri yaşı değerleri
bgf.fit(df_cltv["frequency"]
        ,df_cltv["recency_cltv_weekly"],
        df_cltv["T_weekly"])

# a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve  olarak cltv dataframe'ine ekleyiniz.
df_cltv["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(12,
                                                                                       df_cltv["frequency"],
                                                                                       df_cltv["recency_cltv_weekly"],
                                                                                       df_cltv["T_weekly"])
df_cltv.sort_values("exp_sales_3_month", ascending=False)

# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

df_cltv["cltv"] = bgf.conditional_expected_number_of_purchases_up_to_time(24,
                                                                          df_cltv["frequency"],
                                                                          df_cltv["recency_cltv_weekly"],
                                                                          df_cltv["T_weekly"])
(df_cltv.sort_values("exp_sales_6_month", ascending=False)).head(20)

# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(df_cltv['frequency'], df_cltv['monetary_cltv_avg'])
df_cltv["exp_average_value"] = ggf.conditional_expected_average_profit(df_cltv['frequency'], df_cltv['monetary_cltv_avg'])
df_cltv.head()

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   df_cltv['frequency'],
                                   df_cltv['recency_cltv_weekly'],
                                   df_cltv['T_weekly'],
                                   df_cltv['monetary_cltv_avg'],
                                   time=6, #aylık
                                   freq="W", # haftalık bilgi week
                                   discount_rate=0.01) #kampanya etkisi
df_cltv["cltv"] = cltv

df_cltv.head()


# b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
df_cltv.sort_values("cltv",ascending=False)[:20]

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması

# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
# 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

df_cltv["segment"] = pd.qcut(df_cltv["cltv"],4, labels = ["D","C","B","A"])
df_cltv.head()

