
        ###############################################################
        # RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
        ###############################################################

        ###############################################################
        # İş Problemi (Business Problem)
        ###############################################################
        # FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
        # Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

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

        # GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
                   # 1. flo_data_20K.csv verisini okuyunuz.
                   # 2. Veri setinde
                             # a. İlk 10 gözlem,
                             # b. Değişken isimleri,
                             # c. Betimsel istatistik,
                             # d. Boş değer,
                             # e. Değişken tipleri, incelemesi yapınız.
                   # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
                   # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
                   # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
                   # 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
                   # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
                   # 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
                   # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.

        import pandas as pd
        import datetime as dt
        df_= pd.read_csv("datasets/flo_data_20k.csv")
        df= df_.copy()
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width",500)
        df.head(10)
        df.columns
        df.describe().T #aykırı değerler var, treshold!
        df.isnull().sum() #boş değer yok
        df.dtypes #gün değişkenleri kategorik, recency için sorun oluşabilir!

        # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş
        # yaptığını ifade etmektedir. Herbir müşterinin
        # toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
        # order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
        # order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
        rfms = pd.DataFrame()
        rfms = df.groupby("master_id").agg({"order_num_total_ever_online" : "sum"})
        df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
        rfms["order_num_total_ever_offline"] = df.groupby("master_id").agg({"order_num_total_ever_offline" : "sum"})
        rfms["order_num"] = rfms["order_num_total_ever_offline"] + rfms["order_num_total_ever_online"]

        # customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
        # customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
        df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
        rfms["total_value"] = df.groupby("master_id").agg({"total_value" : "sum"})

        rfms.head()

        #4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

        df.head()
        df.dtypes
        df["first_order_date"] =df["first_order_date"].apply(pd.to_datetime)

        df["last_order_date"] =df["last_order_date"].apply(pd.to_datetime)

        df["last_order_date_online"] =df["last_order_date_online"].apply(pd.to_datetime)

        df["last_order_date_offline"] =df["last_order_date_offline"].apply(pd.to_datetime)


        # 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
        df.groupby("order_channel").agg({"master_id": "count",
                                         "total_order_num" : "mean",
                                         "total_value": "mean" })

        # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

        new_df = df.sort_values("total_value",ascending=False)[:10]

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
new= df.sort_values("total_order_num", ascending=False)[:10]

# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
        def rfm(df):
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 500)
            print("*********Veri Seti***********")
            print(df.head(10))
            print("******************Sütun İsimleri***************************")
            print(df.columns)
            print("*************Sayısal Değerler******************")
            print(df.describe().T)
            print("Boş değerler var mı ?:")
            print(df.isnull().sum())
            print("*******************Veri Tipleri******************")
            print(df.dtypes)
            df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
            df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
            df["first_order_date"] = df["first_order_date"].astype("datetime64[ns]")
            df["last_order_date"] = df["last_order_date"].astype("datetime64[ns]")
            df["last_order_date_online"] = df["last_order_date_online"].astype("datetime64[ns]")
            df["last_order_date_offline"] = df["last_order_date_offline"].astype("datetime64[ns]")
            print("*******************Veri Tipleri******************")
            print(df.dtypes)
            print("******************* Kanallarımız ******************")
            kanal = df.groupby("order_channel").agg({"master_id": "count",
                                                     "total_order_num": "mean",
                                                     "total_value": "mean"})

            print(kanal)
            ttlvalue = df.sort_values("total_value",ascending=False)[:10]
            ttlorder = df.sort_values("total_order_num", ascending=False)[:10]
            print("****************En Çok Harcayanlar******************")
            print(ttlvalue)
            print("****************En Çok Alışveriş Yapanlar******************")
            print(ttlorder)

prepro(df)



# GÖREV 2: RFM Metriklerinin Hesaplanması
df.head()
today_date= dt.datetime(2021,7,1)
df["last_order_date"].max()
df["recency"] = (today_date- df["last_order_date"])
df = df.rename(columns={"total_order_num": "frequency", "total_value": "monetary"}, errors="raise")
rfm= pd.DataFrame()
rfm= df[["master_id","recency","frequency","monetary"]]

rfm.head

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması
rfm["recency_score"]= pd.qcut(rfm["recency"],5, labels= [5,4,3,2,1])
rfm["frequency_score"]= pd.qcut(rfm["frequency"].rank(method="first"),5, labels=[1,2,3,4,5])
rfm["monetary_score"]=pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])
rfm["recency"].max()
rfm["RFScore"]= (rfm["recency_score"].astype(str)+rfm["frequency_score"].astype(str))
rfm["RFMScore"] = (rfm["recency_score"].astype(str)+rfm["frequency_score"].astype(str)+rfm["monetary_score"].astype(str))


# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm['segment'] = rfm['RFScore'].replace(seg_map, regex=True)
df["segment"] = rfm["segment"]
rfm.head()
rfm_summary= rfm[["master_id", "recency", "frequency", "monetary", "segment"]]

#loyal_customer
customer = pd.DataFrame()

###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm["recency"].mean()
rfm.groupby("segment").agg({"recency":"mean",
                            "frequency": "mean",
                            "monetary":"mean"})


# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.
rfm.head()
df.head()
yeni_marka_hedef_müşteri_id= df.loc[(df["segment"]== "loyal_customers") |  (df["segment"]== "champions")  & (df["interested_in_categories_12"].str.contains("KADIN"))]
yeni_marka_hedef_müşteri_id= yeni_marka_hedef_müşteri_id["master_id"]

yeni_marka_hedef_müşteri_id.to_csv("yeni_marka_hedef_müşteri_id.csv")

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
target=['hibernating', 'cant_loose', "about_to_sleep"]
indirim_hedef_müşteri_ids = df[(df["segment"].isin(target)) & df["interested_in_categories_12"].str.contains("ERKEK")|df["interested_in_categories_12"].str.contains("COCUK")]["master_id"]

indirim_hedef_müşteri_ids.to_csv("indirim_hedef_müşteri_ids.cvs")
