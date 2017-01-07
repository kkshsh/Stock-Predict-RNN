from dataapiclient import Client

client = Client()
client.init('your token')
url1='/api/market/getMktEqud.json?field=&beginDate=&endDate=&secID=&ticker=&tradeDate=20150513'
code, result = client.getData(url1)
if code == 200:
    print(result)
else:
    print(code)
    print(result)
url2='/api/subject/getThemesContent.csv?field=&themeID=&themeName=&isMain=1&themeSource='
code, result = client.getData(url2)
if(code==200):
    file_object = open('thefile.csv', 'w')
    file_object.write(result)
    file_object.close( )
else:
    print(code)
    print(result)
