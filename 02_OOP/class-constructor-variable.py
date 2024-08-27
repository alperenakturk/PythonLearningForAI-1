# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

class Calisan:
    zam_orani = 1.8
    counter = 0
    
    def __init__ (self,isim,soyisim,maas): #constructor
        self.isim = isim
        self.soyisim = soyisim
        self.maas = maas
        self.email = isim+soyisim+"@asd.com"
        
        Calisan.counter = Calisan.counter + 1
    
    def giveNameSurname(self):
        return self.isim +" " +self.soyisim
    
    def zam_yap(self):
        self.maas = self.maas + self.maas*self.zam_orani
        
        
#isci1 = Calisan("ali", "veli",100) 
#print(isci1.maas)
#print(isci1.giveNameSurname())

# class variable
calisan1 = Calisan("ali", "veli",100) 
print("ilk maas: ",calisan1.maas)
calisan1.zam_yap()
print("yeni maas: ",calisan1.maas)

calisan2 = Calisan("ayse", "hatice",200) 
calisan3 = Calisan("ayse", "yelda",600) 
calisan4 = Calisan("eren", "hilal",500) 

