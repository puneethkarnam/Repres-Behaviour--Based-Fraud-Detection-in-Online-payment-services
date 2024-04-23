from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class fraud_detection_type(models.Model):

    step= models.CharField(max_length=300)
    Product_name= models.CharField(max_length=3000)
    type= models.CharField(max_length=300)
    amount= models.CharField(max_length=300)
    nameOrig= models.CharField(max_length=300)
    oldbalanceOrg= models.CharField(max_length=300)
    newbalanceOrig= models.CharField(max_length=300)
    nameDest= models.CharField(max_length=300)
    oldbalanceDest= models.CharField(max_length=300)
    newbalanceDest= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



