clear all
clc

imgs = imageDatastore("UnlabeledImages\");

imageLabeler(imgs)