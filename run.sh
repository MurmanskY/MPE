# Entry
# run initial model accuracy test process
# if you want to run, uncomment the following codes
#cd ./models/pretrainedModels/
#python resnet18.py > resnet18_output.txt
#python resnet50.py > resnet50_output.txt
#python resnet101.py > resnet101_output.txt
#python vgg11.py > vgg11_output.txt
#python vgg13.py > vgg13_output.txt
#python vgg16.py > vgg16_output.txt
#python vgg19.py > vgg19_output.txt
#cd ../../




# run embedded model accuracy test process
# if you want to run, uncomment the following codes
cd ./models/embeddedModels/
#python resnet18.py > resnet18_output.txt
#python resnet50.py > resnet50_output.txt
#python resnet101.py > resnet101_output.txt
#python vgg11.py > vgg11_output.txt
python vgg13.py > vgg13_output.txt
#python vgg16.py > vgg16_output.txt
python vgg19.py > vgg19_output.txt
cd ../../