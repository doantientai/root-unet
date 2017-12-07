###################################################
#
#   Script to execute the prediction
#
##################################################

import os, sys
import ConfigParser
import smtplib

#config file to read from
config = ConfigParser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))
#===========================================
#name of the experiment!!
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('testing settings', 'nohup')   #std output on log file?

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

#create a folder for the results if not existing already
result_dir = "Experiments/" + name_experiment
print "\n1. Create directory for the results (if not already existing)"
if os.path.exists(result_dir):
    pass
elif sys.platform=='win32':
    os.system('md ' + result_dir)
else:
    os.system('mkdir -p ' + result_dir)


# finally run the prediction
if nohup:
    print "\n2. Run the prediction on GPU  with nohup"
    os.system(run_GPU +' nohup python -u ./src/retinaNN_predict.py > ' + './' +result_dir +'/'+name_experiment+'_prediction.nohup')
else:
    print "\n2. Run the prediction on GPU (no nohup)"
    os.system(run_GPU +' python ./src/retinaNN_predict.py')

    
def sendEmail(addr_from, password, addr_to, msg):
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.ehlo()
    server.starttls()
    server.login(addr_from,password)
    server.sendmail(addr_from, addr_to, msg)
    server.quit()
    
sendEmail('doantientaipc@gmail.com','DoanTienTai','doantientai@gmail.com', "The test " + name_experiment + ' is Done!')
