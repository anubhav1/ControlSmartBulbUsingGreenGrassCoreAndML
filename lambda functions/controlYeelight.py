import logging
import sys
from yeelight import Bulb
from yeelight.flows import *
import greengrasssdk

bulb = Bulb("192.168.1.100")

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Create MQTT client
mqtt_client = greengrasssdk.client('iot-data')

def send_mqtt_message(message):
    logging.info("Publishing")
    mqtt_client.publish(topic='bulb/confirmation', payload=message)
    

def get_input_message(event):
    try:
        message = event['Command']
        logging.info("Message Received : " + message)
    except Exception as e:
        logging.error('Message could not be parsed. ' + repr(e))
    return message

# The function to be triggered on subscription of a topic in GreenGrass Group
def handler(event, context):
    try:
        command = get_input_message(event)
        if command == "OFF":
            bulb.turn_off()
            send_mqtt_message('Turned off.')
            
        elif command == "ON":
            bulb.turn_on()
            send_mqtt_message('Turned on.')
        
        elif command == "RED":
            bulb.set_rgb(255, 0, 0)
            send_mqtt_message('Changed to Red.')
        
        elif command == "GREEN":
            bulb.set_rgb(0, 255, 0)
            send_mqtt_message('Changed to Green.')
        
        elif command == "BLUE":
            bulb.set_rgb(0, 0, 255)
            send_mqtt_message('Changed to Blue.')
        
        elif command == "DISCO":
            bulb.start_flow(disco())
            send_mqtt_message('Changed to Disco.')
        
        elif command == "STROBE":
            bulb.start_flow(strobe_color())
            send_mqtt_message('Changed to Strobe.')
        
        elif command == "POLICE":
            bulb.start_flow(police())
            send_mqtt_message('Changed to Police.')
        
        elif command == "MOVIE":
            bulb.start_flow(movie()) 
            send_mqtt_message('Changed to Movie.')
        
        elif command == "CANDLE":
            bulb.start_flow(candle_flicker())
            send_mqtt_message('Changed to Candle.')
            
        elif command == "ROMANCE":
            bulb.start_flow(romance())  
            send_mqtt_message('Changed to Romance.')
            
        else:
            send_mqtt_message(command +' not recognised')
        
    except Exception as e:
        logger.exception(e)
        send_mqtt_message('Exception occurred while changing the state of the bulb.. Please check logs for troubleshooting: /greengrass/ggc/var/log.')