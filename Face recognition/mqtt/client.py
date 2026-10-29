
import time
import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883

# Use the same credentials configured in your ESP8266 sketch.
MQTT_USER = "YOUR_MQTT_USERNAME"
MQTT_PASS = "YOUR_NEW_MQTT_PASSWORD"

TOPIC_ANGLE = "tachometer/angle"

_connected = False

client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    client_id="face_tracking_controller",
)


def on_connect(client, userdata, flags, reason_code, properties):
    global _connected
    _connected = not reason_code.is_failure

    if _connected:
        print("Connected to MQTT broker.")
    else:
        print(f"MQTT connection failed: {reason_code}")


def on_disconnect(
    client, userdata, disconnect_flags, reason_code, properties
):
    global _connected
    _connected = False
    print("Disconnected from MQTT broker.")


client.on_connect = on_connect
client.on_disconnect = on_disconnect


def connect_mqtt():
    global _connected

    try:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
        client.loop_start()

        for _ in range(50):
            if _connected:
                return True
            time.sleep(0.1)

        print("MQTT broker did not confirm the connection.")
        return False

    except Exception as error:
        print(f"MQTT connection error: {error}")
        return False


def publish_angle(angle):
    """Publish a numeric tachometer angle between 0 and 180 degrees."""
    if not _connected:
        print("Cannot publish: MQTT is not connected.")
        return False

    angle = max(0.0, min(180.0, float(angle)))

    result = client.publish(
        TOPIC_ANGLE,
        f"{angle:.1f}",
        qos=1,
    )

    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        print("Failed to publish angle.")
        return False

    return True


def publish_pan_command(command):
    """
    Compatibility with the existing tracker.
    STOP means center the tachometer using the numeric protocol.
    """
    command = command.strip().upper()

    if command == "STOP":
        return publish_angle(90)

    print(
        f"Unsupported text command {command!r}; "
        "publish an angle with publish_angle() instead."
    )
    return False


def disconnect_mqtt():
    global _connected

    try:
        if _connected:
            publish_angle(90)

        client.disconnect()
        client.loop_stop()
        _connected = False
        print("MQTT client stopped.")

    except Exception as error:
        print(f"MQTT disconnect error: {error}")